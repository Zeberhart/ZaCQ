import json as js
import time
import traceback
import sys
import argparse
import os
import nltk
import threading
import asyncio
import pickle 
import nltk
import re
import concepts
import websockets
import traceback
import pandas as pd
import numpy as np

from lemminflect import getLemma
from re import finditer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import pairwise
from sanic import Sanic
from sanic.response import file
from sanic.response import json, text

from nltk.corpus import stopwords
stop = list(stopwords.words('english'))
stop.extend("return returns param params parameter parameters code class inheritdoc".split())

rootdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..")
sys.path.append(os.path.join(rootdir, "cq"))

import zacq
import kwcq
import tasks

application = Sanic(name="SearchAPI")
language = "java"

###ZACQ
cq_cache={}
task_extractor = tasks.TaskExtractor()

def generate_zacq(indices, identifiers, docstrings, set_values, not_values, query):
    task_df = None
    new_indices = []
    documents = []
    for index, identifier, docstring in zip(indices, identifiers, docstrings):
        if index in cq_cache:
            if task_df is None:
                task_df = cq_cache[index]
            else:
                task_df=task_df.append(cq_cache[index])
        else:
            new_indices.append(index)
            documents.append(create_document(identifier, docstring))
    if documents:
        new_task_df = zacq.create_task_df(documents, new_indices, task_extractor)
        for index in new_indices:
            cq_cache[index] = new_task_df.loc[new_task_df["row"]==index]
        if task_df is None:
            task_df = new_task_df
        else:
            task_df=task_df.append(new_task_df)
    question, inferred_values, target_slot, answers = zacq.generate_cq(task_df = task_df, 
                                                                       set_values=set_values, 
                                                                       not_values=not_values,
                                                                       query=query)
    
    #Get set of rejectable functions
    task_df = zacq.apply_set_values(task_df, set_values)
    filtered_rows = set()
    current_not_values = inferred_values.copy()
    if target_slot:
        if target_slot != "role":
            for answer in answers:
                current_not_values[target_slot] = answer
                not_values.append(current_not_values.copy())
        else:
            for answer in answers:
                verb, do, prep, po = answer
                current_not_values["verb"] = verb
                if do:
                    current_not_values["direct_object"] = do
                else:
                    current_not_values["preposition"] = prep
                    current_not_values["preposition_object"] = po
                not_values.append(current_not_values.copy())
                current_not_values = inferred_values.copy()
    else:
        not_values.append(current_not_values) 
    task_df, diff_rows = zacq.filter_not_dicts(task_df, not_values)
    diff_rows = [int(v) for v in diff_rows]
    reject_candidates = task_df["row"].unique()
    reject_candidates = [int(v) for v in reject_candidates]
    
    if target_slot=="role":
        answers = {",".join([word if word else "" for word in k]): [int(v) for v in answers[k]] for k in answers}
    else: 
        answers = {k: [int(v) for v in answers[k]] for k in answers}
    return {"type": "cq", "question":question, "inferred_values":inferred_values, "target":target_slot, "answers":answers, "rejectables":diff_rows, "reject_candidates":reject_candidates}



###KWCQ
kw_cache={}
NUM_KEYWORDS=25
print("loading keyword data...")
keyword_data = pickle.load(open("../data/output/keyword_data", "rb"))

class CodeTokenizer():
    w_tokenizer = nltk.tokenize.RegexpTokenizer(r'[^\d\W]+')
    
    def camel_case_split(self, identifier):
        matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]

    def tokenize(self, s):
        tokens = self.w_tokenizer.tokenize(s)
        tokens = sum([self.camel_case_split(t) for t in tokens], [])
        tokens = sum([t.split("_") for t in tokens], [])
        tokens = [t.lower() for t in tokens if len(t)>2]
        tokens = [t for t in tokens if not re.findall("[^\u0000-\u05C0\u2100-\u214F]+", t)]
        for i, token in enumerate(tokens):
            v_t = getLemma(token, "VERB", lemmatize_oov=False)
            n_t = getLemma(token, "NOUN", lemmatize_oov=False)
            if v_t:
                tokens[i] = v_t[0]
            elif n_t:
                tokens[i] = n_t[0]
        return tokens
code_tokenizer = CodeTokenizer()
    

def create_context(result_indices, identifiers, docstrings, query):

    result_vectors = keyword_data[language]["fvectors"][result_indices]
    avg_result_vector = np.average(result_vectors, axis=0).reshape(1, -1)
    other_indices = list(set(range(len(keyword_data[language]["fvectors"]))).difference(set(result_indices)))
    other_vectors = keyword_data[language]["fvectors"][other_indices]
    avg_other_vector = np.average(other_vectors, axis=0).reshape(1, -1)
    result_term_similarities = pairwise.cosine_similarity(avg_result_vector, keyword_data[language]["kvectors"])[0]
    other_term_similarities = pairwise.cosine_similarity(avg_other_vector, keyword_data[language]["kvectors"])[0]
    similarity_diff = result_term_similarities-other_term_similarities

    term_indices_in_results = list(set(keyword_data[language]["train_data"][result_indices].nonzero()[1]))
    mask = np.full(len(similarity_diff), -np.inf)
    mask[term_indices_in_results] = 0
    similarity_diff+=mask

    query_kws = code_tokenizer.tokenize(query)
    num_keywords = NUM_KEYWORDS+len(query_kws)

    ind = np.argpartition(similarity_diff, -num_keywords)[-num_keywords:]
    ind = ind[np.argsort(similarity_diff[ind])]
    all_query_result_kws = [keyword_data[language]["terms"][i] for i in ind]
    all_query_result_kws = [kw for kw in all_query_result_kws if kw not in query_kws][:NUM_KEYWORDS]
    df_list = []
    for identifier, docstring in zip(identifiers, docstrings):
        document = " ".join(identifier.split(".")+[docstring])
        result_kws = code_tokenizer.tokenize(document)
        d = {kw: kw in result_kws for kw in all_query_result_kws}
        df_list.append(d)
        
    df = pd.DataFrame(df_list)
    df.index=result_indices
    
    df=df.loc[df.any(axis='columns')]
    objects = [str(z) for z in df.index.tolist()]
    properties = list(df)
    bools = list(df.fillna(False).astype(bool).itertuples(index=False, name=None))

    c = concepts.Context(objects, properties, bools)
    return c, all_query_result_kws

def create_document(identifier, docstring):
    identifier = identifier.split(".")[-1]
    identifier = " ".join(camel_case_split(identifier))
    identifier = " ".join(identifier.split("_")) + "."
    return " ".join([identifier, docstring]).strip()
    
    
def camel_case_split(identifier):
        matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]
        
def generate_kwcq(indices, identifiers, docstrings, intents, not_intents, query):
    if query in kw_cache:
        context, query_kws = kw_cache[query]
    else:
        context, query_kws = create_context(indices, identifiers, docstrings, query)
        kw_cache[query] = (context, query_kws)
    
    question, inferred_values, target_slot, answers = kwcq.generate_cq(context, intents, not_intents, 5, query_kws)
    reject_candidates, rejectables = kwcq.get_candidates_rejects(context, intents, not_intents+list(answers))
    return {"type": "kw", "question":question, "inferred_values":inferred_values, "target":target_slot, "answers":answers, "rejectables":rejectables, "reject_candidates":reject_candidates}


@application.websocket('/feed')
async def feed(request, ws):
    while True:
        data = await ws.recv()        
        try:
            data = js.loads(data)
            if data["type"] == "generate-cq":
                indices = [int(i) for i in data["indices"]]
                docstrings = data["docstrings"]
                set_values = data["set_values"]
                identifiers = data["identifiers"]
                query = data["query"]
                not_values = data["not_values"]
                await ws.send(js.dumps(generate_zacq(indices, identifiers, docstrings, set_values, not_values, query)))
            elif data["type"] == "generate-kw":
                indices = [int(i) for i in data["indices"]]
                identifiers = data["identifiers"]
                docstrings = data["docstrings"]
                intents = data["intents"]
                not_intents = data["not_intents"]
                query = data["query"]
                await ws.send(js.dumps(generate_kwcq(indices, identifiers, docstrings, intents, not_intents, query)))
        except Exception as e: 
            print("Error:")
            print(e)
            traceback.print_exc()
            await ws.send(js.dumps({"type":"ERROR", "text":"ERROR"}))
            
async def connect(uri):
    async with websockets.connect(uri) as ws:
        await ws.send(js.dumps({"type": "set-server-role", "role": "clarify"}))
        print("connected")
        while True:
            data = await ws.recv()        
            try:
                data = js.loads(data)
                if data["type"] == "generate-cq":
                    indices = [int(i) for i in data["indices"]]
                    docstrings = data["docstrings"]
                    set_values = data["set_values"]
                    identifiers = data["identifiers"]
                    query = data["query"]
                    not_values = data["not_values"]
                    response = generate_zacq(indices, identifiers, docstrings, set_values, not_values, query)
                    response["cid"] = data["cid"]
                    await ws.send(js.dumps(response))
                elif data["type"] == "generate-kw":
                    indices = [int(i) for i in data["indices"]]
                    identifiers = data["identifiers"]
                    docstrings = data["docstrings"]
                    intents = data["intents"]
                    not_intents = data["not_intents"]
                    query = data["query"]
                    response = generate_kwcq(indices, identifiers, docstrings, intents, not_intents, query)
                    response["cid"] = data["cid"]
                    await ws.send(js.dumps(response))
            except Exception as e: 
                print("Error:")
                print(e)
                traceback.print_exc()
                response = {"type":"ERROR", "text":"ERROR"}
                response["cid"] = data["cid"]
                await ws.send(js.dumps(response))
    
if __name__ == '__main__':
    application.run(host="0.0.0.0", 
                    port=os.environ.get('PORT') or 8003, 
                    debug=True)
        














