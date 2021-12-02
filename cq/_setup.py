import pickle
import nltk
from re import finditer
import re
import os
from lemminflect import getLemma
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

import tasks
import zacq
import kwcq

from nltk.corpus import stopwords
stop = list(stopwords.words('english'))
stop.extend("return returns param params parameter parameters code class inheritdoc".split())

LANGUAGES = ('python', 'go', 'javascript', 'java', 'php', 'ruby')
NUM_KEYWORDS=25
NUM_COMPONENTS = 50

csn_path = os.path.join("..", "CodeSearchNet")
data_path = os.path.join("..", "data")

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
    
def create_document(definition):
    identifier = definition["identifier"].split(".")[-1]
    identifier = " ".join(camel_case_split(identifier))
    identifier = " ".join(identifier.split("_")) + "."
    docstring = definition["docstring"]
    return " ".join([identifier, docstring]).strip()
    
def camel_case_split(identifier):
        matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]

task_extractor = tasks.TaskExtractor()
code_tokenizer = CodeTokenizer()

queries_path = os.path.join(csn_path, "resources", "queries.csv")
with open(queries_path, "r") as q_file:
    queries = [line.strip() for line in q_file if line.strip()][1:]

for language in LANGUAGES:    
    print(f"Loading {language} definitions and results...")
    definitions_path = os.path.join(csn_path, "resources", "data", f"{language}_dedupe_definitions_v2.pkl")
    definitions = pickle.load(open(definitions_path, "rb"))
    results_path = os.path.join(data_path, "results", f"{language}_results.pkl")
    results = pickle.load(open(results_path, "rb"))
    
    print(f"Creating task dataframes for {language} search results...")
    task_dfs = {}
    for query in tqdm(queries):
        query_results = results[query][0]
        docstrings = [create_document(definitions[result_index]) for result_index in query_results]
        task_df = zacq.create_task_df(docstrings, query_results, task_extractor)
        task_dfs[query]=task_df
    pickle.dump(task_dfs, open(os.path.join(data_path, "tasks", f"{language}_task_dfs.pkl"), "wb"))
                
    print(f"Running SVD on {language}...")
    keyword_data = {}
    documents = [create_document(definition) for definition in definitions]
    # Vectorize document using TF-IDF
    tfidf = TfidfVectorizer(lowercase=False,
                            stop_words=stop,
                            ngram_range = (1,1),
                            tokenizer = code_tokenizer.tokenize,
                            max_df = .6,
                            min_df = 2)
    # Fit and Transform the documents
    train_data = tfidf.fit_transform(documents)   
    terms = tfidf.get_feature_names()
    keyword_data["terms"] = terms
    keyword_data["train_data"] = train_data
    # Create SVD object
    lsa = TruncatedSVD(n_components=NUM_COMPONENTS, n_iter=25, random_state=42)
    # Fit SVD model on data
    svd_function_vectors = lsa.fit_transform(train_data)
    svd_keyword_vectors = lsa.components_.T
    keyword_data["fvectors"] = svd_function_vectors
    keyword_data["kvectors"] = svd_keyword_vectors
    pickle.dump(keyword_data, open(os.path.join(data_path, "keyword_data", f"{language}_keyword_data.pkl"), "wb"))
