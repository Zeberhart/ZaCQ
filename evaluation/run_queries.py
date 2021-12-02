import re
import os
import sys
import pickle
import nltk
import concepts
import numpy as np
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import pairwise
from re import finditer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from lemminflect import getLemma

rootdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
sys.path.append(os.path.join(rootdir, "cq"))
sys.path.append(os.path.join(rootdir, "codesearch"))

import tasks
import zacq
import kwcq
import vocq

NUM_KEYWORDS=25
LANGUAGES = ['java', 'python', 'ruby', 'php', 'javascript']

######################################################################
######################################################################
####################### Load Data ####################################
######################################################################
######################################################################
csn_path = os.path.join("..", "CodeSearchNet")
data_path = os.path.join("..", "data")
model_path = os.path.join(data_path, "models", "neuralbowmodel.pkl.gz")
queries_path = os.path.join(csn_path, "resources", "queries.csv")

with open(queries_path, "r") as q_file:
    queries = [line.strip() for line in q_file if line.strip()][1:]

def load_relevances(filepath):
    relevance_annotations = pd.read_csv(filepath)
    per_query_language = relevance_annotations.pivot_table(
        index=['Query', 'Language', 'GitHubUrl'], values='Relevance', aggfunc=np.mean)

    # Map language -> query -> url -> float
    relevances = defaultdict(lambda: defaultdict(dict))  # type: Dict[str, Dict[str, Dict[str, float]]]
    for (query, language, url), relevance in per_query_language['Relevance'].items():
        relevances[language.lower()][query.lower()][url] = relevance
    return relevances
relevances = load_relevances(os.path.join(csn_path, "resources", "annotationStore.csv"))


definitions = {}
results = {}
all_task_dfs = {}
code_representations = {}
keyword_data={}
for language in LANGUAGES:
    print(f"{language}...")
    print("Loading Definitions")
    definitions_path = os.path.join(csn_path, "resources", "data", f"{language}_dedupe_definitions_v2.pkl")
    definitions[language] = pickle.load(open(definitions_path, 'rb'))
    print("Loading Results")
    results_path = os.path.join(data_path, "results", f"{language}_results.pkl")
    results[language] = pickle.load(open(results_path, "rb"))
    print("Loading Vectors")
    vectors_path = os.path.join(data_path, "vectors", f"{language}_vectors.pkl")
    code_representations[language] = pickle.load(open(vectors_path, "rb"))
    print("Loading Task DFs")
    tasks_path = os.path.join(data_path, "tasks", f"{language}_task_dfs.pkl")
    all_task_dfs[language] = pickle.load(open(tasks_path, "rb"))
    print("Loading Keywords")
    keywords_path=os.path.join(data_path, "keyword_data", f"{language}_keyword_data.pkl")
    keyword_data[language] = pickle.load(open(keywords_path, "rb"))
    
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

def get_document(definition):
    def camel_case_split(identifier):
        matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]
    identifier = definition["identifier"].split(".")[-1]
    identifier = " ".join(camel_case_split(identifier))
    identifier = " ".join(identifier.split("_")) + "."
    docstring = definition["docstring"]
    return " ".join([identifier, docstring]).strip()

code_tokenizer = CodeTokenizer()
all_concepts = {}
kws={}

for language in LANGUAGES:
    print(language)
    all_concepts[language] = {}
    kws[language] = {}
    for query in tqdm(queries):
        result_indices = results[language][query][0]
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
        for result_index in results[language][query][0]:
            result_kws = code_tokenizer.tokenize(get_document(definitions[language][result_index]))
            d = {kw: kw in result_kws for kw in all_query_result_kws}
            df_list.append(d)
        
        df = pd.DataFrame(df_list)
        df.index=results[language][query][0]

        df=df.loc[df.any(axis='columns')]
        objects = [str(z) for z in df.index.tolist()]
        properties = list(df)
        bools = list(df.fillna(False).astype(bool).itertuples(index=False, name=None))

        c = concepts.Context(objects, properties, bools)
        all_concepts[language][query] = c
        kws[language][query] = all_query_result_kws
        
        
######################################################################
######################################################################
################### Prepare Query Subsets ############################
######################################################################
######################################################################

#Create subset of queries that have at least one positive result with at least one task

query_subsets = defaultdict(list)
all_good_results = defaultdict(dict)
all_included_results = defaultdict(dict)

MIN_SCORE = 2
MIN_RESULTS = 3

def is_list_decreasing(L):
    '''
        Checks if values in a list decrease monotonically
    '''
    return all(x>=y for x, y in zip(L, L[1:]))


for language in LANGUAGES:
    for i, query in enumerate(queries):
        relevance_scores = relevances[language][query]
        all_result_fids = results[language][query][0]
        all_result_defs = [definitions[language][fid] for fid in all_result_fids]
        good_results = []
        included_results = []
        scores = []
        
        # Record the search results that have ratings, and those that have positive ratings
        for j, fid in enumerate(all_result_fids):
            if all_result_defs[j]["url"] in relevance_scores:
                included_results.append(fid)
                score = relevance_scores[all_result_defs[j]["url"]]
                scores.append(score)
                if score >= MIN_SCORE:
                    good_results.append(fid)
        all_good_results[language][query]=good_results
        all_included_results[language][query]=included_results
        
        # Record queries meeting three criteria:
        #    1. All rated results are not already in the correct order OR the top search result is incorrect
        #    2. Have at least one rated result with a positive score that has at least one task and one keyword
        #    3. Have at least MIN_RESULTS search results with ratings
        if is_list_decreasing(scores) and (good_results and all_result_fids.index(good_results[0])==0): 
            continue
        task_df = all_task_dfs[language][query]        
        task_df_result_fids = task_df["row"].unique()
        task_df_result_indexes = [results[language][query][0].index(j) for j in task_df_result_fids]
        context = all_concepts[language][query]
        context_result_fids = [int(fid) for fid in context.lattice.supremum.extent]
        context_result_indexes = [results[language][query][0].index(j) for j in context_result_fids]
        usable_result_indexes = set(task_df_result_indexes) & set(context_result_indexes)
        for result_idx in usable_result_indexes:
            result_def = all_result_defs[result_idx]
            result_fid = all_result_fids[result_idx]
            if result_def["url"] in relevance_scores and result_fid in good_results and len(included_results)>=MIN_RESULTS:
                query_subsets[language].append((i,query))
                break

for language, subset in query_subsets.items():
    print(language, len(subset))


    
    
######################################################################
######################################################################
######################## Metrics #####################################
######################################################################
######################################################################

def ndcg(predictions, relevance_scores, ignore_rank_of_non_annotated_urls=True):
    num_results = 0
    ndcg_sum = 0

    for query, query_relevance_annotations in relevance_scores.items():
        if query in predictions:
        
            current_rank = 1
            query_dcg = 0
            relevant_relevances = []
            
            for url in predictions[query]:
                if url in query_relevance_annotations:
                    query_dcg += (2**query_relevance_annotations[url] - 1) / np.log2(current_rank + 1)
                    current_rank += 1
                    #New
                    relevant_relevances.append(query_relevance_annotations[url])
                elif not ignore_rank_of_non_annotated_urls:
                    current_rank += 1
            #New

            query_idcg = 0
            for i, ideal_relevance in enumerate(sorted(relevant_relevances, reverse=True), start=1):
#             for i, ideal_relevance in enumerate(sorted(query_relevance_annotations.values(), reverse=True), start=1):
                query_idcg += (2 ** ideal_relevance - 1) / np.log2(i + 1)
            if query_idcg == 0:
                # We have no positive annotations for the given query, so we should probably not penalize anyone about this.
                continue
            num_results += 1
            ndcg_sum += query_dcg / query_idcg
    return ndcg_sum / num_results

def coverage_per_language(predictions, relevance_scores, with_positive_relevance=False):
    """
    Compute the % of annotated URLs that appear in the algorithm's predictions.
    """
    num_annotations = 0
    num_covered = 0
    for query, url_data in relevance_scores.items():
        if query in predictions:
            urls_in_predictions = set(predictions[query])
            for url, relevance in url_data.items():
                if not with_positive_relevance or relevance > 0:
                    num_annotations += 1
                    if url in urls_in_predictions:
                        num_covered += 1
    return num_covered / num_annotations

def run_ndcg(relevance_scores, predictions, language, log=True):
    predictions = {query: [definitions[language][p]["url"] for p in predictions[query]] for query in predictions}

    # Now Compute the various evaluation results
    if log:
        print('% of URLs in predictions that exist in the annotation dataset:')
        print(f'\t{language}: {coverage_per_language(predictions, relevance_scores)*100:.2f}%')

        print('% of URLs in predictions that exist in the annotation dataset (avg relevance > 0):')
        print(f'\t{language}: {coverage_per_language(predictions, relevance_scores, with_positive_relevance=True) * 100:.2f}%')

        print('NDCG:')
        print(f'\t{language}: {ndcg(predictions, relevance_scores):.3f}')

        print('NDCG (full ranking):')
        print(f'\t{language}: {ndcg(predictions, relevance_scores, ignore_rank_of_non_annotated_urls=False):.3f}')
        print()
    return (ndcg(predictions, relevance_scores), ndcg(predictions, relevance_scores, ignore_rank_of_non_annotated_urls=False))


def run_mrr(relevance_scores, predictions, language,log=True):
    predictions = {query: [definitions[language][p]["url"] for p in predictions[query]] for query in predictions}
    mrr_list = []
    for query, query_relevance_annotations in relevance_scores.items():
        if query in predictions:
            for i, url in enumerate(predictions[query]):
                if url in query_relevance_annotations and query_relevance_annotations[url]>1:
                    mrr_list.append(1/(i+1))
                    break
    mrr_score = sum(mrr_list)/len(mrr_list)
    if log:
        print('MRR:')
        print(f'\t{language}: {mrr_score:.3f}')
    return mrr_score


def run_map(relevance_scores, predictions, language,log=True):
    predictions = {query: [definitions[language][p]["url"] for p in predictions[query]] for query in predictions}
    map_list = []
    for query, query_relevance_annotations in relevance_scores.items():
        if query in predictions:
            p_list = []
            
            for i, url in enumerate(predictions[query]):
                if url in query_relevance_annotations and query_relevance_annotations[url]>1:
                    p_list.append((1+len(p_list))/(i+1))
            ap_score = sum(p_list)/len(p_list)
            map_list.append(ap_score)
    map_score = sum(map_list)/len(map_list)
    if log:
        print('MAP:')
        print(f'\t{language}: {map_score:.3f}')
    return map_score





######################################################################
######################################################################
#################### ZaCQ and VO Evaluation ##########################
######################################################################
######################################################################

HYPERPARAMETER_CONFIGURATIONS = [
 (1, .8),
 (1, .85),
 (3, .15),
 (3, .2),
 (3, .25),
] + [(support, confidence) for support in np.arange(10,21,2) for confidence in np.arange(.1,.44,.03)]

###### Top-level functions to run series of experiments #######

def test_no_cqs(use_full_query_set=True, language="java"):
    if use_full_query_set:
        _queries = [(i, query) for i, query in enumerate(queries)]
    else:
        _queries = query_subsets[language]
    predictions = {query: results[language][query][0] for i, query in _queries}
    ndcg_score, full_ndcg_score =  run_ndcg(relevances[language], predictions, language)
    mrr_score =  run_mrr(relevances[language], predictions, language)
    map_score =  run_map(relevances[language], predictions, language)
    return ndcg_score, full_ndcg_score

def tune_ask_cqs(use_full_query_set=True, max_results=None, n=None, apos=None, aneg=None):
    """
        Experiments with different parametersfor user input weights
        Returns the best scores and optimal parameters
    """
    best_new_ndcg_score= 0
    best_new_full_ndcg_score= 0
    bap=0
    bapf=0
    ban=0
    banf=0
    if apos != None:
        aprange = [apos]
    else:
        aprange = np.arange(0, 5, .5)
    if aneg != None:
        anrange = [aneg]
    else:
        anrange = np.arange(0, 5, .5)  
        
    for apos in aprange:
        for aneg in anrange:
            print(apos,aneg)
            new_ndcg_score, new_full_ndcg_score = test_ask_cqs(use_full_query_set, max_results, n, apos, aneg)
            if new_ndcg_score>best_new_ndcg_score: 
                best_new_ndcg_score=new_ndcg_score
                bap=apos
                ban=aneg
            if new_full_ndcg_score>best_new_full_ndcg_score: 
                best_new_full_ndcg_score=new_full_ndcg_score
                bapf=apos
                banf=aneg
    return best_new_ndcg_score, best_new_full_ndcg_score, bap, ban, bapf, banf


def tune_reranking_params(cq,
                          use_full_query_set=True, 
                          max_results=5, 
                          infer_support=4, 
                          infer_conf=.3, 
                          option_support=3,
                          n=5,
                          language="java",
                          use_query_df=False):
    """
      Runs ask_cqs with a range of apos and aneg values
      Stores the best ndcg and params at each n
    """
    # Run ask_cqs to get predictions for each n, tuned
    if use_full_query_set:
        _queries = [(i, query) for i, query in enumerate(queries)]
    else:
        _queries = query_subsets[language]
    predictions = {query: results[language][query][0] for i, query in _queries}
    n = range(n)
    apos = list(np.arange(0,6.1,.25))+[8,9,10,20,100,1000]
    aneg = list(np.arange(0,6.1,.25))+[8,9,10,20,100,1000]
    new_preds = ask_cqs(cq,
                        _queries, 
                          predictions, 
                          max_results=max_results, 
                          n=n, 
                          apos=apos, 
                          aneg=aneg, 
                          infer_support=infer_support, 
                          infer_conf=infer_conf, 
                          option_support=option_support,
                          language=language,
                          use_query_df=use_query_df)
    # Iterate through all apos/aneg configurations for weighting results at each n, record best ones
    ndcg_results = {}
    for _n in new_preds:
        ndcg_results[_n] = {}
        best_new_ndcg= 0
        best_new_full_ndcg= 0
        bparams = (0,0)
        bfparams = (0,0)
        bmrr=0
        bmap_score=0
        for params in new_preds[_n]:
            new_ndcg, new_full_ndcg = run_ndcg(relevances[language], new_preds[_n][params], language,False)
            mrr = run_mrr(relevances[language], new_preds[_n][params], language,False)
            map_score = run_map(relevances[language], new_preds[_n][params], language,False)
            if new_ndcg>best_new_ndcg: 
                best_new_ndcg=new_ndcg
                bparams=params
            if new_full_ndcg>best_new_full_ndcg: 
                best_new_full_ndcg=new_full_ndcg
                bfparams = params
            if mrr>bmrr:
                bmrr=mrr
            if map_score>bmap_score:
                bmap_score=map_score
        ndcg_results[_n]["bndcg"] =best_new_ndcg
        ndcg_results[_n]["bfndcg"] = best_new_full_ndcg
        ndcg_results[_n]["bparams"] = bparams
        ndcg_results[_n]["bfparams"] = bfparams
        ndcg_results[_n]["mrr"] = bmrr
        ndcg_results[_n]["map"] = bmap_score
    return ndcg_results
    
    print() 
    
def tune_cq_params(cq, use_full_query_set=False, max_results=5, n=5, language="java",  params=None, use_query_df = True):
    """
      Runs tune_reranking_params with a range of CQ param values
      Stores all results
    """
    score, full_score = test_no_cqs(use_full_query_set, language)    
    #Run all permutations
    all_results = {}
    if params:
        infer_support, infer_conf, option_support = params
        ndcg_results= tune_reranking_params(cq=cq,
                                            use_full_query_set=use_full_query_set, 
                                            max_results=max_results, 
                                            n=n,
                                            infer_support=infer_support, 
                                            infer_conf=infer_conf, 
                                            option_support=option_support,
                                            language=language,
                                            use_query_df = use_query_df)
        all_results[(infer_support, infer_conf, option_support)] = ndcg_results
        print(infer_support, infer_conf, option_support)
        for _n in ndcg_results:
            print(f"{_n}: {str(ndcg_results[_n])}")
        print() 
    else: 
        option_support = 1
        for infer_support, infer_conf in HYPERPARAMETER_CONFIGURATIONS:
            ndcg_results= tune_reranking_params(cq=cq,
                                                use_full_query_set=use_full_query_set, 
                                                max_results=max_results, 
                                                n=n,
                                                infer_support=infer_support, 
                                                infer_conf=infer_conf, 
                                                option_support=option_support,
                                                language=language,
                                                use_query_df = use_query_df)
            all_results[(infer_support, infer_conf, option_support)] = ndcg_results
            print(infer_support, infer_conf, option_support)
            for _n in ndcg_results:
                print(f"{_n}: {str(ndcg_results[_n])}")
            print() 
    return all_results


def ask_cqs(cq,
            queries, 
            predictions, 
            max_results=5, 
            n=[5], 
            apos=[2.5], 
            aneg=[2.0], 
            infer_support=5, 
            infer_conf=.25, 
            option_support=4,
            language="java",
            log_session=False,
            use_query_df=False,
            output_questions=False):  
    """Evaluates CQ method on specified queries 
    
    Returns:
        A dict for each query containing the ndcg after asking each of the first n questions.
    """
    new_predictions = defaultdict(lambda: defaultdict(lambda: dict()))
    if output_questions: 
        question_list = []
        
    for i, query in queries:
        #Logging
        if log_session:
            print(query)
            print()
        if output_questions:
            query_question_list = []
        
        #Convenient access to frequent data
        task_df = all_task_dfs[language][query].copy()
        cq.normalize_task_df(task_df)
        relevance_scores = relevances[language][query]
        query_results = predictions[query]
        good_results = all_good_results[language][query]
        
        #Clarify until a single task phrase is clarified, or a single valid search result remains
        set_values = defaultdict(str)
        not_list = []
        question=inferred=target=options=None
        filtered_rows = set()
        for _n in n:
            _query = query if use_query_df else None
            question, inferred, target, options = cq.generate_cq(
                task_df=task_df, 
                set_values = set_values,
                max_results=max_results,
                infer_support=infer_support, 
                infer_conf=infer_conf, 
                option_support=option_support,
                query=_query)
            
            if  target or inferred: 
                #Look at answer options
                sorted_answers = sorted(list(options.items()), key=lambda x:len(x[1]), reverse=True)
                if max_results>0:
                    shown_answers = [x[0] for x in sorted_answers[:max_results]]
                else:
                    shown_answers = [x[0] for x in sorted_answers]
                response = None    
                
                #Select an answer
                for answer in reversed(shown_answers):
                    if set(good_results) & set(options[answer]):
                        response = answer
                        break
            
                #Process answer
                if response:
                    set_values.update(inferred)
                    if target:
                        if target != "role":
                            set_values[target] = response
                        else:
                            verb, do, prep, po = response
                            for slot, slot_value in {"verb": verb, "direct_object": do, "preposition": prep, "preposition_object":po}.items():
                                if slot_value:
                                    set_values[slot] = slot_value
                else:
                    current_not_values = inferred
                    if target:
                        if target != "role":
                            for answer in reversed(shown_answers):
                                current_not_values[target] = answer
                                not_list.append(current_not_values.copy())
                        else:
                            for answer in reversed(shown_answers):
                                verb, do, prep, po = answer
                                current_not_values["verb"] = verb
                                if do:
                                    current_not_values["direct_object"] = do
                                else:
                                    current_not_values["preposition"] = prep
                                    current_not_values["preposition_object"] = po
                                not_list.append(current_not_values.copy())
                                current_not_values = inferred
                    else:
                        not_list.append(current_not_values) 
                        
                if log_session: 
                    print(question)
                    if options:
                        print(shown_answers)
                    print(response)
                    print()
                    
                if output_questions:
                    query_question_list.append({'question':question, 'options':sorted_answers, 'answer':response})
                
                #Filter task_df with answer 
                relevant_funcs = []
                task_df = cq.apply_set_values(task_df, set_values)
                task_df,diff_rows = cq.filter_not_dicts(task_df, not_list) 
                filtered_rows.update(diff_rows)
                for result_fid in task_df["row"].unique():
                    relevant_funcs.append(result_fid)
                    
                if relevant_funcs:
                    relevant_func_vectors = [code_representations[language][j] for j in relevant_funcs]
                    avg_relevant_func_vector = np.average(relevant_func_vectors, axis=0)
                if filtered_rows:
                    non_relevant_funcs = list(filtered_rows)
                    non_relevant_func_vectors = [code_representations[language][j] for j in non_relevant_funcs]
                    avg_non_relevant_func_vector = np.average(non_relevant_func_vectors, axis=0)
                
                # Calculate and record predictions for each relevance feedback hyperparameter configuration
                for ap in apos:
                    for an in aneg:
                        query_vector = results[language][query][2]
                        if relevant_funcs:
                            weighted_relevant_func_vector = np.multiply(avg_relevant_func_vector, ap)
                            query_vector = np.add(query_vector, weighted_relevant_func_vector)
                        if filtered_rows:
                            weighted_non_relevant_func_vector = np.multiply(avg_non_relevant_func_vector, an)                
                            query_vector = np.subtract(query_vector, weighted_non_relevant_func_vector)
                        #Get new predictions
                        prediction_vectors = [code_representations[language][idx] for idx in predictions[query]]
                        similarities = pairwise.cosine_similarity(prediction_vectors, [query_vector])
                        res_sim_tups = list(zip(query_results, similarities))
                        res_sim_tups.sort(key=lambda x: x[1], reverse=True)
                        _new_predictions = [p[0] for p in res_sim_tups]
                        new_predictions[_n][(ap, an)][query] = _new_predictions
            
            # If there is no clarification question, store the previous predictions
            elif _n>0:
                for ap in apos:
                    for an in aneg:
                        new_predictions[_n][(ap, an)][query] = new_predictions[_n-1][(ap, an)][query]
                        
        if log_session:
            print(relevant_funcs)
            print(good_results)
            print([query_results.index(i) for i in good_results])
            print([new_predictions[_n][(ap, an)][query].index(fid) for fid in good_results])
            print()
            print() 
            print() 
            print() 
            print() 
            
        if output_questions:
            question_list.append(query_question_list)
            
    if output_questions:
        return question_list
    else: 
        return new_predictions
    
# Run experiments

zacq_results={}
for language in LANGUAGES:
    zacq_results[language] = tune_cq_params(n=10, language=language, cq=zacq)
pickle.dump(zacq_results, open("../data/output/zacq_results.pkl", "wb"))

vocq_results={}
for language in LANGUAGES:
    vocq_results[language] = tune_cq_params(max_results=5, n=10, language=language, cq=vocq, params=(0,0,0))
pickle.dump(vocq_results, open("../data/output/vo_results.pkl", "wb"))



######################################################################
######################################################################
########################## KW Evaluation #############################
######################################################################
######################################################################

def kw_tune_reranking_params(cq,
                          use_full_query_set=True, 
                          max_results=5, 
                          n=5,
                          language="java"):
    """
      Runs ask_cqs with a range of apos and aneg values
      Stores the best ndcg and params at each n
    """
    # Run ask_cqs to get predictions for each n, tuned
    if use_full_query_set:
        _queries = [(i, query) for i, query in enumerate(queries)]
    else:
        _queries = query_subsets[language]
    predictions = {query: results[language][query][0] for i, query in _queries}
    n = range(n)
    apos = list(np.arange(0,6.1,.25))+[7,8,9,10,20,100,1000]
    aneg = list(np.arange(0,6.1,.25))+[7,8,9,10,20,100,1000]
    new_preds = kw_ask_cqs(kwcq,
            _queries, 
            predictions, 
            max_results=max_results, 
            n=n, 
            apos=apos, 
            aneg=aneg, 
            language=language,
            log_session=False)
    # Iterate through all apos/aneg configurations for weighting results at each n, record best ones
    ndcg_results = {}
    for _n in new_preds:
        ndcg_results[_n] = {}
        best_new_ndcg= 0
        best_new_full_ndcg= 0
        bparams = (0,0)
        bfparams = (0,0)
        bmrr=0
        bmap_score=0
        for params in new_preds[_n]:
            new_ndcg, new_full_ndcg = run_ndcg(relevances[language], new_preds[_n][params], language,False)
            mrr = run_mrr(relevances[language], new_preds[_n][params], language,False)
            map_score = run_map(relevances[language], new_preds[_n][params], language,False)
            if new_ndcg>best_new_ndcg: 
                best_new_ndcg=new_ndcg
                bparams=params
            if new_full_ndcg>best_new_full_ndcg: 
                best_new_full_ndcg=new_full_ndcg
                bfparams = params
            if mrr>bmrr:
                bmrr=mrr
            if map_score>bmap_score:
                bmap_score=map_score
        ndcg_results[_n]["bndcg"] =best_new_ndcg
        ndcg_results[_n]["bfndcg"] = best_new_full_ndcg
        ndcg_results[_n]["bparams"] = bparams
        ndcg_results[_n]["bfparams"] = bfparams
        ndcg_results[_n]["mrr"] = bmrr
        ndcg_results[_n]["map"] = bmap_score
    return ndcg_results
    
    print() 
    
def kw_tune_cq_params(cq, use_full_query_set=False, max_results=5, n=5, language="java"):
    """
      Runs tune_reranking_params with a range of CQ param values
      Stores all results
    """
    score, full_score = test_no_cqs(use_full_query_set, language)    
    #Run all permutations
    all_results = {}
    ndcg_results= kw_tune_reranking_params(cq=cq,
                                        use_full_query_set=use_full_query_set, 
                                        max_results=max_results, 
                                        n=n,
                                        language=language)
    for _n in ndcg_results:
        print(f"{_n}: {str(ndcg_results[_n])}")
    print() 
    return ndcg_results


def kw_ask_cqs(cq,
            queries, 
            predictions, 
            max_results=5, 
            n=[5], 
            apos=[2.5], 
            aneg=[2.0], 
            language="java",
            log_session=False,
            output_questions=False):  
    """Evaluates CQ method on specified queries 
    
    Returns:
        A dict for each query containing the ndcg after asking each of the first n questions.
    """
    if output_questions:
        question_list = []
    new_predictions = defaultdict(lambda: defaultdict(lambda: dict()))
    for i, query in queries:
        #Logging
        if log_session:
            print(query)
            print()
        if output_questions:
            query_question_list = []
        #Convenient access to frequent data
        context = all_concepts[language][query]
        q_kws = kws[language][query]
        relevance_scores = relevances[language][query]
#         result_defs = [definitions[language][idx] for idx in predictions[query]]
        query_results = predictions[query]
        good_results = all_good_results[language][query]

        #Go through and ask questions until done
        intents = []
        not_intents = []
        question=inferred=target=options=None
        filtered_rows = set()
        for _n in n:

            question, inferred, target, options = cq.generate_cq(context, intents, not_intents, max_results, q_kws)
            
            if  options: 
                #Look at answer options
                sorted_answers = sorted(list(options.items()), key=lambda x:len(x[1]), reverse=True)
                if max_results>0:
                    shown_answers = [x[0] for x in sorted_answers[:max_results]]
                else:
                    shown_answers = [x[0] for x in sorted_answers]
                response = None    
                
                #Select an answer
                for answer in reversed(shown_answers):
                    if set(good_results) & set(options[answer]):
                        response = answer
                        break
                
                #Process answer
                if response:
                    intents.append(response)
                else:
                    not_intents+=shown_answers
                        
                if log_session: 
                    print(question)
                    if options:
                        print(shown_answers)
                    print(response)
                    print()
                    
                if output_questions:
                    query_question_list.append({'question':question, 'options':sorted_answers, 'answer':response})

                relevant_funcs, non_relevant_funcs = cq.get_candidates_rejects(context, intents, not_intents)
                relevant_funcs = [int(i) for i in relevant_funcs]
                non_relevant_funcs = [int(i) for i in non_relevant_funcs]
                
                if relevant_funcs:
                    relevant_func_vectors = [code_representations[language][j] for j in relevant_funcs]
                    avg_relevant_func_vector = np.average(relevant_func_vectors, axis=0)
                if non_relevant_funcs:
                    non_relevant_func_vectors = [code_representations[language][j] for j in non_relevant_funcs]
                    avg_non_relevant_func_vector = np.average(non_relevant_func_vectors, axis=0)
                
                for ap in apos:
                    for an in aneg:
                        query_vector = results[language][query][2]
                        if relevant_funcs:
                            weighted_relevant_func_vector = np.multiply(avg_relevant_func_vector, ap)
                            query_vector = np.add(query_vector, weighted_relevant_func_vector)
                        if non_relevant_funcs:
                            weighted_non_relevant_func_vector = np.multiply(avg_non_relevant_func_vector, an)                
                            query_vector = np.subtract(query_vector, weighted_non_relevant_func_vector)
                        #Get new predictions
                        prediction_vectors = [code_representations[language][idx] for idx in predictions[query]]
                        similarities = pairwise.cosine_similarity(prediction_vectors, [query_vector])
                        res_sim_tups = list(zip(query_results, similarities))
                        res_sim_tups.sort(key=lambda x: x[1], reverse=True)
                        _new_predictions = [p[0] for p in res_sim_tups]
                        new_predictions[_n][(ap, an)][query] = _new_predictions
                        
            elif _n>0:
                for ap in apos:
                    for an in aneg:
                        new_predictions[_n][(ap, an)][query] = new_predictions[_n-1][(ap, an)][query]
                        
        if log_session:
            print(relevant_funcs)
            print(good_results)
            print([query_results.index(i) for i in good_results])
            print([new_predictions[_n][(ap, an)][query].index(i) for i in good_results])
            print()
            print() 
            print() 
            print() 
            print() 

        if output_questions:
            question_list.append(query_question_list)
            
    if output_questions:
        return question_list
    else: 
        return new_predictions

kwcq_results={}
for language in LANGUAGES:
    kwcq_results[language] = kw_tune_cq_params(kwcq, use_full_query_set=False, max_results=5, n=25, language=language)
pickle.dump(kwcq_results, open("../data/output/kw_results.pkl", "wb"))
