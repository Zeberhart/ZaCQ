import stanza
from simplenlg import *
from stanza.server import CoreNLPClient
from nltk.tree import Tree
import pandas as pd
from sklearn.metrics import pairwise

import pickle
import re
import shutil
import sys

import os.path
from collections import defaultdict, Counter
from annoy import AnnoyIndex
from docopt import docopt
from dpu_utils.utils import RichPath
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb
from wandb.apis import InternalApi

from dataextraction.python.parse_python_data import tokenize_docstring_from_string
import model_restore_helper


class CSNCodeSearch(object):

    def __init__(self, model_path, search_index_path="", vectors_path=""):
        model_path = RichPath.create(model_path, None)
        self.model = model_restore_helper.restore(
            path=model_path,
            is_train=False,
            hyper_overrides={})
        if search_index_path and vectors_path:
            if os.path.isfile(search_index_path) and os.path.isfile(vectors_path):
                self.load_vectors(vectors_path)
                self.load_search_index(search_index_path)
            else:
                print("Search index or vectors not found!")
    
    
    def load_definitions(self, definitions_path):
        print("Loading definitions...")
        with open(definitions_path,"rb") as f:
            self.definitions = pickle.load(f)
        
        
    def load_vectors(self, vectors_path):
        print("Loading vectors...")
        with open(vectors_path,"rb") as f:
            self.vectors = pickle.load(f)
    
    
    def load_search_index(self, search_index_path):
        print("Loading search index...")
        self.search_index = AnnoyIndex(self.vectors[0].shape[0], 'angular')
        self.search_index.load(search_index_path)
    
    
    def build_vectors(self, output_path, load=True):
        print("Building vectors...")
        indexes = [{'code_tokens': d['function_tokens'], 'language': d['language']} for d in tqdm(self.definitions)]
        vectors = self.model.get_code_representations(indexes)
        if load:
            self.vectors = vectors
        pickle.dump(vectors, open(output_path, "wb"))
        
                    
    def build_search_index(self, output_path, load=True):
        print("Building search index...")
        search_index = AnnoyIndex(self.vectors[0].shape[0], 'angular')
        for i, vector in enumerate(tqdm(self.vectors)):
            if vector is not None:
                search_index.add_item(i, vector)
        search_index.build(200)
        if load:
            self.search_index = search_index
        search_index.save(output_path)
        
    
    def get_vectors(self):
        return self.code_representations
    
    
    def get_definitions(self, indices=None):
        if indices:
            return [self.definitions[i] for i in indices]
        else:
            return self.definitions
        
    
    def search_text(self, query, language="java", results=50):
        query_vector = self.model.get_query_representations([{'docstring_tokens': tokenize_docstring_from_string(query),
                                                            'language': language}])[0]
        return self.search(query_vector, language, results)
    
    
    def rerank(self, results, query_vector, candidates=[], rejects=[], candidate_weight=3, reject_weight=2.5):
        candidates = list(set(candidates)-set(rejects))
        
        if candidates:
            candidate_vectors = [self.vectors[j] for j in candidates]
            avg_candidate_vector = np.average(candidate_vectors, axis=0)
            weighted_candidate_vector = np.multiply(avg_candidate_vector, candidate_weight)
            query_vector = np.add(query_vector, weighted_candidate_vector)
        if rejects:
            reject_vectors = [self.vectors[j] for j in rejects]
            avg_reject_vector = np.average(reject_vectors, axis=0)
            weighted_reject_vector = np.multiply(avg_reject_vector, reject_weight)                
            query_vector = np.subtract(query_vector, weighted_reject_vector)
        #Get new predictions
        result_vectors = [self.vectors[idx] for idx in results]
        result_similarities = pairwise.cosine_similarity(result_vectors, [query_vector])
        result_sim_tups = list(zip(results, result_similarities))
        result_sim_tups.sort(key=lambda x: x[1], reverse=True)
        new_results = [p[0] for p in result_sim_tups]
        new_distances = [p[1] for p in result_sim_tups]
        return new_results, new_distances, query_vector
    
    
    def search(self, query_vector, language="java", results=50):
        if not self.search_index or not self.vectors:
            raise Exception("Search index and vectors not loaded")
        
        idxs, distances = self.search_index.get_nns_by_vector(query_vector, results, include_distances=True)
        return idxs, distances, query_vector
    

        
if __name__ == "__main__":
    definitions_path = '../codesearchnet/resources/data/java_dedupe_definitions_v2.pkl'
    model_path = "../codesearchnet/resources/saved_models/neuralbowmodel-2021-05-20-17-32-09_model_best.pkl.gz"
    code_search = CSNCodeSearch(definitions_path, model_path, index_path)
    indices, distances = code_search.search("How do I graph a plot in Javascript?")
    for definition in code_search.get_function_definitions(indices):
        print(definition["identifier"])
        print()

