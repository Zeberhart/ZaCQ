import sys
import os
import pickle
from tqdm import tqdm
import codesearch

LANGUAGES = ('python', 'go', 'javascript', 'java', 'php', 'ruby')

csn_path = os.path.join("..", "CodeSearchNet")
data_path = os.path.join("..", "data")
model_path = os.path.join(data_path, "models", "neuralbowmodel.pkl.gz")
queries_path = os.path.join(csn_path, "resources", "queries.csv")

with open(queries_path, "r") as q_file:
    queries = [line.strip() for line in q_file if line.strip()][1:]
    
for language in LANGUAGES:
    print(f"Language: {language}")
    
    definitions_path = os.path.join(csn_path, "resources", "data", f"{language}_dedupe_definitions_v2.pkl")
    output_vectors_path = os.path.join(data_path, "vectors", f"{language}_vectors.pkl")
    output_indexes_path = os.path.join(data_path, "annoy_indexes", f"{language}_indexes.pkl")
    output_results_path = os.path.join(data_path, "results", f"{language}_results.pkl")

    code_search = codesearch.CSNCodeSearch(model_path)
    code_search.load_definitions(definitions_path)
    
    code_search.build_vectors(output_vectors_path)
    code_search.build_search_index(output_indexes_path)
    results = {}
    print("Running queries")
    for query in tqdm(queries):
        results[query] = code_search.search_text(query)
    with open(output_results_path, "wb") as o_file:
        pickle.dump(results, o_file)