import time
import traceback
import sys
import argparse
import os
import nltk
import threading
import asyncio
import websockets
import json as js

from sanic import Sanic
from sanic.response import file
from sanic.response import json, text

rootdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..","..")
sys.path.append(os.path.join(rootdir, "codesearch"))
import codesearch

language = "java"
application = Sanic(name="SearchAPI")

definitions_path = os.path.join(rootdir, "CodeSearchNet", "resources", "data", f"{language}_dedupe_definitions_v2.pkl") 
vectors_path = os.path.join(rootdir, "data", "vectors", f"{language}_vectors.pkl") 
index_path = os.path.join(rootdir, "data", "annoy_indexes", f"{language}_indexes.pkl") 
model_path = os.path.join(rootdir, "data", "models", "neuralbowmodel.pkl.gz") 

code_search = codesearch.CSNCodeSearch(model_path, search_index_path = index_path, vectors_path = vectors_path)
code_search.load_definitions(definitions_path)
print(f"{language} Definitions loaded")

def search(query):
    indices, distances, query_vector = code_search.search_text(query)
    return {"type": "results", 'indices': indices, 'distances':distances, 'query_vector':query_vector.tolist()}

def search_and_rerank(query, candidates, rejects, mode):
    indices, distances, query_vector = code_search.search_text(query)
    if mode=="cq":
        candidate_weight, reject_weight = 3.5, 1
    else:
        candidate_weight, reject_weight = 4, 0.5
    indices, distances, new_query_vector = code_search.rerank(indices, query_vector, candidates, rejects, candidate_weight, reject_weight)
    return {"type": "results",'indices': indices,}
    
def get_data(indices):
    return {"type": "data", "data":code_search.get_definitions(indices)}

@application.websocket('/feed')
async def feed(request, ws):
    while True:
        data = await ws.recv()  
        try:
            data = js.loads(data)
            if data["type"] == "search":
                query = data["query"]
                await ws.send(js.dumps(search(query)))
            elif data["type"] == "search-rerank":
                query = data["query"]
                candidates = data["candidates"]
                rejects = data["rejects"]
                mode = data["mode"]
                await ws.send(js.dumps(search_and_rerank(query, candidates,rejects, mode)))
            elif data["type"] == "get-data":
                indices = data["indices"]
                await ws.send(js.dumps(get_data(indices)))
        except Exception as e: 
            print("Error:")
            print(e)
            traceback.print_exc()
            await ws.send(js.dumps({"type":"ERROR", "text":"ERROR"}))


if __name__ == '__main__':
    application.run(host="0.0.0.0", 
                    port=os.environ.get('PORT') or 8002, 
                    debug=True)
    















