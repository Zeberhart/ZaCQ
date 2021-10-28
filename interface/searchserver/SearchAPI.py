import json as js
import time
import traceback
import sys
import argparse
import os
import nltk
import threading
import asyncio
import websockets

from sanic import Sanic
from sanic.response import file
from sanic.response import json, text

import codesearch

# rootdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", "..")
# sys.path.append(os.path.join(rootdir, "src"))

application = Sanic(name="SearchAPI")

definitions_path = '../codesearchnet/resources/data/java_dedupe_definitions_v2.pkl'
vectors_path = "../data/output/java_function_vectors.pkl"
index_path = "../data/output/java_search_index.pkl"
model_path = "../codesearchnet/resources/saved_models/neuralbowmodel-2021-08-16-14-31-23_model_best.pkl.gz"

code_search = codesearch.CSNCodeSearch(model_path, search_index_path = index_path, vectors_path = vectors_path)
code_search.load_definitions(definitions_path)
print("(Java) Definitions loaded")

# @application.route('/functions/<library>')
# async def functions(request, library):
#     data = shared_envs[library].functions
#     return json(data, headers={"Access-Control-Allow-Origin": "*"})

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

@application.route('/aaa')
async def aaa(request):
    return json({"type":"AAAHHHH", "text":"AAAAHHHHHHH"})


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
                await ws.send(js.dumps(search_and_rerank(query, candidates,rejects, )))
            elif data["type"] == "get-data":
                indices = data["indices"]
                await ws.send(js.dumps(get_data(indices)))
        except Exception as e: 
            print("Error:")
            print(e)
            traceback.print_exc()
            await ws.send(js.dumps({"type":"ERROR", "text":"ERROR"}))
            

async def connect(uri):
    async with websockets.connect(uri) as ws:
        await ws.send(js.dumps({"type": "set-server-role", "role": "search"}))
        print("connected")
        while True:
            data = await ws.recv()        
            try:
                data = js.loads(data)
                if data["type"] == "search":
                    query = data["query"]
                    response = search(query)
                    response['cid'] = data["cid"]
                    await ws.send(js.dumps(response))
                elif data["type"] == "search-rerank":
                    query = data["query"]
                    candidates = data["candidates"]
                    rejects = data["rejects"]
                    mode = data["mode"]
                    response = search_and_rerank(query, candidates,rejects, mode)
                    response['cid'] = data["cid"]
                    await ws.send(js.dumps(response))
                elif data["type"] == "get-data":
                    indices = data["indices"]
                    response = get_data(indices)
                    response['cid'] = data["cid"]
                    await ws.send(js.dumps(response))
            except Exception as e: 
                print("Error:")
                print(e)
                traceback.print_exc()
                response = {"type":"ERROR", "text":"ERROR"}
                response['cid'] = data["cid"]
                await ws.send(js.dumps(response))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', action='store_true')
    args = parser.parse_args()
    if args.s:
        application.run(host="0.0.0.0", 
                        port=os.environ.get('PORT') or 8002, 
                        debug=True)
    else:
        uri = "wss://handoff-server.herokuapp.com/server"
        while True:
            try:
                asyncio.get_event_loop().run_until_complete(connect(uri))
            except:
                print("Lost connection, retrying in 1 second...")
                time.sleep(5)
        















