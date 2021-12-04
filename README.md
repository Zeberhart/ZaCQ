# Clarifying Questions for Query Refinement in Source Code Search

This code is part of the reproducibility package for the SANER 2022 paper "Generating Clarifying Questions for Query Refinement in Source Code Search".

It consists of five folders:

* codesearch/ - API to access the CodeSearchNet datasets and neural bag-of-words code retrieval method.

* cq/ - Implementation of the ZaCQ system, including an implementation of the the [TaskNav](https://www.cs.mcgill.ca/~swevo/tasknav/) development task extraction algorithm and two baseline query refinement methods.

* data/ - Includes pretrained code search model and config files for task extraction.

* evaluation/ - Scripts to run and evaluate ZaCQ.

* interface/ - Backend and Frontend servers for a search interface implementing ZaCQ. 

## Setup

1. Clone the CodeSearchNet package to the root directory, and download the CSN datasets

```
cd ZaCQ
git clone https://github.com/github/CodeSearchNet.git
cd CodeSearchNet/scripts
./download_and_preprocess
```

2. Use a CSN model to create vector representations for candidate code search results. A pretrained Neural BoW model is included in this package.

```
cd codesearch
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python _setup.py
```

This will save and index vectors in the `data` folder. It will also generate search results for the 99 CSN queries.

3. Task extraction is fairly quick for small sets of code search results, but it is expensive to do repeatedly. To expedite the evaluation, we cache the extracted tasks for the results of the 99 CSN queries, as well as keywords for all functions in the datasets. 

```
cd cq
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python _setup.py
```

Cached tasks and keywords are stored in the `data` folder.

## Evaluation

To evaluate the ZaCQ and the other query refinement methods on the CSN queries, you may use the following:

```
cd evaluation
python run_queries.py
python evaluate.py
```

The `run_queries` script determines the subset of CSN queries that can be automatically evaluated, and simulates interactive refinement sessions for all valid questions for each language in CSN. For ZaCQ, the script runs through a set of predefined hyperparameter combinations. The script calculates NDCG, MAP, and MRE metrics for each refinement method and hyperparameter configuration, and stores them in the `data/output` folder

The `evaluate` script averages the metrics across all languages after 1-*N* rounds of refinement. For ZaCQ, it also records the best-performing hyperparamter combination after *n* rounds of refinement.


## Interface

To run the interactive search interface, you need to run two backend servers and start the GUI server:

```
cd interface/cqserver
python ClarifyAPI.py
```

```
cd interface/searchserver
python SearchAPI.py
```

```
cd interface/gui
npm start
```

By default, you can access the GUI at `localhost:3000`




