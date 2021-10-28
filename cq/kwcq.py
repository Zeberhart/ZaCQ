#####Top-level methods#####
import pandas as pd
import numpy as np

from sklearn.metrics import pairwise
from itertools import chain
from collections import defaultdict, Counter
from lemminflect import getLemma, getAllLemmas, getAllLemmasOOV, isTagBaseForm
from lemminflect import getInflection, getAllInflections, getAllInflectionsOOV

import nltk
from re import finditer
import re
import concepts

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from nltk.corpus import stopwords
stop = list(stopwords.words('english'))
stop.extend("return returns param params parameter parameters code class inheritdoc".split())

NO_QUESTION = ("", {}, None, {})



def generate_cq(context=None,
                intent = [],
                not_intent=[], 
                max_results=5,
                keyword_order=None):
    """
        Top-level method to generate a clarification question given a lit of docstrings
        
        Returns the question string, as well as the functions associated w/ each answer
    """
    if keyword_order:
        keyword_order = {word: i for i, word in enumerate(keyword_order)}
    kws = get_keywords(context, intent, not_intent, max_results, keyword_order)
    answers = get_cq_answers(context, intent, not_intent, kws)
    if answers:
        cq = create_string(list(answers))
        return(cq, {}, None, answers)
    else:
        return NO_QUESTION
        

def get_keywords(context, intent, not_intent, max_results, keyword_order):
    o, _ = get_candidates_rejects(context, intent, not_intent)
    o = [str(ob) for ob in o]
    p = context.properties
    o, p, b = context.definition().take(o, p, reorder=True)
    if not o:
        return []
    context = concepts.Context(o,p,b)
        
    kws = []
    seen_kws = set(context.lattice[intent].intent)|set(not_intent)
    children_concepts = [c for c in context.lattice.downset_union([context.lattice[intent]])][:-1]
    children_intents = [set(child.intent) for child in children_concepts]
    
    while len(kws)<max_results and any(children_intents):
        for intent_set in children_intents:
            intent_set.difference_update(seen_kws)
            if intent_set:
                if keyword_order:
                    kw = sorted(intent_set, reverse=True, key=lambda x:keyword_order[x])[0]
                else:
                    kw = list(intent_set)[0]
                kws.append(kw)
                seen_kws.add(kw)
                if len(kws)>=max_results:
                    break
    return kws


def serialize_string_list(candidates):
    candidates.sort()
    if len(candidates)==2:
        obj = f"{candidates[0]}, {candidates[1]}"
    else:
        obj = ""
        for i in range(len(candidates)-1):
            obj += candidates[i] + ", "
        obj += candidates[len(candidates)-1]
    return obj
        
    
def create_string(candidates):
    """
        #4) Based on set values and target slot, select and fill a CQ template
    """
    
    return f"Related to your search:\n{serialize_string_list(candidates)}"


       
def get_cq_answers(context, intent, not_intents, candidates):
    """
        #5) For each potential answer to the CQ, find the functions that satisfy that answer
    """
    entries = {c:get_candidates_rejects(context, intent+[c], not_intents)[0] for c in candidates}
    entries = {k:v for k,v in entries.items() if v}
    return entries


def get_candidates_rejects(context, intents, not_intents, strict=True):
    if strict:
        rejects = [[int(i) for i in context.lattice[intents+[not_intent]].extent] for not_intent in not_intents]
        rejects = list(set(sum(rejects,[])))
#           filtered_rows.difference_update(relevant_funcs)#all of the ones in not_intents that aren't in relevant
        candidates = [int(i) for i in context.lattice[intents].extent]
        candidates = list(set(candidates).difference(rejects))
    else:
        o = context.objects
        p = context.properties
        dp = list(set(p).difference(not_intents))
        o, p, b = context.definition().take(o, dp, reorder=True)
        c = concepts.Context(o,p,b)
        
        candidates = c.lattice[intents].extent
        rejects_not_intent_only = set(c.lattice.supremum.objects) #Ones that consist only of not_intents
        objects_at_intent = context.lattice[intents].objects #Ones that match intent exactly
        objects_at_filtered_intent = c.lattice[intents].objects #Ones that are only intent, and ones that are intent with not_intents
        rejects_not_intent_and_intent = set(objects_at_filtered_intent).difference(objects_at_intent)
        rejects = rejects_not_intent_only.union(rejects_not_intent_and_intent)
        candidates = set(candidates).difference(rejects)
#         rejects = list(set(o).difference(candidates))
        candidates = [int(i) for i in candidates]
        rejects = [int(i) for i in rejects]
    return candidates, rejects
    