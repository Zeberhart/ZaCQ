#####Top-level methods#####
import pandas as pd
import numpy as np

from sklearn.metrics import pairwise
from itertools import chain
from collections import defaultdict, Counter
from lemminflect import getLemma, getAllLemmas, getAllLemmasOOV, isTagBaseForm
from lemminflect import getInflection, getAllInflections, getAllInflectionsOOV

from collections import Counter
from string import punctuation

NO_QUESTION = ("", {}, None, {})



def generate_cq(docstrings=None, 
                task_df=None, 
                set_values=defaultdict(str), 
                not_values=[], 
                max_results=5,
                infer_support=3, 
                infer_conf=.5, 
                option_support=1,
               query=None):
    """
        Top-level method to generate a clarification question given a lit of docstrings
        
        Returns the question string, as well as the functions associated w/ each answer
    """
    if docstrings:
        task_df = create_task_df(docstrings)
    elif task_df is not None:
        task_df = task_df.copy()
    else:
        print("Must supply docstrings or task dataframes/temp ids.")
        return    
    
    if not_values:
        task_df, diff_rows = filter_not_dicts(task_df, not_values)
    if set_values:
        task_df = apply_set_values(task_df, set_values)
    set_values = defaultdict(str,set_values)
    
    if len(task_df["row"].unique())<=1 or len(set_values)==2:
        return NO_QUESTION

    target_slot, candidates = select_target_slot(task_df, set_values) 

    if target_slot and candidates:
        cq, gerund_candidates = create_string(set_values, target_slot, candidates, max_results)
        answers = get_cq_answers(task_df, target_slot, candidates, max_results)
        return(cq, {}, target_slot, answers)
    else:
        return NO_QUESTION
        
def normalize_task_df(task_df):
    for column in task_df:
        if column=="row":continue
        normalized_vals = {}
        for val in task_df[column].unique():
            normalized_val = val.lower().strip(punctuation)
            if normalized_val in normalized_vals:
                task_df.loc[task_df[column]==val, column]=normalized_vals[normalized_val]
            else:
                normalized_vals[normalized_val]=val 


def create_task_df(docstrings, task_extractor):
    """
         #1) Create dataframes storing each docstring's tasks
    """
    tasks = [sum(t, []) for t in task_extractor.extract_task_dicts(docstrings)]
    tasks = [task_extractor.lemmatize_dicts(t) for t in tasks]
    for i, task_set in enumerate(tasks):
        for task in task_set:
            task["row"] = i
    tasks = sum(tasks, [])
    task_df = pd.DataFrame(tasks).explode("direct_object_modifiers").explode("preposition_object_modifiers").replace(np.nan, '', regex=True)
    return task_df


def apply_set_values(task_df, set_values):
    for slot, value in set_values.items():
        task_df = apply_slot_value(task_df, slot, value)
    return task_df
        
        
def apply_slot_value(task_df, slot, value):
    """
        #2.2) Filters a list of task dataframes with a particular slot value
    """
    if not slot or not value: return task_df
    if slot == "object_modifiers":
        return task_df[(task_df["direct_object_modifiers"] == value) | (task_df["preposition_object_modifiers"] == value)]
    elif slot == "object":
        return task_df[(task_df["direct_object"] == value) | (task_df["preposition_object"] == value)]
    else:
        return task_df[task_df[slot] == value]

def filter_not_dicts(task_df, not_dicts):
    all_diff_rows  = set()
    for not_dict in not_dicts:
        task_df, diff_rows = filter_not_dict(task_df, not_dict)
        all_diff_rows.update(diff_rows)
    return task_df, all_diff_rows
    
def filter_not_dict(task_df, not_dict):
    """
        #2.2) Filters a list of task dataframes with a particular slot value
    """
    if "object" in not_dict or "object_modifiers" in not_dict:
        not_dict_1 = not_dict.copy()
        not_dict_2 = not_dict.copy()
        
        if "object" in not_dict:
            del not_dict_1["object"]
            del not_dict_2["object"]
            not_dict_1["direct_object"] = not_dict["object"]
            not_dict_2["preposition_object"] = not_dict["object"]
    
        if "object_modifiers" in not_dict:
            del not_dict_1["object_modifiers"]
            del not_dict_2["object_modifiers"]
            not_dict_1["direct_object_modifiers"] = not_dict["object_modifiers"]
            not_dict_2["preposition_object_modifiers"] = not_dict["object_modifiers"]
            
        f_task_df = task_df.loc[~task_df[list(not_dict_1.keys())].isin(not_dict_1.values()).all(axis=1), :]
        f_task_df = f_task_df.loc[~f_task_df[list(not_dict_2.keys())].isin(not_dict_2.values()).all(axis=1), :]
    else:
        f_task_df = task_df.loc[~task_df[list(not_dict.keys())].isin(not_dict.values()).all(axis=1), :]
    diff_rows = set(task_df["row"].unique()).difference(set(f_task_df["row"].unique()))
    return f_task_df, diff_rows


def select_target_slot(task_df, set_values):
    """
        #3) Determine which slot the CQ should ask about. If there isn't a slot
        with enough support, instead ask to confirm the inferred values
        
        Returns a tuple: either None, None, indicating that the CQ should confirm inferred values;
        or the target slot and candidate values for that slot
    """
    values_to_search = []
    value_candidates = Counter()   
    target_slot = ""
    
    if not set_values:
        target_slot = "verb"
    elif "verb" in set_values and "direct_objest" not in set_values:
        target_slot = "direct_object"

    if target_slot:
        for result_id in task_df["row"].unique():
            unique_values = task_df.loc[task_df["row"]==result_id][target_slot].unique()
            for value in unique_values:
                if isinstance(value,str) and value.strip(): value_candidates[value] += 1
        return target_slot, value_candidates
    else:
        return None
            
            
def serialize_string_list(candidates):
    if len(candidates)==2:
        obj = f"{candidates[0]} or {candidates[1]}"
    else:
        obj = ""
        for i in range(len(candidates)-1):
            obj += candidates[i] + ", "
        obj += "or "+candidates[len(candidates)-1]
    return obj
        
    
def create_string(set_values, target_slot, candidates, max_results):
    """
        #4) Based on set values and target slot, select and fill a CQ template
    """
    
    binary_question = target_slot is None
    
    if target_slot == "verb":
        candidates = {getInflection(c, "VBG")[0]:candidates[c] for c in candidates}
    
    if candidates and len(candidates)==1:
        set_values[target_slot] = list(candidates.keys())[0]
        binary_question = True
    
    if binary_question:
        verb = ""
        obj = ""
        if "verb" in set_values:
            verb = getInflection(set_values['verb'], "VBG")[0]
        if "direct_object" in set_values:
            obj = " ".join([set_values["direct_object_modifiers"], set_values["direct_object"]])
        if verb or obj:
            task = " ".join([verb, obj])   
            #Get rid of lingering internal whitespace
            task = " ".join(task.split()).strip()  
            return f"Are you interested in {task}?", candidates
    else:
        candidates = sorted(list(candidates.items()), key=lambda x:x[1], reverse=True)
        if max_results>0:
            candidates = [x[0] for x in candidates[:max_results]]
        else:
            candidates = [x[0] for x in candidates]
            
        if target_slot == "verb":
    #       return "Are you interested in doing any of the following?", candidates
            return f"Are you interested in {serialize_string_list(candidates)} something?", candidates

        elif target_slot == "direct_object":
            verb = getInflection(set_values['verb'], "VBG")[0]
#           return f"Are you interested in {verb} any of the following?", candidates
            return f"Are you interested in {verb} {serialize_string_list(candidates)}?", candidates

    return "Error",candidates


       
def get_cq_answers(task_df, target_slot, candidates, max_results):
    """
        #5) For each potential answer to the CQ, find the functions that satisfy that answer
    """
    if not target_slot:
        # Get all non-empty task_dfs
        entries = task_df["row"].unique()
        return {True: list(entries)}
    else:
        candidates = sorted(list(candidates.items()), key=lambda x:x[1], reverse=True)
        if max_results>0:
            candidates = [x[0] for x in candidates[:max_results]]
        else:
            candidates = [x[0] for x in candidates]
        
        entries = {c:list() for c in candidates}
        for c in candidates:
            for result_id in task_df["row"].unique():
                if not task_df[(task_df["row"] == result_id) & (task_df[target_slot] == c)].empty:
                    entries[c].append(result_id)
        return entries