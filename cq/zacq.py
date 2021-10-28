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

import tasks

task_extractor = tasks.TaskExtractor()

NO_QUESTION = ("", {}, None, {})



def generate_cq(docstrings=None, 
                indices = None,
                task_df=None, 
                set_values=defaultdict(str), 
                not_values=[], 
                max_results=5,
                infer_support=14, 
                infer_conf=.23, 
                option_support=1,
                query=None):
    """
        Top-level method to generate a clarification question given a lit of docstrings
        
        Returns the question string, as well as the functions associated w/ each answer
    """
    if docstrings:
        task_df = create_task_df(docstrings, incides)
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
    
    if len(task_df["row"].unique())<=1 or len(task_df.drop('row', axis=1).drop_duplicates())<=1:
        return NO_QUESTION
    
    if not set_values and not not_values and query:
        query=create_task_df([query], indices=None, task_extractor=task_extractor)
        if not query.empty:
            query = query.iloc[0]
        else:
            query=None
    else:
        query=None
        
    inferred_task_df, inferred_set_values, inferred_values = infer_best_slots(
                                            task_df, 
                                            set_values,
                                            infer_support=infer_support, 
                                            infer_conf=infer_conf,
                                            option_support=option_support,
                                            query=query)
    target_slot, candidates = select_target_slot(
                                            inferred_task_df, 
                                            inferred_set_values,
                                            inferred_values, 
                                            min_support=option_support)
    if candidates and len(candidates)==1:
        inferred_value=list(candidates)[0]
        if target_slot!="role":
            inferred_values[target_slot] = inferred_value
            inferred_set_values[target_slot] = inferred_value
            inferred_task_df = apply_slot_value(inferred_task_df, target_slot, inferred_value)
        else:
            verb, do, prep, po = inferred_value
            for slot, slot_value in {"verb": verb, 
                                     "direct_object": do, 
                                     "preposition": prep, 
                                     "preposition_object":po}.items():
                if slot_value:
                    inferred_values[slot] = slot_value
                    inferred_set_values[slot] = slot_value
                    inferred_task_df = apply_slot_value(inferred_task_df, slot, slot_value)
        target_slot = candidates = None
    
    if inferred_values or target_slot:
        cq, gerund_candidates = create_string(inferred_set_values, target_slot, candidates, max_results, inferred_task_df=inferred_task_df, query=query)
        answers = get_cq_answers(inferred_task_df, target_slot, candidates, max_results)
        
        return(cq, inferred_values, target_slot, answers)
    else:
        return NO_QUESTION
        


def create_task_df(docstrings, indices=None, task_extractor=None):
    """
         #1) Create dataframes storing each docstring's tasks
    """
    if not task_extractor:
        print("Need task extractor")
        return None
    tasks = [sum(t, []) for t in task_extractor.extract_task_dicts(docstrings)]
    tasks = [task_extractor.lemmatize_dicts(t) for t in tasks]
    if indices: task_sets = zip(indices, tasks)
    else: task_sets = enumerate(tasks)
    for i, task_set in task_sets:
        for task in task_set:
            task["row"] = i
    tasks = sum(tasks, [])
    task_df = pd.DataFrame(tasks)
    if not task_df.empty:
        task_df = task_df.explode("direct_object_modifiers").explode("preposition_object_modifiers").replace(np.nan, '', regex=True)
    normalize_task_df(task_df)
    return task_df

def normalize_task_df(task_df):
    for column in task_df:
        if column=="row":continue
        for val in task_df[column].unique():
            task_df.loc[task_df[column]==val, column]=val.lower().strip(punctuation)
            
#         normalized_vals = {}
#         for val in task_df[column].unique():
#             normalized_val = val.lower().strip(punctuation)
#             if normalized_val in normalized_vals:
#                 task_df.loc[task_df[column]==val, column]=normalized_vals[normalized_val]
#             else:
#                 normalized_vals[normalized_val]=val 

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


def get_next_slot(task_df, set_values):
    """
        #2.1) Determine the next best slot to ask about
        
        Returns the slot, the slot-value w/ the highest support, the support and conf. of that value, and all other candidate values
    """
    values_to_search = []
    if not set_values:
        values_to_search+=["verb", "object"]
    elif ("object" in set_values 
        and "direct_object" not in set_values 
        and "preposition_object" not in set_values):
        values_to_search.append("role")
        if "verb" not in set_values:
            values_to_search.append("verb")
        if not "object_modifiers" in set_values:
            values_to_search.append("object_modifiers")
    elif "verb" in set_values:
        if "preposition" in set_values and "preposition_object" not in set_values:
            values_to_search.append("preposition_object")
        elif "preposition_object" in set_values and "preposition" not in set_values:
            values_to_search.append("preposition")
        else:
            if "direct_object" not in set_values:
                values_to_search.append("direct_object")  
            elif "direct_object_modifiers" not in set_values:
                values_to_search.append("direct_object_modifiers")
            if "preposition_object" not in set_values:
                values_to_search.append("preposition_object")  
                values_to_search.append("preposition")  
            elif "preposition_object_modifiers" not in set_values:
                values_to_search.append("preposition_object_modifiers")

        

    # Count all unique values for each potential value type
    value_candidates = defaultdict(lambda: Counter())        
    for value_type in values_to_search:
        if value_type == "role":
            obj=set_values["object"]
            for result_id in task_df["row"].unique():
                #Get all "roles" in that result (that is, the verb, and the object as a DO, or PO w/ prep)
                unique_do_verbs = task_df.loc[(task_df["row"]==result_id) & (task_df["direct_object"]==obj)]["verb"].unique()
                for verb in unique_do_verbs:
                    if isinstance(verb,str) and verb.strip(): value_candidates[value_type][(verb,obj,None,None)] += 1
                unique_po_verbs = task_df.loc[(task_df["row"]==result_id) & (task_df["preposition_object"]==obj)]["verb"].unique()
                for verb in unique_po_verbs:
                    if isinstance(verb,str) and verb.strip():
                        unique_preps = task_df.loc[(task_df["verb"]==verb) & (task_df["row"]==result_id) & (task_df["preposition_object"]==obj)]["preposition"].unique()
                        for prep in unique_preps:
                            if isinstance(prep,str) and prep.strip(): value_candidates[value_type][(verb,None,prep,obj)] += 1
        elif value_type not in ["object", "object_modifiers"]:
            for result_id in task_df["row"].unique():
                unique_values = task_df.loc[task_df["row"]==result_id][value_type].unique()
                for value in unique_values:
                    if isinstance(value,str) and value.strip(): value_candidates[value_type][value] += 1
        elif value_type == "object":
             for result_id in task_df["row"].unique():
                unique_dos = list(task_df.loc[task_df["row"]==result_id]["direct_object"].unique())
                unique_pos = list(task_df.loc[task_df["row"]==result_id]["preposition_object"].unique())
                unique_values = unique_dos + unique_pos
                for value in unique_values:
                    if value and isinstance(value,str) and value.strip(): value_candidates[value_type][value] += 1
        elif value_type == "object_modifiers":
            for result_id in task_df["row"].unique():
                unique_dos = list(task_df.loc[task_df["row"]==result_id]["direct_object_modifiers"].unique())
                unique_pos = list(task_df.loc[task_df["row"]==result_id]["preposition_object_modifiers"].unique())
                unique_values = unique_dos + unique_pos
                for value in unique_values:
                    if value and isinstance(value,str) and value.strip(): value_candidates[value_type][value] += 1

    # If there are any candidate slot/values, return the best one and all candidates
    if value_candidates.keys():
        max_slot = max(value_candidates.keys(), key=lambda k: value_candidates[k].most_common(1)[0][1])
        max_value, max_value_support = value_candidates[max_slot].most_common(1)[0]
        max_value_conf = max_value_support/sum(value_candidates[max_slot].values())
        return max_slot, max_value, max_value_support, max_value_conf, value_candidates
    else:
        return None
    

def infer_best_slots(task_df, set_values, infer_support=2, infer_conf=.5, option_support=1, query=None):
    """
        #2) Automatically infer attributes given common patterns in the tasks
        
        Returns a filtered set of task dataframes and a dict storing the inferred values
    """
    inferred_values = defaultdict(str)
    slot_tuple = get_next_slot(task_df, set_values)
    while slot_tuple:
        max_slot, max_value, max_value_support, max_value_conf, value_candidates = slot_tuple
        if query is not None:
            # Query checks
            if max_slot == "role":
                found_one=False
                for verb, do, prep, po in value_candidates[max_slot]:
                    if ((not verb or query["verb"]==verb) 
                        and (not do or query["direct_object"]==do)
                        and (not prep or query["preposition"]==prep) 
                        and (not po or query["preposition_object"]==po) 
                    ):
                        for slot, slot_value in {"verb": verb, 
                                                 "direct_object": do, 
                                                 "preposition": prep, 
                                                 "preposition_object":po}.items():
                                if slot_value:
                                    set_values[slot] = slot_value
                                    inferred_values[slot] = slot_value
                                    task_df = apply_slot_value(task_df, slot, slot_value)
                        slot_tuple = get_next_slot(task_df, set_values)
                        found_one=True
                        break
                if found_one:
                    continue   
            elif max_slot == "object":
                if query["direct_object"] in value_candidates[max_slot]:
                    set_values[max_slot] = query["direct_object"]
                    inferred_values[max_slot] = query["direct_object"]
                    task_df = apply_slot_value(task_df, max_slot, query["direct_object"])
                    slot_tuple = get_next_slot(task_df, set_values)
                    continue
                elif query["preposition_object"] in value_candidates[max_slot]:
                    set_values[max_slot] = query["preposition_object"]
                    inferred_values[max_slot] = query["preposition_object"]
                    task_df = apply_slot_value(task_df, max_slot, query["preposition_object"])
                    slot_tuple = get_next_slot(task_df, set_values)
                    continue
            elif max_slot == "object_modifiers":
                if query["direct_object_modifiers"] in value_candidates[max_slot]:
                    set_values[max_slot] = query["direct_object_modifiers"]
                    inferred_values[max_slot] = query["direct_object_modifiers"]
                    task_df = apply_slot_value(task_df, max_slot, query["direct_object_modifiers"])
                    slot_tuple = get_next_slot(task_df, set_values)
                    continue
                elif query["preposition_object_modifiers"] in value_candidates[max_slot]:
                    set_values[max_slot] = query["preposition_object_modifiers"]
                    inferred_values[max_slot] = query["preposition_object_modifiers"]
                    task_df = apply_slot_value(task_df, max_slot, query["preposition_object_modifiers"])
                    slot_tuple = get_next_slot(task_df, set_values)
                    continue    
            elif max_slot != "preposition" and query[max_slot] in value_candidates[max_slot]:
                set_values[max_slot] = query[max_slot]
                inferred_values[max_slot] = query[max_slot]
                task_df = apply_slot_value(task_df, max_slot, query[max_slot])
                slot_tuple = get_next_slot(task_df, set_values)
                continue
        if ((max_value_support >= infer_support and max_value_conf>=infer_conf)
            or (max_slot in ["preposition", "preposition_object"] 
                and set(inferred_values).isdisjoint(set(["preposition", "preposition_object"]))
                and (sum(value_candidates[max_slot].values())>=option_support or not inferred_values))):
            if max_slot != "role":
                set_values[max_slot] = max_value
                inferred_values[max_slot] = max_value
                task_df = apply_slot_value(task_df, max_slot, max_value)
            else:
                verb, do, prep, po = max_value
                for slot, slot_value in {"verb": verb, "direct_object": do, "preposition": prep, "preposition_object":po}.items():
                    if slot_value:
                        set_values[slot] = slot_value
                        inferred_values[slot] = slot_value
                        task_df = apply_slot_value(task_df, slot, slot_value)
            slot_tuple = get_next_slot(task_df, set_values)
            continue
        break
    
    return task_df, set_values, inferred_values


def select_target_slot(task_df, set_values, inferred_values, min_support=5):
    """
        #3) Determine which slot the CQ should ask about. If there isn't a slot
        with enough support, instead ask to confirm the inferred values
        
        Returns a tuple: either None, None, indicating that the CQ should confirm inferred values;
        or the target slot and candidate values for that slot
    """
    
    """
        If we've inferred prep or PO (but not both), the next q MUST be a multiple choice for the remaining one
        If we haven't inferred either, but the next slot to ask about is one:
            If we've inferred NOTHING, infer that one and then ask about the other one
            If we've inferred something,
                If we have slot_support sufficient to normally do multiple choice, infer it then ask
                Otherwise, just ask about inference as normal
    """ 
    
    slot_tuple = get_next_slot(task_df, set_values)
    if slot_tuple:
        max_slot, max_value, max_value_support, max_value_conf, value_candidates = slot_tuple
        slot_support = sum(value_candidates[max_slot].values())
    if slot_tuple and (("preposition" in set_values and not "preposition_object" in set_values) or (not "preposition" in set_values and "preposition_object" in set_values)):
        return max_slot, value_candidates[max_slot]
    elif not slot_tuple or (inferred_values and slot_support<min_support):
        return None, None
    else:
        if("object" in set_values 
        and "verb" == max_slot):
            return "role", value_candidates["role"]
        else:
            return max_slot, value_candidates[max_slot]
            
            
def serialize_string_list(candidates):
    candidates.sort()
    if len(candidates)==2:
        obj = f"{candidates[0]} or {candidates[1]}"
    else:
        obj = ""
        for i in range(len(candidates)-1):
            obj += candidates[i] + ", "
        obj += "or "+candidates[len(candidates)-1]
    return obj
        
    
def create_string(set_values, target_slot, candidates, max_results, inferred_task_df=None, query=None):
    """
        #4) Based on set values and target slot, select and fill a CQ template
    """
    
    binary_question = target_slot is None
    
    if target_slot == "verb":
        candidates = {getInflection(c, "VBG")[0]:candidates[c] for c in candidates}
    
    if candidates and len(candidates)==1:
        if target_slot != "role":
            set_values[target_slot] = list(candidates.keys())[0]
        else:
            v, do, p, po = list(candidates.keys())[0]
            set_values["verb"] = v
            if do:
                set_values["direct_object"] = do
            elif po:
                set_values["preposition"] = p
                set_values["preposition_object"] = po
        binary_question = True
    
    for k, v in set_values.items():
        if not v:
            del set_values[k]
            
    if binary_question:

        verb = ""
        dobj = ""
        prep = ""
        obj = ""
        
        if "verb" in set_values:
            verb = getInflection(set_values['verb'], "VBG")[0]
        if "direct_object" in set_values:
            dobj = " ".join([set_values["direct_object_modifiers"], set_values["direct_object"]])
        if "preposition_object" in set_values:
            prep = " ".join([set_values["preposition"], set_values["preposition_object_modifiers"], set_values["preposition_object"]])
        if "object" in set_values:
            obj = " ".join([set_values["object_modifiers"], set_values["object"]])
        if prep and not dobj and inferred_task_df is not None and inferred_task_df["direct_object"].all:
            if len(list(set(inferred_task_df["direct_object"])))==1:
                dobj = list(set(inferred_task_df["direct_object"]))[0]
            elif query is None or not query["direct_object"]:
                dobj = "something"
        if dobj or prep:
            task = " ".join([verb, dobj, prep])   
        else:
            task = " ".join([verb, obj])   
        #Get rid of lingering internal whitespace
        task = " ".join(task.split()).strip()  
        
        if query is not None:
            for slot, value in set_values.items():
                if slot not in ["role", "object", "object_modifiers"] and value and value != query[slot]:
                    return f"Are you interested in {task}?", candidates
            num_funcs = len(inferred_task_df['row'].unique())
            return f"Found {num_funcs} function{'s' if num_funcs>1 else ''} that specifically mention{'s' if num_funcs==1 else ''} {task}. Would you like to see {'them' if num_funcs>1 else 'it'} first?", candidates
        return f"Are you interested in {task}?", candidates

        
    else:
        candidates = sorted(list(candidates.items()), key=lambda x:x[1], reverse=True)
        if max_results>0:
            candidates = [x[0] for x in candidates[:max_results]]
        else:
            candidates = [x[0] for x in candidates]
            
        if target_slot == "object":
    #         return "Are you looking for any of the following?", candidates
            if len(candidates)==2:
                return f"Are you looking for {serialize_string_list(candidates)}?", candidates
            else:
                return f"Are you looking for any of the following?", candidates
#                 return f"Are you looking for any of the following: {serialize_string_list(candidates)}?", candidates

        elif target_slot == "verb":
            if "object" not in set_values:
    #             return "Are you interested in doing any of the following?", candidates
                if len(candidates)==2:
                    return f"Are you interested in {serialize_string_list(candidates)} something?", candidates
                else:
                    return f"Are you looking to perform any of these actions?", candidates
#                     return f"Are you interested in doing any of the following: {serialize_string_list(candidates)}?", candidates
            elif "object" in set_values:
                obj = " ".join([set_values["object_modifiers"], set_values["object"]]).strip()
    #             return f"Are you interested in doing any of the following with {obj}?", candidates  
                if len(candidates)==2:
                    return f"Are you interested {serialize_string_list(candidates)} with {obj}?", candidates 
                else:
                    return f"Are you looking to perform any of these actions with {obj}?", candidates 
#                     return f"Are you interested in doing any of the folowing with {obj}: {serialize_string_list(candidates)}?", candidates 

        elif target_slot == "role":
            obj = " ".join([set_values["object_modifiers"], set_values["object"]]).strip()
            correct_verb_cands = [(getInflection(c[0], "VBG")[0],)+c[1:] for c in candidates]
            str_candidates = [" ".join([word for word in c if word]) for c in correct_verb_cands]
    #         return f"Are you interested in doing any of the following with {obj}?", candidates 
            if len(candidates)==2:
                return f"Are you interested in {serialize_string_list(str_candidates)}?", candidates  
            else:
                return f"Are you looking to perform any of these actions?", candidates  
#                 return f"Are you interested in doing any of the following: {serialize_string_list(str_candidates)}?", candidates  

        elif target_slot == "direct_object":
            if "verb" in set_values:
                verb = getInflection(set_values['verb'], "VBG")[0]
                if "preposition_object" in set_values:
                    prep = set_values["preposition"]
                    prep_obj = " ".join([set_values["preposition_object_modifiers"], set_values["preposition_object"]]).strip()
#                     return f"Are you interested in {verb} {serialize_string_list(candidates)} {prep} {prep_obj}?", candidates
                    if len(candidates)==2:
                        return f"Are you interested in {verb} {serialize_string_list(candidates)} {prep} {prep_obj}?", candidates
                    else:
#                         return f"Are you interested in {verb} any of the following {prep} {prep_obj}: {serialize_string_list(candidates)}?", candidates
                        return f"Are you interested in {verb} any of the following {prep} {prep_obj}?", candidates
                else:
#                     return f"Are you interested in {verb} {serialize_string_list(candidates)}?", candidates
                    if len(candidates)==2:
                    
                        return f"Are you interested in {verb} {serialize_string_list(candidates)}?", candidates
                    else:
                        return f"Are you interested in {verb} any of the following?", candidates
#                         return f"Are you interested in {verb} any of the following: {serialize_string_list(candidates)}?", candidates
                    
            else:
    #             return "Are you looking for any of the following?", candidates 
                if len(candidates)==2:
                    return f"Are you looking for {serialize_string_list(candidates)}", candidates 
                else:
                    return f"Are you looking for any of the following: {serialize_string_list(candidates)}", candidates 

        elif target_slot=="preposition_object":
            verb = getInflection(set_values["verb"], "VBG")[0]
            prep = set_values["preposition"]
            dobj = ""
            if "direct_object" in set_values:
                dobj = " ".join([set_values["direct_object_modifiers"], set_values["direct_object"]]).strip()
            elif inferred_task_df is not None and inferred_task_df["direct_object"].all:
                if len(list(set(inferred_task_df["direct_object"])))==1:
                    dobj = list(set(inferred_task_df["direct_object"]))[0]                    
                elif query is None or not query["direct_object"]:
                    dobj = "something"
            if dobj:
                if len(candidates)==2:
    #               return f"Are you interested in {verb} {dir_obj} {prep} {serialize_string_list(candidates)}?", candidates
                    return f"Are you interested in {verb} {dobj} {prep} {serialize_string_list(candidates)}?", candidates
                else:
                    return f"Are you interested in {verb} {dobj} {prep} any of the following?", candidates
#                     return f"Are you interested in {verb} {dobj} {prep} any of the following: {serialize_string_list(candidates)}?", candidates
                    
#             return f"Are you interested in {verb} {prep} {serialize_string_list(candidates)}?", candidates
            if len(candidates)==2:
                return f"Are you interested in {verb} {prep} {serialize_string_list(candidates)}?", candidates
            else:
                return f"Are you interested in {verb} {prep} any of the following?", candidates
#                 return f"Are you interested in {verb} {prep} any of the following: {serialize_string_list(candidates)}?", candidates
                

        elif target_slot=="preposition":
            verb = getInflection(set_values["verb"], "VBG")[0]
            verb = set_values["verb"]
            prep_obj = set_values["preposition_object"]
            dobj = ""
            if "direct_object" in set_values:
                dobj = " ".join([set_values["direct_object_modifiers"], set_values["direct_object"]]).strip()
            elif inferred_task_df is not None and inferred_task_df["direct_object"].all:
                if len(list(set(inferred_task_df["direct_object"])))==1:
                    dobj = list(set(inferred_task_df["direct_object"]))[0]
                elif query is None or not query["direct_object"]:
                    dobj = "something"
            if dobj:
    #           return f"Are you interested in {verb} {dobj} {serialize_string_list(candidates)} {prep_obj}?", candidates
                if len(candidates)==2:
                    return f"Are you interested in {verb} {dobj} {serialize_string_list(candidates)} {prep_obj}?", candidates
                else:
                    return f"How would you like to {verb} {dobj}: {serialize_string_list(candidates)} {prep_obj}?", candidates

#             return f"Are you interested in {verb} {serialize_string_list(candidates)} {prep_obj}?", candidates
            if len(candidates)==2:
                return f"Are you interested in {verb} {serialize_string_list(candidates)} {prep_obj}?", candidates
            else:
                return f"How would you like to {verb}: {serialize_string_list(candidates)} {prep_obj}?", candidates

        elif target_slot=="object_modifiers":
            obj = set_values["object"]
#             return f"Are you interested in {serialize_string_list(candidates)} {obj}?", candidates
            if len(candidates)==2:
                return f"Are you looking for {serialize_string_list(candidates)} {obj}?", candidates
            else:
                return f"What kind of {obj} are you looking for?", candidates
#                 return f"What kind of {obj} are you looking for: {serialize_string_list(candidates)}?", candidates

        elif target_slot=="direct_object_modifiers":
            verb_phrase = getInflection(set_values["verb"], "VBG")[0]
            dir_obj = set_values["direct_object"]
            if "preposition_object" in set_values:
                prep_obj =  " ".join([set_values["preposition_object_modifiers"], set_values["preposition_object"]]).strip()
                prep = set_values["preposition"]
#                 return f"Are you interested in {verb_phrase} {serialize_string_list(candidates)} {dir_obj} {prep} {prep_obj}?", candidates
                if len(candidates)==2:
                    return f"Are you interested in {verb_phrase} {serialize_string_list(candidates)} {dir_obj} {prep} {prep_obj}?", candidates
                else:
                    return f"What kind of {dir_obj} are you interested in {verb_phrase} {prep} {prep_obj}?", candidates
#                     return f"What kind of {dir_obj} are you interested in {verb_phrase} {prep} {prep_obj}: {serialize_string_list(candidates)}?", candidates
#             return f"Are you interested in {verb_phrase} {serialize_string_list(candidates)} {dir_obj}?", candidates
            if len(candidates)==2:
                return f"Are you interested in {verb_phrase} {serialize_string_list(candidates)} {dir_obj}?", candidates
            else:
#                 return f"What kind of {dir_obj} are you interested in {verb_phrase}: {serialize_string_list(candidates)}?", candidates
                return f"What kind of {dir_obj} are you interested in {verb_phrase}?", candidates

        elif target_slot=="preposition_object_modifiers":
            verb_phrase = getInflection(set_values["verb"], "VBG")[0]
            prep_obj = set_values["preposition_object"]
            prep = set_values["preposition"]
            if "direct_object" in set_values:
                dir_obj = " ".join([set_values["direct_object_modifiers"], set_values["direct_object"]]).strip()
                verb_phrase += f" {dir_obj}"
            elif inferred_task_df is not None and inferred_task_df["direct_object"].all:
                if len(list(set(inferred_task_df["direct_object"])))==1:
                    dobj = list(set(inferred_task_df["direct_object"]))[0]
                    verb_phrase += f" {dobj}"
                elif query is None or not query["direct_object"]:
                    dobj = "something"
                    verb_phrase += f" {dobj}"
#         return f"Are you interested in {verb_phrase} {prep} {serialize_string_list(candidates)} {prep_obj}?", candidates
#         return f"{prep.capitalize()} what kind of {prep_obj} are you interested in {verb_phrase}: {serialize_string_list(candidates)}?", candidates
        if len(candidates)==2:
            return f"Are you interested in {verb_phrase} {prep} {serialize_string_list(candidates)} {prep_obj}?", candidates
        else:
#             return f"What kind of {prep_obj} are you interested in {verb_phrase} {prep}: {serialize_string_list(candidates)}?", candidates
            return f"What kind of {prep_obj} are you interested in {verb_phrase} {prep}?", candidates

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
            if target_slot == "role":
                verb, do, prep, po = c
                for result_id in task_df["row"].unique():
                    if do:
                        if not task_df[(task_df["row"] == result_id) & (task_df["verb"] == verb) & (task_df["direct_object"] == do)].empty:
                            entries[c].append(result_id)
                    else:
                        if not task_df[(task_df["row"] == result_id) & (task_df["verb"] == verb) & (task_df["preposition"] == prep) & (task_df["preposition_object"] == po)].empty:
                            entries[c].append(result_id)
            elif target_slot == "object_modifiers":
                for result_id in task_df["row"].unique():
                    if not task_df[(task_df["row"] == result_id & (task_df["direct_object_modifiers"] == c) | (task_df["preposition_object_modifiers"] == c))].empty:
                        entries[c].append(result_id)
            elif target_slot == "object":
                for result_id in task_df["row"].unique():
                    if not task_df[(task_df["row"] == result_id) & ((task_df["direct_object"] == c) | (task_df["preposition_object"] == c))].empty:
                        entries[c].append(result_id)
            else:
                for result_id in task_df["row"].unique():
                    if not task_df[(task_df["row"] == result_id) & (task_df[target_slot] == c)].empty:
                        entries[c].append(result_id)
        return entries