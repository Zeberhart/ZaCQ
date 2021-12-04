""" Functions to extract and process tasks from docstrings.
    Note: Spacey requires tf >= 2.0, while the csn model uses <2.0
    
    Example:
        task_trees = extract_tasks(docsting)
        tasks:  = parse_tasks(task_trees)
"""
import re
from typing import List, Tuple
import itertools
from itertools import permutations
from collections import defaultdict
import time
from tqdm import tqdm

# import stanza
# from stanza.server import CoreNLPClient
import nltk
from nltk import tokenize
from simplenlg import *
from lemminflect import getLemma, getAllLemmas, getAllLemmasOOV, isTagBaseForm
from lemminflect import getInflection, getAllInflections, getAllInflectionsOOV
import spacy

import nltk.tree
from spacy.matcher import Matcher, DependencyMatcher
from spacy.symbols import ORTH, POS, NOUN, VERB
from spacy.language import Language
from pathlib import Path

rules_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "data", "tasks")


class TaskExtractor:

    object_prepositions = ["of"]
    
    def __init__(self, rules_dir=rules_path):
        self.nlp = spacy.load("en_core_web_trf")
        with open(os.path.join(rules_dir, "coderegex.txt"), "r") as c_file:
            self.regexes = [line.strip() for line in c_file if line.strip()]
        with open(os.path.join(rules_dir, "domain.txt"), "r") as d_file:
            self.domain_terms = [line.strip() for line in d_file if line.strip()]
        with open(os.path.join(rules_dir, "filtered_verbs.txt"), "r") as v_file:
            self.filtered_verbs = [line.strip() for line in v_file if line.strip()]
        with open(os.path.join(rules_dir, "generics.txt"), "r") as g_file:
            self.generic_terms = set([line.strip() for line in g_file if line.strip()])
        self.filtered_verb_lemmas = set([getLemma(verb, "VERB")[0] for verb in self.filtered_verbs])

    def extract_task_trees(self, docstrings: List[str]):
        """Extracts task trees from a docstring

        Args:
            docstring:

        Returns
        """
        for docstring in docstrings:
            sentences = self.preprocess_docstring(docstring)
            sentences, replacements = self.format_sentences(sentences)
            dependencies = self.extract_dependencies(sentences)
            task_trees = self.create_trees(dependencies, replacements)
            yield task_trees

    def extract_task_dicts(self, docstrings: List[str]):
        """Extracts task trees from a docstring

        Args:
            docstring:

        Returns
        """
        for docstring in docstrings:
            if not docstring:
                yield []
                continue
            sentences = self.preprocess_docstring(docstring)
            sentences, replacements = self.format_sentences(sentences)
            dependencies = self.extract_dependencies(sentences)
            if not dependencies:
                yield []
                continue
            task_dicts = self.create_dicts(dependencies, replacements)
            yield task_dicts


    def preprocess_docstring(self, docstring:str) -> List[str]:
        """Splits docstring into sentences and processes text for task analysis
        
        Args:
            docstring (str): A function's docstring. 
        
        Returns:
            Tuple[List[str], List[List[str]]]: Tuple containing the list of 
            processed sentences alongside a parallel list of OOV/domain-specific
            tokens substituted in each sentence 

        """
        docstring = self._remove_docstring_formatting(docstring)
        sentences = self._split_sentences(docstring)
        return [s.strip() for s in sentences if s.strip()]


    def _split_sentences(self, docstring:str) -> List[str]:
        """Splits docstrings into on newlines and relevant punctuation.
        """
        sentences = []
        
        sections = docstring.split("\n\n")
        for section in sections:
            lines = section.split("\n")
            sentence = ""
            num_space = 0
            for line in lines:
                new_num_space = len(re.split("\S", line)[0].replace("\t", "    "))
                line = line.strip()
                if line.startswith("@"):
                    if sentence:
                        sentences += tokenize.sent_tokenize(sentence)
                        sentence = ""
                    if len(line.split())>2:
                        sentence = line.split(maxsplit=2)[-1].strip()
#                 elif new_num_space != num_space:
#                     if sentence:
#                         sentences += tokenize.sent_tokenize(sentence)
#                     sentence = line
                elif line:
                    sentence += " " + line
                num_space = new_num_space
            if sentence:
                sentences += tokenize.sent_tokenize(sentence)
        return sentences


    def _remove_docstring_formatting(self, docstring) -> str:
        """Removes special formatting text from docstrings

        For instance, text like "{@link Foo}" is replaced with "Foo".
        We're assuming that this formatting information will not be used
        for downstream tasks.
        """
        #Java formatting {@class term} -> term
        docstring = re.sub(r"\{@\w+?\s+?#*(.*?)\}", r"\1", docstring)
        #Python formatting :attr:`(term)` -> term
        docstring = re.sub(r"\:\w+?\:\`(.*?)\`", r"\1", docstring)
        docstring = re.sub(r"\n\W*?\:\w+?\:\`(.*?)\`", r"\n\n\1", docstring)
        #Python formatting \n:attr (term): -> term
        docstring = re.sub(r"\n\W*?\:.+?\:", "\n\n", docstring)
        docstring = re.sub(r"<(\S+?)>", "", docstring)
        return docstring


    def format_sentences(self, sentences: list) -> Tuple[List[str], List[List[str]]]:
        """Prepares docstring sentences for task extraction.
        """
        #Stage 1: add periods/split sentences and remove parentheticals
        for i, sentence in enumerate(sentences):
            if sentence[-1] not in [".", "!", "?"]:
                sentence+="."
            for match in re.finditer(r"\W\(.+?\)\W", sentence):
                term = match[0].strip()
                sentences[i] = sentence.replace(term, "")
        #Stage 2: Replace regular expressions with generic token
        sentence_replacements = [[] for _ in sentences]
        for regex_str in self.regexes:
            for i, sentence in enumerate(sentences):
                for match in re.finditer(rf"{regex_str}", sentences[i]):
                    n = len(sentence_replacements[i])
                    term = match[0].strip()
                    if term and "rep_item" not in term:
                        sentence_replacements[i].append(term)
                        sentences[i] = re.sub(rf"{re.escape(term)}", f" rep_item_{n} ", sentences[i])
        for domain_term in self.domain_terms:
            for i, sentence in enumerate(sentences):
                for match in re.finditer(rf"\b{domain_term}\b", sentences[i]):
                    n = len(sentence_replacements[i])
                    term = match[0].strip()
                    if term and "rep_item" not in term:
                        sentence_replacements[i].append(term)
                        sentences[i] = re.sub(rf"\b{re.escape(term)}\b", f"rep_item_{n}", sentences[i])       
        #Step 3: Add "For"/"This" as needed based on initial word in sentence.
        
        for i, sentence in enumerate(sentences):
            tokens = sentence.split()
            firstword = tokens[0].lower()
            secondword = tokens[1].lower() if len(tokens)>2 else ""

            if firstword in ["function", "method"] and secondword == "to":
                sentences[i] = sentence.split(secondword, 1)[1]
#             if firstword in self.verbs_vbz:
#                 sentences[i] = "This " + sentence[0].lower()+sentence[1:]
#             elif firstword in self.verbs_vbg:
#                 sentences[i] = "For " + sentence[0].lower()+sentence[1:]
#             else:
                # # Code to get lemmas for OOV words
                # if not getAllLemmas(firstword):
                #     lemmas = getLemma(firstword, "VERB")
                # else:
            lemmas = getLemma(firstword, "VERB", lemmatize_oov=False)
            for lemma in lemmas:
                vbz = getInflection(lemma, tag="VBZ", )
                vbg = getInflection(lemma, tag="VBG")
                if firstword in vbz:
                    sentences[i] = "This " + sentence[0].lower()+sentence[1:]
                    break
                elif firstword in vbg:
                    sentences[i] = "For " + sentence[0].lower()+sentence[1:]
                    break
        sentences = [sentence.lower() for sentence in sentences]
        sentences = [" ".join(s.split()) for s in sentences]
        return (sentences, sentence_replacements)

    
    def _is_acceptable_object_preposition(self, prep_word):
            if prep_word in self.object_prepositions:
                return True
            else:
                return False
            
            
    def extract_dependencies(self, sentences):
        dependencies = []
        for sentence in sentences:
            doc = self.nlp(sentence)
            # Find all verbs in the doc
            matcher = DependencyMatcher(self.nlp.vocab)
            verb_pattern = [
                {
                    "RIGHT_ID": "verb",
                    "RIGHT_ATTRS": {"POS": "VERB"}
                }
            ]
            matcher.add("VERB", [verb_pattern])
            verb_matches = matcher(doc)
            verbs = [doc[match[1][0]] for match in verb_matches]
            # Trace relevant dependencies for each verb
            sentence_dependencies = []
            for verb in verbs:
#                 if getLemma(verb.text, "VERB")[0] not in self.verb_lemmas:
                if getLemma(verb.text, "VERB")[0] in self.filtered_verb_lemmas:
                    continue
                neg = None
                prt = None
                objs = []
                preps = []
                objpreps = defaultdict(list)

                # Get all objects and prepositions for verb
                for child in verb.children:
                    if child.dep_ == "neg":
                        neg=child
                    elif child.dep_ == "prt":
                        prt = child
                    elif child.dep_ == "prep":
                        preps.append(child)
                    elif child.dep_ in ["dobj", "nsubjpass"]:
                        objs.append(child)
                        for obj_child in child.children:
                            if obj_child.dep_ =="prep" and not self._is_acceptable_object_preposition(obj_child.text):
                                preps.append(obj_child)
                                
                if verb.dep_ == "relcl":
                    if not (verb.head.dep_ == "pobj" and verb.head.head.text == "of"):
                        objs.append(verb.head)

                # Separate compound objects and prepositions
                objs += [conjunct for obj in objs for conjunct in obj.conjuncts]
                preps += [conjunct for prep in preps for conjunct in prep.conjuncts]

                # Create all task tuples with as much detail as possible
                verb_tuple = (verb,)
                if prt: verb_tuple = verb_tuple+(prt,)
                # if neg: verb_tuple = verb_tuple+(neg,)
                if objs:
                    for obj in objs:
                        if preps:
                            for prep in preps:
                                sentence_dependencies.append(verb_tuple+(obj, prep))
                        else:
                            sentence_dependencies.append(verb_tuple+(obj,))  
                elif preps:
                    for prep in preps:
                        sentence_dependencies.append(verb_tuple+(prep,))
            dependencies.append(sentence_dependencies)
        return dependencies

    def create_dicts(self, dependencies, sentence_replacements):
        def is_acceptable(object_word):
            if (object_word.lower() not in self.generic_terms
                and re.search('[a-zA-Z]', object_word) is not None
                and len(object_word)>1):
                return True
            else:
                return False
        
        def recoverWord(token, replacement_idx):
            if token.text.startswith("rep_item_"):
                t = token.text.split("_")[-1]
                i = int(re.split(r'\D+',t)[0])
                return sentence_replacements[replacement_idx][i].strip().rstrip(",.")
            else:
                return token.text.strip().rstrip(",.")
          
        def processObject(obj_token, i):
            obj_dict = {
                "object_det":"",
                "object":"",
                "object_modifiers":[],
            }
            # Check if child is generic, and if so, skip
            obj_word = recoverWord(obj_token, i)
            compound_words = ""
            if is_acceptable(obj_word):
                obj_dict["object"] = obj_word
                # If child is noun, process noun phrase and add
                #if obj_token.pos_ == "NOUN":
                for obj_child in obj_token.children:
                    modifier_word = recoverWord(obj_child, i)
                    if obj_child.dep_ == "det":
                        obj_dict["object_det"]  = modifier_word
                    elif obj_child.dep_ in ["compound", "nmod"]:
                        compound_word = ""
                        compounds = [obj_child]
                        while compounds:
                            current_compound_obj = compounds.pop(0)
                            current_compount_word = recoverWord(current_compound_obj, i)
                            if current_compound_obj.dep_ in ["compound", "nmod"] and is_acceptable(current_compount_word):
                                compound_word = f"{current_compount_word} {compound_word}".strip()
                                for compound_child in current_compound_obj.children:
                                    compounds.insert(0, compound_child)
                        compound_words = f"{compound_words} {compound_word}".strip()
                    elif obj_child.dep_ in ["amod", "nummod"] and is_acceptable(modifier_word):
                        obj_dict["object_modifiers"].append(modifier_word)
                    elif obj_child.dep_ =="prep":
                        prep_word = obj_child.text
                        if self._is_acceptable_object_preposition(prep_word):
                            prep_children = [t for t in obj_child.children]
                            while prep_children and prep_children[0].dep_ == "prep":
                                prep_word = " ".join([prep_word, prep_children[0].text])
                                prep_children = [t for t in prep_children[0].children]
                            if len(prep_children) >= 1:
                                prep_obj_dict = processObject(prep_children[0], i)
                                if prep_obj_dict:
                                    prep_obj_word = prep_obj_dict["object"]
                                    if prep_obj_dict["object_modifiers"]:
                                        prep_obj_modifiers = " ".join(prep_obj_dict["object_modifiers"])
                                        prep_obj_word = " ".join([prep_obj_modifiers, prep_obj_word])
                                    if prep_obj_dict["object_det"]:
                                        prep_obj_word = " ".join([prep_obj_dict['object_det'], prep_obj_word])
                                    obj_dict["object"] = " ".join([obj_dict['object'], prep_word, prep_obj_word])  
                obj_dict["object"] = f"{compound_words} {obj_dict['object']}".strip()     
            if obj_dict["object"]: 
                return obj_dict
            else:
                return None
                    
            
        #Step 1: Import generic terms and domain verbs
#         use_domain_verbs = True

        # Step 1.5: Filter out not-in-domain verbs
#         if use_domain_verbs:
#             for i, sentence_dependencies in enumerate(dependencies):
#                 filtered_sentence_dependencies = []
#                 for j, dependency_tuple in enumerate(sentence_dependencies):
#                     verb = recoverWord(dependency_tuple[0], i)
#                     lemma = getLemma(verb, "VERB")[0]
#                     if lemma in self.verb_lemmas:
#                         filtered_sentence_dependencies.append(dependency_tuple)
#                 dependencies[i] = filtered_sentence_dependencies
                
        # Step 2: Create trees, filtering out generic terms and incomplete tasks
        task_dicts = []
        for i, sentence_dependencies in enumerate(dependencies):    
            sentence_task_dicts = []
            for j, dependency_tuple in enumerate(sentence_dependencies):
                #Extract all children of top-level verb phrase
                task_dict = {
                    "verb":"",
                    "particle":"",
                    "direct_object_det":"",
                    "direct_object":"",
                    "direct_object_modifiers":[],
                    "preposition":"",
                    "preposition_object_det":"",
                    "preposition_object":"",
                    "preposition_object_modifiers":[],
                }
                task_dict["verb"] = recoverWord(dependency_tuple[0], i)
                for token in dependency_tuple[1:]:
                    token_word = recoverWord(token, i)
                    
                    # 1. Prepositions
                    # Leave out if prep. object is generic term
                    # Otherwise, leave out any generic modifiers
                    if token.dep_ == "prep":
                        prep_word = token.text
                        token_children = [t for t in token.children]
                        while token_children and token_children[0].dep_ == "prep":
                            prep_word = " ".join([prep_word, token_children[0].text])
                            token_children = [t for t in token_children[0].children]
                        if len(token_children) >= 1:
                            prep_obj_dict = processObject(token_children[0], i)
                            if prep_obj_dict:
                                task_dict["preposition"] = prep_word
                                task_dict["preposition_object"] = prep_obj_dict["object"]
                                task_dict["preposition_object_modifiers"] = prep_obj_dict["object_modifiers"]
                                task_dict["preposition_object_det"] = prep_obj_dict["object_det"]
                    # 2: Direct objects
                    # Leave out if direct object is generic term
                    # Otherwise, leave out any generic modifiers
                    else:
                        direct_obj_dict = processObject(token, i)
                        if direct_obj_dict:
                            task_dict["direct_object"] = direct_obj_dict["object"]
                            task_dict["direct_object_modifiers"] = direct_obj_dict["object_modifiers"]
                            task_dict["direct_object_det"] = direct_obj_dict["object_det"]
                if task_dict["preposition_object"] or task_dict["direct_object"]:        
                    sentence_task_dicts.append(task_dict)
            task_dicts.append(sentence_task_dicts)
        return task_dicts  

        
    def parse_task_trees(self, task_trees):
        for sentence_task_trees in task_trees:
            for sentence_task_tree in sentence_task_trees:
                tokens = sentence_task_tree.leaves()
                verb = tokens[0]
                lemma = getLemma(verb, "VERB")[0]
                verb = getInflection(lemma, "VB")[0]
                tokens[0] = verb
                task = " ".join(tokens)
                yield task

    def parse_task_dicts(self, task_dicts):
        for sentence_task_dicts in task_dicts:
            for task_dict in sentence_task_dicts:
                lemma = getLemma(task_dict["verb"], "VERB")[0]
                verb = getInflection(lemma, "VB")[0]
                tokens = [verb]
                if task_dict["particle"]:
                    tokens.append(task_dict["particle"])
                if task_dict["direct_object_modifiers"]:
                    tokens += task_dict["direct_object_modifiers"]
                if task_dict["direct_object"]:
                    tokens.append(task_dict["direct_object"])
                if task_dict["preposition"]:
                    tokens.append(task_dict["preposition"])
                if task_dict["preposition_object_modifiers"]:
                    tokens += task_dict["preposition_object_modifiers"]
                if task_dict["preposition_object"]:
                    tokens.append(task_dict["preposition_object"])
                task = " ".join(tokens)
                yield task

    def lemmatize_dicts(self, task_dicts):
        for i, task_dict in enumerate(task_dicts):
            if "verb" in task_dict:
                lemma = getLemma(task_dict["verb"], "VERB")
                task_dict["verb"] = lemma[0]
#             if "direct_object" in task_dict:
#                 lemma = getLemma(task_dict["direct_object"], "NOUN", False)
#                 if lemma: task_dict["direct_object"] = lemma[0]
#             if "preposition_object" in task_dict:
#                 lemma = getLemma(task_dict["preposition_object"], "NOUN", False)
#                 if lemma: task_dict["preposition_object"] = lemma[0]
        return task_dicts

if __name__ == "__main__":
    docstrings = [
        "the file to which the current properties and their values will be written",
        "converting OmegaObj.time4peas to ints"
    ]
    task_extractor = TaskExtractor()
    for tasks in task_extractor.extract_task_dicts(docstrings):
        for task in task_extractor.parse_task_dicts(tasks):
            print(task)
