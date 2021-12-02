# Clarifying Questions for Query Refinement in Source Code Search

## This repository is currently a WIP. More details and a quickstart script will be available shortly. 

This code is part of the reproducibility package for the SANER 2022 paper "Generating Clarifying Questions for Query Refinement in Source Code Search".

It consists of three parts:

* codesearch/ - API to access the CodeSearchNet datasets and neural bag-of-words code retrieval method.

* cq/ - Implementation of the ZaCQ system, including an implementation of the the [TaskNav](https://www.cs.mcgill.ca/~swevo/tasknav/) development task extraction algorithm and two baseline query refinement methods.

* evaluation/ - Scripts to run and evaluate ZaCQ.

* gui/ - Backend and Frontend servers for a search interface implementing ZaCQ. 