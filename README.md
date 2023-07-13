# Experimental VELD executable

This repo extracts and converts all gold data from https://gitlab.oeaw.ac.at/acdh-ch/apis/spacy-ner
into conll2003, which is persisted here: 
https://gitlab.oeaw.ac.at/acdh-ch/nlp/veld_data_5_apis_ner_conll2003

The VELD chain which wires together the input, this code here, and the output is here: 
https://github.com/steffres/veld_chain_6__apis_ner_transform_to_conll2003

**notes on duplicated legacy code:**

Since some of the gold data is persisted as python pickle and contains references to classes from
the spacy-ner repo, these classes and their code context was copied from 
https://gitlab.oeaw.ac.at/acdh-ch/apis/spacy-ner/-/tree/8e75d3561e617f1bd135d4c06fbb982285f6f544/notebooks/ner 
into here: [./src/ner/](./src/ner/)

