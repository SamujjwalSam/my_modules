#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__synopsis__    : Tools for NLP related operations using various libraries (Spacy, NLTK, etc)
__description__ :
__project__     : my_modules
__author__      : 'Samujjwal Ghosh'
__version__     :
__date__        : June 2018
__copyright__   : "Copyright (c) 2018"
__license__     : "Python"; (Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html)

__classes__     :

__variables__   :

__methods__     :

TODO            : 1.
"""

import sys,platform
if platform.system() == 'Windows':
    sys.path.append('D:\GDrive\Dropbox\IITH\\0 Research')
    sys.path.append('D:\GDrive\Dropbox\IITH\\0 Research\Datasets')
else:
    sys.path.append('/home/cs16resch01001/codes')
    sys.path.append('/home/cs16resch01001/datasets')


import my_modules as mm

entity_types = ['PERSON','NORP','FACILITY','ORG','GPE','LOC','PRODUCT','EVENT',
                'WORK_OF_ART','LAW','LANGUAGE','DATE','TIME','PERCENT','MONEY',
                'QUANTITY','ORDINAL','CARDINAL','PER','MISC']

def process_dict_spacy(train):
    print("Method: process_dict_spacy(train)")
    import spacy
    from collections import OrderedDict
    try:
        nlp = spacy.load('en')
    except Exception as e:
        nlp = spacy.load('en_core_web_sm')
    
    # from textblob import TextBlob

    for id, val in train.items():
        # print(id)
        twt = val['parsed_tweet']
        # twt = TextBlob(twt)
        # twt = twt.correct()
        # doc = nlp(str(twt))
        doc = nlp(twt)
        pos = []
        lemmas = []
        for token in doc:
            pos.append(token.pos_)
            lemmas.append(token.lemma_)
        val['lemma'] = " ".join(lemmas)
        val['pos'] = pos

    return train


def process_spacy(s,entity=False):
    import spacy
    from collections import OrderedDict
    try:
        nlp = spacy.load('en')
    except Exception as e:
        nlp = spacy.load('en_core_web_sm')

    doc = nlp(s)
    print(doc)

    result = OrderedDict()
    tokens = []
    pos = []
    lemmas = []
    for token in doc:
        #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        # token.shape_, token.is_alpha, token.is_stop)
        tokens.append(token.text)
        pos.append(token.pos_)
        lemmas.append(token.lemma_)
    result["tokens"] = tokens
    result["pos"] = pos
    result["lemmas"] = lemmas

    if entity:
        labels=[]
        ents_list = []
        for ent in doc.ents:
            ents_list.append(str(ent))

        for i,t in enumerate(list(tokens)):
            if t in "".join(ents_list):
                labels.append(True)
            else:
                labels.append(False)
        result["labels"] = labels
        result["ents_list"] = ents_list
        result["ents"] = doc.ents

    return result


def most_similar_spacy(word, k=10):
    from spacy.en import English
    parser = English()

    from numpy import dot
    from numpy.linalg import norm

    # you can access known words from the parser's vocabulary
    word_vocab = parser.vocab[word]

    # cosine similarity
    cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))

    # gather all known words, take only the lowercased versions
    allWords = list({w for w in parser.vocab if w.has_vector and w.orth_.islower() and w.lower_ != word_vocab})

    # sort by similarity to word
    allWords.sort(key=lambda w: cosine(w.vector, word_vocab.vector))
    allWords.reverse()
    #print("Top 10 most similar words: ",w)
    sim_list = []
    for word_vocab in allWords[:k]:
        #print(word,word.orth_)
        sim_list.append(word_vocab.orth_)
    return sim_list


def pos_nltk(str):
    import nltk
    text = nltk.word_tokenize(str)
    return nltk.pos_tag(text)


def spelling_correction(tweet):
    from textblob import TextBlob
    b = TextBlob(tweet)
    return b.correct()


def main():
    s = '#NepalEarthquake India plz 1230,1485 #NDRF team, -2 dogs and 3.2 tonnes equipment to Nepal-Army for rescue operations: Indian Embassy'
    # print(find_phone(s,replace=""))
    # print(spelling_correction(s))

    print(process_spacy(s))

    print(most_similar_spacy('short'))


if __name__ == "__main__": main()
