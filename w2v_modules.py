import os
import numpy as np
from collections import OrderedDict


def init_w2v(nlp=True):
    if nlp:
        nlp_path = '/home/cs16resch01001/data/crisisNLP_word2vec_model/'
        nlp_file = 'crisisNLP_word_vector.bin'
        w2v = open_word2vec(os.path.join(nlp_path,nlp_file))
        print("NLP Word2Vec selected")
    else:
        g_path   = '/home/cs16resch01001/data/'
        g_file   = 'GoogleNews-vectors-negative300.bin'
        w2v = open_word2vec(os.path.join(g_path,g_file))
        print("Google Word2Vec selected")
    return w2v


def open_word2vec(word2vec):
    from gensim.models.keyedvectors import KeyedVectors
    model = KeyedVectors.load_word2vec_format(word2vec, binary=True)
    return model


def use_word2vec(train,w2v):
    train_vec = OrderedDict()
    for id,val in train.items():
        s_vec = np.zeros(300)
        for word in val['parsed_tweet'].split(" "):
            if word in w2v.vocab:
                # train_vec[id][word] = w2v[word].tolist()
                s_vec = np.add(s_vec, w2v[word])
            else:
                pass
                # print("Word [",word,"] not in vocabulary")
            # print("\n")
        train_vec[id]=s_vec
    return train_vec


def expand_tweet(w2v,tweet):
    new_tweet = []
    for word in tweet.split(" "):
        new_tweet= new_tweet+[word]
        if word in w2v.vocab:
            w2v_words=w2v.most_similar(positive=[word], negative=[], topn=3)
            for term,val in w2v_words:
                new_tweet= new_tweet+[term]
    return new_tweet


def expand_tweets(w2v,dict):
    # print("Method: expand_tweets(dict)")
    for id,val in dict.items():
        val['expanded_tweet'] = "".join(expand_tweet(w2v,val['parsed_tweet']))
    return dict
