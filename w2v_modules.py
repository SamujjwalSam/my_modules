import os
import numpy as np
from collections import OrderedDict
import my_modules as mm


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
            if word in w2v.wv.vocab:
                # train_vec[id][word] = w2v[word].tolist()
                s_vec = np.add(s_vec, w2v[word])
            else:
                pass
                # print("Word [",word,"] not in vocabulary")
            # print("\n")
        train_vec[id]=s_vec
    return train_vec


def find_sim(w2v,word,c=None):

    print(w2v)
    print(type(w2v))
    #print("Find similar words of: ",word)
    w2v_words = []
    if word in w2v.wv.vocab:
        print(w2v.most_similar(positive=[word], negative=[], topn=c))
        w2v_words = w2v.most_similar(positive=[word], negative=[], topn=c)
        print("here2: ",w2v_words)
        #for term,val in list(w2v_words):
         #   word_list = word_list + [term]
    print(w2v_words)
    return w2v_words


def find_sim_list(w2v,words,c=None):

    for word in words:
        words = words + find_sim(w2v,word,c)

    words = mm.remove_dup_list(words, case=True)
    return words[0:c]


def expand_tweet(w2v,tweet,c=3):
    new_tweet = []
    for word in tweet.split(" "):
        new_tweet= new_tweet+[word]
        w2v_words = find_sim(w2v,word,c)
        #if word in w2v.vocab:
         #   w2v_words=w2v.most_similar(positive=[word], negative=[], topn=c)
        for term,val in w2v_words:
            new_tweet= new_tweet+[term]
    return new_tweet


def expand_tweets(w2v,dict):
    # print("Method: expand_tweets(dict)")
    for id,val in dict.items():
        val['expanded_tweet'] = "".join(expand_tweet(w2v,val['parsed_tweet']))
    return dict


def create_w2v(corpus,size=1000,window=5,min_count=3,workers=10):
    from gensim.models import Word2Vec
    w2v = Word2Vec(corpus,size,window,min_count,workers)
    print(w2v)
    print(type(w2v))
    return w2v


def main():
    pass


if __name__ == "__main__": main()