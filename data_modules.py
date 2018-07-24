#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__synopsis__    : Tools for data manipulation
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

import os, random, re
from collections import OrderedDict
import my_modules as mm


def tag_dict(data, tag_text):
    for id, val in data.items():
        data[id]['tagged_text'] = tag_text
    return data


def merge_dicts(*dict_args):
    """ Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts. """
    # print("Method: merge_dicts(*dict_args)")
    result = OrderedDict()
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def split_data(lab_tweets, test_size, random_state=0):
    """ splits the data based on test_size"""
    print("Method: split_data(lab_tweets,test_size)")
    from sklearn.model_selection import train_test_split
    all_list = list(lab_tweets.keys())
    train_split, test_split = train_test_split(all_list, test_size=test_size,
                                               random_state=random_state)
    train = OrderedDict()
    test = OrderedDict()
    for id in train_split:
        train[id] = lab_tweets[id]
    for id in test_split:
        test[id] = lab_tweets[id]
    return train, test


def randomize_dict(dict_rand):
    dict_keys = list(dict_rand.keys())
    random.shuffle(dict_keys)
    rand_dict = OrderedDict()
    for key in dict_keys:
        rand_dict[key] = dict_rand[key]
    return rand_dict


def tokenize(s, lowercase=False, remove_emoticons=True):
    # print("Method: tokenize(s,lowercase=False,remove_emoticons=True)")
    import re
    emoticons_str = r'''
        (?:
            [:=;] # Eyes
            [oO\-]? # Nose (optional)
            [D\)\]\(\]/\\OpP] # Mouth
        )'''
    regex_str = [
        emoticons_str,
        r'<[^>]+>',  # HTML tags
        r'(?:@[\w_]+)',  # @-mentions
        r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f]['
        r'0-9a-f]))+',  # URLs
        r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
        r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
        r'(?:[\w_]+)',  # other words
        r'(?:\S)'  # anything else
    ]
    tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')',
                           re.VERBOSE | re.IGNORECASE)
    emoticon_re = re.compile(r'^' + emoticons_str + '$',
                             re.VERBOSE | re.IGNORECASE)

    # TODO: remove emoticons only (param: remove_emoticons).
    tokens = tokens_re.findall(str(s))
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for
                  token in tokens]
    return tokens


def get_acronyms(terms):
    acro_path = mm.get_dataset_path()
    acronym_dict = mm.read_json(os.path.join(acro_path, "acronym"))
    for i, term in enumerate(terms):
        if not term.isupper():
            terms[i] = get_acronym(acronym_dict, term)
    return terms


def get_acronym(acronym_dict, term):
    """Check in acronym.json file and returns the acronym of the term"""
    # print("Method: get_acronyms(term)",term)
    if term in acronym_dict.keys():
        # print(term," -> ",acronym_dict[term])
        return acronym_dict[term]
    else:
        return term


def count_class(class_list, n_classes):
    """count new tweets class poportions"""
    # print("Method: count_class(list)")
    class_count = [0] * n_classes
    for i in range(len(class_list)):
        for cls in class_list[i]:
            class_count[cls] = class_count[cls] + 1
    return class_count


def arrarr_bin(arrarr, n_classes, threshold=False):
    print("Method: arrarr_bin(arrarr, n_classes, threshold=False)")
    """Converts array of array to np.matrix of bools"""
    import numpy as np
    # b = [[False]*len(arrarr[0]) for i in range(len(arrarr))]
    c = [False] * n_classes
    b = [c for i in range(len(arrarr))]
    for i, arr in enumerate(arrarr):
        b[i] = arr_bin(arr, n_classes, threshold)

    return np.matrix(b)


def arr_bin(arr, n_classes, threshold=False):
    # print("Method: arr_bin(arr, n_classes, threshold=False)")
    """Converts a single array of numbers to array of bools"""
    votes_bin = [False] * n_classes
    for i, a in enumerate(arr):
        if threshold:
            if a >= threshold:
                votes_bin[i] = True
        else:
            votes_bin[a] = True

    return votes_bin


def remove_dup_list(seq, case=False):  # Dave Kirby
    """Removes duplicates from a list. Order preserving"""
    seen = set()
    if case: return [x.lower() for x in seq if
                     x.lower() not in seen and not seen.add(x)]
    return [x for x in seq if x not in seen and not seen.add(x)]


def word_cloud(corpus):
    from os import path
    from scipy.misc import imread
    import matplotlib.pyplot as plt
    import random

    from wordcloud import WordCloud, STOPWORDS

    text = 'all your base are belong to us all of your base base base'
    wordcloud = WordCloud(font_path='/Library/Fonts/Verdana.ttf',
                          relative_scaling=1.0,
                          stopwords={'to', 'of'}
                          # set or space-separated string
                          ).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def stemming(word, lemmatize=False):
    if lemmatize:
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        return wnl.lemmatize(word)
    else:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        return ps.stem(word)


def digit_count_str(s):
    return len(str(abs(int(s))))


def is_float(w):
    """takes str and returns if it is a decimal number"""
    try:
        num = float(w)
        return True, num
    except ValueError as e:
        return False, w


def find_numbers(text, replace=False):
    """:param text: strings that contains digit and words
    :param replace: bool to decide if numbers need to be replaced.
    :return: text, list of numbers
    Ex:
    '1230,1485': 8d
    '-2': 1d
    3.0 : 2f
    """
    import re
    numexp = re.compile(r'(?:(?:\d+,?)+(?:\.?\d+)?)')
    numbers = numexp.findall(" ".join(text))
    # print(numbers)
    if replace:
        for num in numbers:
            try:
                i = text.index(num)
                if num.isdigit():
                    text[i] = str(len(num)) + "d"
                else:
                    try:
                        num = float(num)
                        text[i] = str(len(str(num)) - 1) + "f"
                    except ValueError as e:
                        if len(num) > 9:
                            text[i] = "phonenumber"
                        else:
                            text[i] = str(len(num) - 1) + "d"
            except ValueError as e:
                pass
                # print("Could not find number [",num,"] in tweet: [",text,"]")
    return text, numbers


def remove_symbols(tweet, stopword=False, punct=False, specials=False):
    # print("Method: remove_symbols(tweet,stopword=False,punct=False,
    # specials=False)")
    if stopword:
        from nltk.corpus import stopwords
        stopword_list = stopwords.words('english') + ['rt', 'via', '& amp',
                                                      '&amp', 'amp', 'mr']
        tweet = [term for term in tweet if term not in stopword_list]
    # print("stopword: ", tweet)

    if punct:
        from string import punctuation
        tweet = [term for term in tweet if term not in list(punctuation)]
    # print("punct: ", tweet)

    if specials:
        trans_dict = {chars: ' ' for chars in special_chars}
        trans_table = str.maketrans(trans_dict)
        tweet = tweet.translate(trans_table)

        for pos in range(len(tweet)):
            tweet[pos] = tweet[pos].replace("@", "")
            tweet[pos] = tweet[pos].replace("#", "")
            tweet[pos] = tweet[pos].replace("-", " ")
            tweet[pos] = tweet[pos].replace("&", " and ")
            tweet[pos] = tweet[pos].replace("$", " dollar ")
            tweet[pos] = tweet[pos].replace("  ", " ")
    # print("specials: ", tweet)
    return tweet


def case_folding(tweet, all_caps=False):
    # tweet = tweet.split()
    for pos in range(len(tweet)):
        if tweet[pos].isupper():
            continue
        else:
            tweet[pos] = tweet[pos].lower()
    return tweet


def parse_tweets(train, remove_url=True, token=True, lowercase=False,
                 r_symbols=True,\
                 stopword=True, punct=True, specials=True, replace=True,
                 num=True, acro=True,\
                 process=True, case=True, all_caps=True):
    print("Method: parse_tweets(train)")

    for id, val in train.items():
        # TODO: expand URL
        twt_list, _ = parse_tweet(val['text'], remove_url, token,\
                                  lowercase, r_symbols, stopword, punct,
                                  specials, replace, num, acro, False,\
                                  case, all_caps)
        # print(twt_list)
        val['parsed_tweet'] = " ".join(twt_list)
    train = mm.process_dict_spacy(train)
    return train


def parse_tweet(tweet, remove_url=True, token=True, lowercase=False,
                r_symbols=True,\
                stopword=True, punct=True, specials=True, replace=True,
                num=True, acro=True,\
                process=True, case=True, all_caps=True):
    # print("Method: parse_tweet(tweet,token=True,lowercase=False,
    # r_symbols=True,\

    # stopword=True,punct=True,specials=True,replace=True,num=True,acro=True,\
    # process=True,case=True,all_caps=True")
    processing_done = False
    try:
        if remove_url:
            # TODO: expand url instead of removing
            tweet = re.sub(r"http\S+", "urlurl", tweet)
            # print("remove_url: ", tweet)
        if token:
            tweet = tokenize(tweet, lowercase, remove_emoticons=True)
            # print("tokenize: ", tweet)
        if r_symbols:
            tweet = remove_symbols(tweet, stopword, punct, specials)
            # print("remove_symbols: ", tweet)
        if num:
            tweet, _ = find_numbers(tweet, replace=True)
            # print("find_numbers: ", tweet)
        if acro:
            tweet = get_acronyms(tweet)
            # print("get_acronyms: ", tweet)
        if process:
            result = mm.process_spacy(" ".join(tweet), entity=True)
            processing_done = True
            # print("process_spacy: ", tweet)
        if case:
            tweet = case_folding(tweet, all_caps=all_caps)
            # print("case_folding: ", tweet)
    except Exception as e:
        print("Parsing tweet FAIL: ", tweet)
        print(e)

    if processing_done:
        return tweet, result
    else:
        return tweet, None


def make_per_class(train, n_classes):
    class_data = OrderedDict()
    for cls in range(n_classes):
        class_data[cls] = []
    print(class_data)
    for i, twt in train.items():
        for cls in twt["classes"]:
            # class_data[cls] = []
            twt["id"] = i
            class_data[cls].append(twt)
    return class_data


def dict_csv(dict_data, csv_file):
    csv_columns = ["id", "text", "parsed_tweet", "retweet_count", "urls", "",
                   "", "", "", ""]
    # for d,v in dict_data:

    print(csv_columns)

    try:
        with open(csv_file + '.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError as e:
        print("I/O error({0}):".format(e))
    return


def create_corpus(data, n_classes):
    print("Method: create_corpus(data,n_classes)")
    total_corpus = []
    class_corpuses = dict((key, []) for key in range(n_classes))
    for id, vals in data.items():
        total_corpus.append(vals["parsed_tweet"])
        for cls in vals["classes"]:
            class_corpuses[cls].append(vals["parsed_tweet"])
    return total_corpus, class_corpuses


def unique_tokens(twt_list):
    cls_freq_dict = OrderedDict()
    for twt in twt_list:
        for t in tokenize(twt):
            if t in cls_freq_dict.keys():
                cls_freq_dict[t] = cls_freq_dict[t] + 1
            else:
                cls_freq_dict[t] = 1
    # print(tokens)
    return cls_freq_dict


def doc_freq(cls_corpuses):
    twt_list_all_cls = []
    for cls, twt_list in cls_corpuses.items():
        twt_list_all_cls.append(twt_list)
    doc_freq_dict = unique_tokens(twt_list_all_cls)
    return doc_freq_dict


def cal_entropy(feature_list,cls_tokens,doc_freq_dict):
    print("Method: cal_entropy(feature_list,cls_tokens,doc_freq_dict)")
    import math
    ent = OrderedDict()
    for feature in feature_list:
        e = 0
        for cls in cls_tokens.keys():
            if feature in cls_tokens[cls].keys():
                tfj = cls_tokens[cls][feature] / doc_freq_dict[feature]
            else:
                tfj = 1e-10
            log_tfj = math.log(tfj,2)
            e = e + (- (tfj * log_tfj))
        ent[feature] = e

    return ent


def update_matrix_CNE(matrix, cls_freq_dict, feature_list, doc_freq_dict,
                      ent, ent_max,alpha=0.5,beta=0.5,test="alpha"):
    print("Method: update_matrix_CNE(matrix,cls_freq_dict,feature_list,doc_freq_dict,ent,ent_max)")
    max_freq = max(cls_freq_dict.values())
    for i, feature in enumerate(feature_list):
        if feature in cls_freq_dict.keys():
            for d in range(matrix.shape[0]):
                if matrix[d, i] > 0.0:
                    if matrix[d, i] <= 1.0:
                        if d % 5000 == 0:
                            print("cf:",cls_freq_dict[feature],"df:",
                                  doc_freq_dict[feature], "term:",feature)
                            # print("test_name",test)
                            # print("class tokens:",len(cls_freq_dict),"total tokens:",len(doc_freq_dict))
                            print("matrix_TFIDF["+str(d)+","+str(i)+"]:",matrix[d,i])

                        NE = (ent_max - ent[feature]) / ent_max
                        if test == "eccd":
                            p_tj_ck = (cls_freq_dict[feature]/doc_freq_dict[feature])
                            p_tj_ck_ = (doc_freq_dict[feature] - cls_freq_dict[feature])/doc_freq_dict[feature]
                            prob_part = p_tj_ck - p_tj_ck_
                            change = prob_part * NE
                            matrix[d,i] *= change
                            if d % 5000 == 0:
                                print("p_tj_ck:", p_tj_ck)
                                print("p_tj_ck_:", p_tj_ck_)
                                print("NE:", NE)
                                print("ECCD change:", change)
                                print("ECCD:", matrix[d,i])
                                print()
                        else:
                            cf = cls_freq_dict[feature]/len(cls_freq_dict)
                            df = doc_freq_dict[feature]/len(doc_freq_dict)
                            IW = cf * df
                            if test == "alphabeta":
                                if d % 5000 == 0:
                                    print("alpha",alpha,"beta",beta)
                                change = (alpha * IW) * (beta * NE)
                            if test == "alpha":
                                if d % 5000 == 0:
                                    print("alpha",alpha)
                                change = (alpha * IW) * ((1 - alpha) * NE)
                            if test == "k":
                                if d % 5000 == 0:
                                    print("k",alpha)
                                change = (IW * NE) / alpha
                            # if test == "add":
                                # if d % 5000 == 0:
                                    # print("k",alpha)
                                # change = (IW + NE) / alpha
                            # if test == "iw":
                                # if d % 5000 == 0:
                                    # print("k",alpha)
                                # change = IW / alpha
                            matrix[d, i] += change

                            if d % 5000 == 0:
                                print("CNE:\t\t     ", matrix[d,i])
                                # print("cf / class tokens:", cf)
                                # print("df / total tokens:", df)
                                print("IW:", IW)
                                print("NE:", NE)
                                print("change:", change)
                                print()
    return matrix


def class_tfidf_CNE(train, vec, train_tfidf_matrix_1, n_classes,alpha=0.5,beta=0.5,test="alpha"):
    print("Method: class_tfidf_CNE(train,vec,train_tfidf_matrix_1,n_classes)")
    import copy
    feature_list = vec.get_feature_names()
    corpus,cls_corpuses = mm.create_corpus(train, n_classes)
    cls_tokens = OrderedDict()
    doc_freq_dict = mm.doc_freq(cls_corpuses)
    matrices = OrderedDict()

    for cls, twt_list in cls_corpuses.items():
        matrices[cls] = copy.deepcopy(train_tfidf_matrix_1)
        cls_tokens[cls] = mm.unique_tokens(twt_list)

    # print("cls_tokens")
    # print(cls_tokens)
    ent = mm.cal_entropy(feature_list,cls_tokens,doc_freq_dict)
    ent_max = max(ent.values())
    print("ent_max",ent_max)
    if test == "alphabeta":
        print("alpha",alpha,"beta",beta)
    if test == "alpha":
        print("alpha",alpha)
    if test == "mul":
        print("mul k",alpha)
    if test == "add":
        print("add k",alpha)
    if test == "iw":
        print("iw k",alpha)
    for cls,twt_list in cls_corpuses.items():
        matrices[cls] = update_matrix_CNE(matrices[cls],cls_tokens[cls],
                                          feature_list,doc_freq_dict,ent,
                                          ent_max,alpha,beta,test)
    return matrices


def main():
    tweet = '#NepalEarthquake India plz 1230,1485 #NDRF team, -2 dogs and 3.2 '\
            'tonnes equipment to Nepal-Army for rescue operations: Indian '\
            'Embassy'
    print(tweet.split())
    print(" ".join(parse_tweet(tweet, acro=False, process=False)))


if __name__ == "__main__": main()
