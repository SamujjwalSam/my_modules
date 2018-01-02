import random, re
from collections import OrderedDict
# import my_modules as mm


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


def parse_tweets(train):
    print("Method: parse_tweets(train)")
    for id, val in train.items():
        val['parsed_tweet'] = parse_tweet(val['text'])
    return train


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
    for i,term in enumerate(terms):
        if not term.isupper():
            terms[i] = get_acronym(term)
    return terms


def get_acronym(term):
    """Check in acronym.json file and returns the acronym of the term"""
    # print("Method: get_acronyms(term)",term)
    acronym_dict = mm.read_json("acronym")
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
    """Converts array of array to np.matrix of bools"""
    import numpy as np
    # b = [[False]*len(arrarr[0]) for i in range(len(arrarr))]
    c = [False] * n_classes
    b = [c for i in range(len(arrarr))]
    for i, arr in enumerate(arrarr):
        b[i] = arr_bin(arr, n_classes, threshold)

    return np.matrix(b)


def arr_bin(arr, n_classes, threshold=False):
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


def find_numbers(text,replace=False):
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
    if replace:
        for num in numbers:
            i = text.index(num)
            if num.isdigit():
                text[i] = str(len(num))+ "d"
            else:
                try:
                    num = float(num)
                    text[i] = str(len(str(num))-1)+ "f"
                except ValueError as e:
                    text[i] = str(len(num)-1)+ "d"
    return text,numbers


def find_phone(text,replace="phonenumber"):
    import re
    phone=re.compile(r'''
                # don't match beginning of string,number can start anywhere
    (\d{3})     # area code is 3 digits (e.g. '800')
    \D*         # optional separator is any number of non-digits
    (\d{3})     # trunk is 3 digits (e.g. '555')
    \D*         # optional separator
    (\d{4})     # rest of number is 4 digits (e.g. '1212')
    \D*         # optional separator
    (\d*)       # extension is optional and can be any number of digits
    $           # end of string
    ''',re.VERBOSE)

    if replace:
        text = re.sub(phone, replace, text)
    return text,len(phone.findall(text))


def remove_symbols(tweet,stopword=False,punct=False,specials=False):
    # print("Method: remove_symbols(tweet,stopword=False,punct=False,specials=False)")
    if stopword:
        from nltk.corpus import stopwords
        stopword_list = stopwords.words('english') + ['rt', 'via', '& amp', '&amp', 'amp', 'mr']
        tweet = [term for term in tweet if term not in stopword_list]
    # print("stopword: ", tweet)

    if punct:
        from string import punctuation
        tweet = [term for term in tweet if term not in list(punctuation)]
    # print("punct: ", tweet)

    if specials:
        for pos in range(len(tweet)):
            tweet[pos] = tweet[pos].replace("@", "")
            tweet[pos] = tweet[pos].replace("#", "")
            tweet[pos] = tweet[pos].replace("-", " ")
            tweet[pos] = tweet[pos].replace("&", " and ")
            tweet[pos] = tweet[pos].replace("$", " dollar ")
            tweet[pos] = tweet[pos].replace("  ", " ")
    # print("specials: ", tweet)
    return " ".join(tweet)


def case_folding(tweet, all_caps=False):
    # tweet = tweet.split()
    for pos in range(len(tweet)):
        if tweet[pos].isupper():
            continue
        else:
            tweet[pos] = tweet[pos].lower()
    return tweet


def parse_tweet(tweet,token=True,r_symbols=True,stopword=False,punct=False, \
    specials=False,phone=True,replace="phonenumber",num=True,acro=True, \
    process=True,case=True,all_caps=True):
    # print("Method: parse_tweet(tweet,token=True,r_symbols=True,stopword=False,punct=False,specials=False,phone=True,replace="phonenumber",num=True,acro=True,process=True,case=True,all_caps=True)")
    #tweet = expand_url(tweet)
    if token:
        tweet = tokenize(tweet, lowercase=False, remove_emoticons=True)
    # print("tokenize: ", tweet)
    if r_symbols:
        tweet = remove_symbols(tweet,True,True,True)
    # print("remove_symbols: ", tweet)
    if phone:
        tweet,_ = find_phone(tweet,replace=replace)
    # print("find_phone: ", tweet)
    if num:
        tweet,_ = find_numbers(tweet.split(),replace=True)
    # print("find_numbers: ", tweet)
    if acro:
        tweet = get_acronyms(tweet)
    # print("get_acronyms: ", tweet)
    if process:
        result= mm.process_spacy(tweet,entity=True)
    # print("process_spacy: ", tweet)
    if case:
        tweet = case_folding(tweet,all_caps=all_caps)
    # print("case_folding: ", tweet)

    if process:
        return tweet,result
    else:
        return tweet


def main():
    tweet = '#NepalEarthquake India plz 1230,1485 #NDRF team, -2 dogs and 3.2 tonnes equipment to Nepal-Army for rescue operations: Indian Embassy'
    print(tweet.split())
    print(parse_tweet(tweet,acro=False,process=False))


if __name__ == "__main__": main()
