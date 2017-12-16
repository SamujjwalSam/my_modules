import random, string, re
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


from nltk.corpus import stopwords


stopword_list = stopwords.words('english') + list(string.punctuation)\
                + ['rt', 'via', '& amp', '&amp','mr']


def parse_tweets(train):
    print("Method: parse_tweets(train)")
    for id, val in train.items():
        val['parsed_tweet'] = parse_tweet(val['text'])
    return train


def remove_stopwords(terms):
    return " ".join([term for term in terms if term not in stopword_list])


def parse_tweet(tweet):
    # print("Method: parse_tweet(tweet)")
    tweet = re.sub(r"http\S+", "urlurl", tweet) # hyperlink to urlurl
    terms = mm.preprocess(tweet, True)
    for term_pos in range(len(terms)):
        terms[term_pos] = terms[term_pos].replace("@", "")
        terms[term_pos] = terms[term_pos].replace("#", "")
        terms[term_pos] = get_acronyms(terms[term_pos])
        terms[term_pos] = mm.contains_phone(terms[term_pos])
        # TODO: pre-process the acronym
    mod_tweet = " ".join([term for term in terms if term not in stopword_list])
    return mod_tweet


acronym_dict = mm.read_json("acronym")


def get_acronyms(term):
    """Check for Acronyms and returns the acronym of the term"""
    # print("Method: get_acronyms(term)",term)
    # acronym_dict = mm.read_json("acronym")
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


def main():
    pass


if __name__ == "__main__": main()
