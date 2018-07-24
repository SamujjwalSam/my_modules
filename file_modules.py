#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__synopsis__    : Tools for file load and save on various formats.
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

import os,sys,json,platform,pickle
import unicodedata
from collections import OrderedDict
import my_modules as mm
from scipy import sparse


date_time_tag = mm.get_date_time_tag()


def get_dataset_path():
    """

    :return:
    """
    if platform.system() == 'Windows':
        dataset_path = 'D:\Datasets\Extreme Classification'
        sys.path.append('D:\GDrive\Dropbox\IITH\\0 Research')
    elif platform.system() == 'Linux':
        dataset_path = '/home/cs16resch01001/datasets/Extreme Classification'
        sys.path.append('/home/cs16resch01001/codes')
    else:  # OS X returns name 'Darwin'
        dataset_path = '/Users/monojitdey/Downloads'
    print(platform.system(),"os detected.")

    return dataset_path
dataset_path = get_dataset_path()


def specials_table(specials="""< >  * ? " / \ : |""",replace=' '):
    """

    :param specials:
    :param replace:
    :return:

    usage: file_name = file_name.translate(trans_table)
    """
    trans_dict = {chars:replace for chars in specials}
    trans_table = str.maketrans(trans_dict)

    return trans_table


def read_unlabeled_json(file_name):
    print("Method: read_unlabeled_json(file_name)")
    unlabeled_tweets_dict = OrderedDict()
    with open(file_name + ".json", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            try:
                tweet_text = line["retweeted_status"]["text"]
            except KeyError:
                tweet_text = line["text"]
            tweet_text = unicodedata.normalize('NFKD', tweet_text).encode(
                'ascii', 'ignore').decode("utf-8")
            unlabeled_tweets_dict[line["id"]] = line
            # TODO: read ids with "id" key. No need to change structure at all.
            # TODO: Only add parsed_text, classes and features
            unlabeled_tweets_dict["parsed_text"] = mm.parse_tweet(tweet_text)
            unlabeled_tweets_dict[line["id"]]['classes'] = []
    f.close()
    return unlabeled_tweets_dict


def read_unlabeled_json_nochange(file_name):
    print("Method: read_unlabeled_json(file_name)")
    unlabeled_tweets_dict = OrderedDict()
    with open(file_name + ".json", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            try:
                tweet_text = line["retweeted_status"]["text"]
            except KeyError:
                tweet_text = line["text"]
            tweet_text = unicodedata.normalize('NFKD', tweet_text).encode(
                                               'ascii', 'ignore').decode("utf-8")
            unlabeled_tweets_dict["parsed_text"] = mm.parse_tweet(tweet_text)
            unlabeled_tweets_dict[line["id"]]['classes'] = []
    f.close()
    return unlabeled_tweets_dict


def read_labeled(labeled_file,n_classes=7,k_similar=15,k_unique_words=25):
    # print("Reading labeled Data from file: ",labeled_file)
    if os.path.exists(labeled_file+"features_train.json") and os.path.exists(labeled_file+"features_validation.json")and os.path.exists(labeled_file+"features_test.json"):
        print("Reading labeled Data from file: ",labeled_file+"features_validation"+".json")
        train=mm.read_json(labeled_file+"features_train")
        validation=mm.read_json(labeled_file+"features_validation")
        test=mm.read_json(labeled_file+"features_test")
    elif os.path.exists(labeled_file + "parsed_train" + ".json") and\
        os.path.exists(labeled_file + "parsed_validation" + ".json") and\
        os.path.exists(labeled_file + "parsed_test" + ".json"):
        print("Reading labeled Data from file: ",labeled_file+"parsed_validation"+".json")
        train = load_json(labeled_file + "parsed_train")
        validation = load_json(labeled_file + "parsed_validation")
        test = load_json(labeled_file + "parsed_test")

        save_json(train, labeled_file + "parsed_train")
        save_json(validation, labeled_file + "parsed_validation")
        save_json(test, labeled_file + "parsed_test")

        mm.derived_features(train,validation,test,n_classes,k_similar)

        total_corpus,class_corpuses = mm.create_corpus(train,n_classes)
        unique_words = mm.unique_words_class(class_corpuses,k_unique_words)

        mm.manual_features(train,unique_words,n_classes)
        mm.manual_features(validation,unique_words,n_classes)
        mm.manual_features(test,unique_words,n_classes)

        mm.save_json(train,labeled_file+"features_train")
        mm.save_json(validation, labeled_file + "features_validation")
        mm.save_json(test,labeled_file+"features_test")
    elif os.path.exists(labeled_file + "train" + ".json") and\
        os.path.exists(labeled_file + "validation" + ".json") and\
        os.path.exists(labeled_file + "test" + ".json"):
        print("Reading labeled Data from file: ",labeled_file+"validation"+".json")
        train = load_json(labeled_file + "train")
        validation = load_json(labeled_file + "validation")
        test = load_json(labeled_file + "test")

        train = mm.parse_tweets(train)
        validation = mm.parse_tweets(validation)
        test = mm.parse_tweets(test)

        save_json(train, labeled_file + "parsed_train")
        save_json(validation, labeled_file + "parsed_validation")
        save_json(test, labeled_file + "parsed_test")

        mm.derived_features(train,validation,test,n_classes,k_similar)

        total_corpus,class_corpuses = mm.create_corpus(train,n_classes)
        unique_words = mm.unique_words_class(class_corpuses,k_unique_words)

        mm.manual_features(train,unique_words,n_classes)
        mm.manual_features(validation,unique_words,n_classes)
        mm.manual_features(test,unique_words,n_classes)

        mm.save_json(train,labeled_file+"features_train")
        mm.save_json(validation, labeled_file + "features_validation")
        mm.save_json(test,labeled_file+"features_test")
    elif os.path.exists(labeled_file + ".json"):
        print("Reading labeled Data from file: ",labeled_file + ".json")
        labeled_dict = load_json(labeled_file)
        train, validation, test = train_test_read_split(labeled_dict)
        save_json(train, labeled_file + "train")
        save_json(validation, labeled_file + "validation")
        save_json(test, labeled_file + "test")

        train = mm.parse_tweets(train)
        validation = mm.parse_tweets(validation)
        test = mm.parse_tweets(test)

        save_json(train, labeled_file + "parsed_train")
        save_json(validation, labeled_file + "parsed_validation")
        save_json(test, labeled_file + "parsed_test")

        mm.derived_features(train,validation,test,n_classes,k_similar)

        total_corpus,class_corpuses = mm.create_corpus(train,n_classes)
        unique_words = mm.unique_words_class(class_corpuses,k_unique_words)

        mm.manual_features(train,unique_words,n_classes)
        mm.manual_features(validation,unique_words,n_classes)
        mm.manual_features(test,unique_words,n_classes)

        mm.save_json(train,labeled_file+"features_train")
        mm.save_json(validation, labeled_file + "features_validation")
        mm.save_json(test,labeled_file+"features_test")
    else:
        smerp_labeled = mm.read_smerp_labeled()
        save_json(smerp_labeled, labeled_file)
        train, validation, test = train_test_read_split(labeled_file)
        print("Number of labeled tweets: ", len(smerp_labeled))

        save_json(train, labeled_file + "train")
        save_json(validation, labeled_file + "validation")
        save_json(test, labeled_file + "test")

        train = mm.parse_tweets(train)
        validation = mm.parse_tweets(validation)
        test = mm.parse_tweets(test)

        save_json(train, labeled_file + "parsed_train")
        save_json(validation, labeled_file + "parsed_validation")
        save_json(test, labeled_file + "parsed_test")

        mm.derived_features(train,validation,test,n_classes,k_similar)

        total_corpus,class_corpuses = mm.create_corpus(train,n_classes)
        unique_words = mm.unique_words_class(class_corpuses,k_unique_words)

        mm.manual_features(train,unique_words,n_classes)
        mm.manual_features(validation,unique_words,n_classes)
        mm.manual_features(test,unique_words,n_classes)

        mm.save_json(train,labeled_file+"features_train")
        mm.save_json(validation, labeled_file + "features_validation")
        mm.save_json(test,labeled_file+"features_test")

    return train, validation, test


def read_smerp_labeled():
    file_names = [0,1,2,3]
    lab = OrderedDict()
    for file in file_names:
        # print("Reading file: ","smerp"+str(file)+".json")
        single = read_json_array("smerp"+str(file))
        for i, val in single.items():
            if i in lab:
                lab[i]["classes"].append(file)
            else:
                lab[i]=val
                lab[i]["classes"]=[]
                lab[i]["classes"].append(file)
        # print("Finished file: ","smerp"+str(file)+".json")
        # lab = merge_dicts(lab,single)
    return lab


def read_json_array(json_array_file):
    print("Method: read_json_array(json_array_file)")
    json_array = OrderedDict()
    data = open(json_array_file + '.json')
    f = json.load(data)
    for line in f:
        line = json.loads(line)
        try:
            tweet_text = line["retweeted_status"]["text"]
        except KeyError:
            tweet_text = line["text"]
        tweet_text = unicodedata.normalize('NFKD', tweet_text).encode('ascii',
                                           'ignore').decode("utf-8")
        json_array[line["id_str"]] = line
        # TODO: read ids with "id" key. No need to change structure at all.
        # TODO: Only add parsed_text, classes and features
        json_array[line["id_str"]]['parsed_tweet'] = mm.parse_tweet(tweet_text)
        json_array[line["id_str"]]['classes'] = []
    return json_array


def read_json_array_nochange(json_array_file,label=False):
    print("Method: read_json_array(json_array_file)")
    json_array = OrderedDict()
    data = open(json_array_file + '.json')
    data = json.load(data)
    for line in data:
        # line = json.loads(line)
        # print(line)
        try:
            tweet_text = line["retweeted_status"]["text"]
        except KeyError:
            tweet_text = line["text"]
        tweet_text = unicodedata.normalize('NFKD', tweet_text).encode('ascii',
                                           'ignore').decode("utf-8")
        json_array[line["id_str"]] = line
        json_array[line["id_str"]]['parsed_tweet'] = mm.parse_tweet(tweet_text)
        if label:
            json_array[line["id_str"]]['classes'] = [label]
        else:
            json_array[line["id_str"]]['classes'] = []
    return json_array


def write_file(data,filename,file_path='',overwrite=False,mode='w',date_time_tag=''):
    """
    Writes to file as string
    :param data:
    :param filename:
    :param file_path:
    :param overwrite:
    :param mode:
    :param date_time_tag:
    :return:
    """
    if not overwrite and os.path.exists(os.path.join(file_path,date_time_tag+filename+".txt")):
        print("File already exists and Overwrite == False.")
        return True
    with open(os.path.join(file_path,date_time_tag+filename+".txt"),mode,encoding="utf-8") as text_file:
        print("Saving text file: ", os.path.join(file_path,date_time_tag+filename+".txt"))
        text_file.write(str(data))
        text_file.write("\n")
        text_file.write("\n")
    text_file.close()
    return True


def load_npz(filename,file_path=''):
    """
    Loads numpy objects from npz files.
    :param filename:
    :param file_path:
    :return:
    """
    print("Reading NPZ file: ",os.path.join(file_path,filename + ".npz"))
    if os.path.exists(os.path.join(file_path,filename + ".npz")):
        npz = sparse.load_npz(os.path.join(file_path,filename + ".npz"))
        return npz
    else:
        print("Warning: Could not open file: ",os.path.join(file_path,filename + ".npz"))
        return False


def save_npz(data,filename,file_path='',overwrite=True):
    """
    Saves numpy objects to file.
    :param data:
    :param filename:
    :param file_path:
    :param overwrite:
    :return:
    """
    print("Saving NPZ file: ",os.path.join(file_path,filename + ".npz"))
    if not overwrite and os.path.exists(os.path.join(file_path,filename + ".npz")):
        print("File already exists and Overwrite == False.")
        return True
    try:
        sparse.save_npz(os.path.join(file_path,filename + ".npz"),data)
        return True
    except Exception as e:
        print("Could not write to npz file:",os.path.join(file_path,filename + ".npz"))
        print("Failure reason:",e)
        return False


def read_file(file_name, mode='r', tag=False):
    # print("Reading file: ",file_name)
    if tag:
        with open(date_time_tag + file_name, mode,
                  encoding="utf-8") as in_file:
            data = str(in_file.read())
        in_file.close()
    else:
        with open(file_name, mode, encoding="utf-8") as in_file:
            data = str(in_file.read())
        in_file.close()
    return data


def read_files_folder(folder, mode='r',type='json'):
    """Reads all [type] files in a folder"""
    files_dict = OrderedDict()
    import glob
    files = glob.glob(folder + '/*.'+type)
    for file_name in files:
        files_dict[file_name] = read_file(os.path.join(folder, file_name),
                                          mode=mode)
    return files_dict


def save_json(data,filename,file_path='',overwrite=False,indent=2,date_time_tag=''):
    """

    :param data:
    :param filename:
    :param file_path:
    :param overwrite:
    :param indent:
    :param date_time_tag:
    :return:
    """
    import json
    print("Saving JSON file: ", os.path.join(file_path,date_time_tag+filename+".json"))
    if not overwrite and os.path.exists(os.path.join(file_path,date_time_tag+filename+".json")):
        print("File already exists and Overwrite == False.")
        return True
    try:
        with open(os.path.join(file_path,date_time_tag+filename+".json"),'w') as json_file:
            try:
                json_file.write(json.dumps(data, indent=indent))
            except Exception as e:
                print("Writing json as string:",os.path.join(file_path,date_time_tag+filename+".json"))
                json_file.write(json.dumps(str(data), indent=indent))
                return True
        json_file.close()
        return True
    except Exception as e:
        print("Could not write to json file:",os.path.join(file_path,filename))
        print("Failure reason:",e)
        print("Writing file as plain text:",filename+".txt")
        write_file(data,filename,date_time_tag=date_time_tag)
        return False


def load_json(filename,file_path='',date_time_tag=''):
    """
    Loads json file as python OrderedDict
    :param filename:
    :param file_path:
    :param date_time_tag:
    :return: OrderedDict
    """
    # print("Reading JSON file: ",os.path.join(file_path,date_time_tag+filename+".json"))
    if os.path.exists(os.path.join(file_path,date_time_tag+filename+".json")):
        with open(os.path.join(file_path,date_time_tag+filename+".json"), encoding="utf-8") as file:
            json_dict = OrderedDict(json.load(file))
        file.close()
        return json_dict
    else:
        print("Warning: Could not open file:",os.path.join(file_path,date_time_tag+filename+".json"))
        return False


def train_test_read_split(data, test_size=0.3, validation_size=0.3):
    """Splits json file into Train, Validation and Test"""
    print("Method: train_test_read_split(dict,test_size=0.3,validation_size=0.3",
          "validation=True)")
    train,test=mm.split_data(data,test_size)
    print("train size:",len(train))
    print("test size:",len(test))
    train,validation=mm.split_data(train,validation_size)
    print("validation size:",len(validation))
    return train,validation,test


def save_pickle(data,pkl_file_name,pkl_file_path,overwrite=False,tag=False):
    """
    saves python object as pickle file
    :param data:
    :param pkl_file_name:
    :param pkl_file_path:
    :param overwrite:
    :return:
    """
    # print("Method: save_pickle(data, pkl, tag=False)")
    print("Writing to pickle file: ",os.path.join(pkl_file_path,pkl_file_name + ".pkl"))
    if not overwrite and os.path.exists(os.path.join(pkl_file_path,pkl_file_name + ".pkl")):
        print("File already exists and Overwrite == False.")
        return True
    try:
        if tag:
            if os.path.exists(date_time_tag + os.path.join(pkl_file_path,pkl_file_name + ".pkl")):
                print("Overwriting on pickle file: ", date_time_tag + os.path.join(pkl_file_path,pkl_file_name + ".pkl"))
            with open(date_time_tag + os.path.join(pkl_file_path,pkl_file_name + ".pkl"), 'wb') as pkl_file:
                pickle.dump(data,pkl_file)
            pkl_file.close()
            return True
        else:
            if os.path.exists(os.path.join(pkl_file_path,pkl_file_name + ".pkl")):
                print("Overwriting on pickle file: ", os.path.join(pkl_file_path,pkl_file_name + ".pkl"))
            with open(os.path.join(pkl_file_path,pkl_file_name + ".pkl"), 'wb') as pkl_file:
                pickle.dump(data,pkl_file)
            pkl_file.close()
            return True
    except Exception as e:
        print("Could not write to pickle file: ", os.path.join(pkl_file_path,pkl_file_name + ".pkl"))
        print("Failure reason: ", e)
        return False


def load_pickle(pkl_file_name,pkl_file_path):
    """
    Loads pickle file from files.
    :param pkl_file_name:
    :param pkl_file_path:
    :return:
    """
    print("Method: load_pickle(pkl_file)")
    try:
        if os.path.exists(os.path.join(pkl_file_path,pkl_file_name + ".pkl")):
            print("Reading pickle file: ",os.path.join(pkl_file_path,pkl_file_name + ".pkl"))
            with open(os.path.join(pkl_file_path,pkl_file_name + ".pkl"),'rb') as pkl_file:
                loaded = pickle.load(pkl_file)
            return loaded
    except Exception as e:
        print("Could not write to pickle file:", os.path.join(pkl_file_path,pkl_file_name + ".pkl"))
        print("Failure reason:", e)
        return False


def read_results(result_file_name):
    if result_file_name.endswith('.json'):
        name,ext = os.path.splitext(result_file_name)
        dataset,param,value = name.split()


def read_nips_papers():
    dataset_name="nips-papers"
    db_name     ="nips-papers_db.sqlite"

    nips_db = mm.connect_sqllite(dataset_name,db_name)
    mm.get_db_details(nips_db)
    # read_sqllite(nips_db,table_name,cols="*",fetch_one=False)


def read_Semantic_Scholar():
    dataset_name="Semantic_Scholar"
    dataset_file="papers-2017-02-21_80.json"
    sem_scho = OrderedDict()
    with open(os.path.join(mm.get_dataset_path(),dataset_name,dataset_file),'r',
        encoding="utf-8") as ss_file:
        for line in ss_file:
            # line = line.decode('unicode_escape').encode('ascii','ignore') # to remove unicode characters
            line = json.loads(line)
            sem_scho[line['id']] = line['paperAbstract']
    # print(sem_scho)
    print(len(sem_scho))
    mm.save_json(sem_scho,'sem_scho')
    # write_file(sem_scho,'sem_scho', mode='w', tag=False)
    # save_pickle(sem_scho,'sem_scho', tag=False)


def read_xlsx(file,sheets=""):
    """Reads xlsx file as pandas dataframe for each sheet"""
    import pandas as pd
    data = pd.ExcelFile(file, sheet_name=sheets)
    xlsx_obj = OrderedDict()
    for sheet in data.sheet_names:
        xlsx_obj[sheet] = data.parse(sheet)
    print(data.sheet_names)
    return data, xlsx_obj


from scipy.io import arff
import pandas as pd
def load_arff(filename):
    print(filename)

    data = arff.loadarff(filename)
    print(type(data))
    df = pd.DataFrame(data[0])

    print(df.head())
    return data,df


def main():
    read_nips_papers()
    return
    dict1 = {1:{}, 2:{}}
    dict1 = OrderedDict(dict1)
    dict1 = mm.tag_dict(dict1, 'h')
    print(dict1)
    pass


if __name__ == "__main__": main()
