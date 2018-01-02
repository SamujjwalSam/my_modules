import os, json
import unicodedata
from collections import OrderedDict
import my_modules as mm


date_time_tag = mm.get_date_time_tag()


def get_dataset_path():
    import platform
    if platform.system() == 'Windows':
        dataset_path = 'D:\Datasets'
    else:
        dataset_path = '/home/cs16resch01001/datasets'
    print(platform.system(), "os detected.")
    return dataset_path


dataset_path = get_dataset_path()


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
            unlabeled_tweets_dict["parsed_text"] = mm.parse_tweet(tweet_text)
            unlabeled_tweets_dict[line["id"]]['classes'] = []
    f.close()
    return unlabeled_tweets_dict


def read_labeled(labeled_file):
    print("Reading labeled Data from file: ",labeled_file)
    if os.path.isfile(labeled_file + "parsed_train" + ".json") and\
            os.path.isfile(labeled_file + "parsed_validation" + ".json")\
            and os.path.isfile(labeled_file + "parsed_test" + ".json"):
        train = read_json(labeled_file + "parsed_train")
        validation = read_json(labeled_file + "parsed_validation")
        test = read_json(labeled_file + "parsed_test")
    elif os.path.isfile(labeled_file + "train" + ".json") and\
            os.path.isfile(labeled_file + "validation" + ".json") and\
            os.path.isfile(labeled_file + "test" + ".json"):
        train = read_json(labeled_file + "train")
        validation = read_json(labeled_file + "validation")
        test = read_json(labeled_file + "test")

        train = mm.parse_tweets(train)
        validation = mm.parse_tweets(validation)
        test = mm.parse_tweets(test)

        save_json(train, labeled_file + "parsed_train")
        save_json(validation, labeled_file + "parsed_validation")
        save_json(test, labeled_file + "parsed_test")
    elif os.path.isfile(labeled_file + ".json"):
        labeled_dict = read_json(labeled_file)
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
        json_array[line["id"]] = line
        json_array[line["id"]]['parsed_tweet'] = mm.parse_tweet(tweet_text)
        json_array[line["id"]]['classes'] = []
    return json_array


def write_file(data, file_name, mode='w', tag=False):
    if tag:
        with open(date_time_tag + file_name + ".txt", mode,
                  encoding="utf-8") as out_file:
            out_file.write(str(data))
            out_file.write("\n")
            out_file.write("\n")
        out_file.close()
    else:
        with open(file_name + ".txt", mode, encoding="utf-8") as out_file:
            out_file.write(str(data))
            out_file.write("\n")
            out_file.write("\n")
        out_file.close()


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


def read_file_folder(folder, mode='r'):
    """Reads all files in a folder"""
    files_dict = OrderedDict()
    import glob
    files = glob.glob(folder + '/*.txt')
    for file_name in files:
        files_dict[file_name] = read_file(os.path.join(folder, file_name),
                                          mode=mode)
    return files_dict


def save_json(data, filename, tag=False):
    try:
        if tag:
            with open(date_time_tag + filename + ".json", 'w') as outfile:
                outfile.write(json.dumps(data, indent=4))
            outfile.close()
            return True
        else:
            with open(filename + ".json", 'w') as outfile:
                outfile.write(json.dumps(data, indent=4))
            outfile.close()
            return True
    except Exception as e:
        print("Could not write to file: ", filename)
        print("Failure reason: ", e)
        print("Writing file as plain text: ", filename + ".txt")
        if tag:
            write_file(data,filename,tag=True)
            return False
        else:
            write_file(data,filename)
            return False


def read_json(filename, alternate=None):
    print("Reading JSON file: ", filename + ".json")
    if os.path.isfile(filename + ".json"):
        with open(filename + ".json", encoding="utf-8") as file:
            json_dict = OrderedDict(json.load(file))
        file.close()
        return json_dict
    elif alternate:
        print("Warning:", filename + " does not exist, reading ", alternate)
        alternate = read_json(alternate)
        return alternate
    else:
        print("Warning: Could not open file: " + filename)
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


def main():
    dict1 = {1:{}, 2:{}}
    dict1 = OrderedDict(dict1)
    dict1 = mm.tag_dict(dict1, 'h')
    print(dict1)
    pass


if __name__ == "__main__": main()
