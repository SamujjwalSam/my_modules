import sys,platform
if platform.system() == 'Windows':
    sys.path.append('D:\GDrive\Dropbox\IITH\\0 Research')
    sys.path.append('D:\GDrive\Dropbox\IITH\\0 Research\Datasets')
else:
    sys.path.append('/home/cs16resch01001/codes')
    sys.path.append('/home/cs16resch01001/datasets')
print(platform.system(),"os detected.")
import my_modules as mm

from .time_modules import get_date_time_tag
from .file_modules import read_labeled,read_json_array,write_file,read_file,read_file_folder,save_json,read_json,get_dataset_path,read_unlabeled_json
from .data_modules import tag_dict,merge_dicts,split_data,randomize_dict,parse_tweets,parse_tweet,get_acronyms,count_class,remove_stopwords
from .ml_modules import k_similar_tweets,get_cosine,sim_tweet_class_vote,create_corpus,unique_words_class,vectorizer,create_tf_idf,manual_features,supervised,derived_features,max_add_unlabeled,select_tweets,add_features_matrix
from .perform_modules import sklearn_metrics,accuracy_multi
from .re_modules import contains_phone,preprocess
from .w2v_modules import init_w2v,open_word2vec,use_word2vec,expand_tweet,expand_tweets
from .db_modules import read_sqllite, connect_sqllite
