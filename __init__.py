#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__synopsis__    : init file for package [my_modules]
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

from .time_modules import get_date_time_tag

from .file_modules import get_dataset_path,read_unlabeled_json,read_labeled,\
    read_smerp_labeled,read_json_array,write_file,read_file,read_files_folder,\
    save_json,load_json,train_test_read_split,save_pickle,load_pickle,\
    read_json_array_nochange,read_nips_papers,read_Semantic_Scholar,read_xlsx

from .data_modules import tag_dict,merge_dicts,split_data,randomize_dict,\
    parse_tweets,parse_tweet,remove_symbols,get_acronyms,count_class,\
    arrarr_bin,arr_bin,remove_dup_list,stemming,find_numbers,digit_count_str,\
    tokenize,is_float,dict_csv,make_per_class,create_corpus,doc_freq,unique_tokens,\
    cal_entropy,update_matrix_CNE,class_tfidf_CNE


from .ml_modules import select_tweets,max_add_unlabeled,supervised,grid_search,\
    grid_search_rand,add_features_matrix,k_similar_tweets,get_cosine,\
    sim_tweet_class_vote,tf,n_containing,idf,tfidf,nltk_install,\
    unique_words_class,vectorizer,create_tf_idf,find_word,unique_word_count_class,\
    derived_features,manual_features,classifier_agreement,find_synms_list,\
    find_synms,supervised_bin

from .perform_modules import sklearn_metrics,accuracy_multi,get_feature_result

from .w2v_modules import init_w2v,open_word2vec,use_word2vec,find_sim,\
    find_sim_list,expand_tweet,expand_tweets,create_w2v

from .db_modules import connect_sqllite,read_sqllite,get_db_details

from .nlp_modules import process_spacy,most_similar_spacy,process_dict_spacy,\
    spelling_correction

from .graph_modules import csr2nx,nx2csr


import sys,platform
if platform.system() == 'Windows':
    sys.path.append('D:\GDrive\Dropbox\IITH\\0 Research')
    sys.path.append('D:\Datasets')
else:
    sys.path.append('/home/cs16resch01001/codes')
    sys.path.append('/home/cs16resch01001/datasets')
    sys.path.append('/home/Embeddings')
print(platform.system(),"os detected.")