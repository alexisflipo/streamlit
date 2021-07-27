import pickle
import pandas as pd 
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import gensim

from gensim.corpora import Dictionary
from gensim import corpora, models


'''
Write functions to load the models used to preprocess new sentences
'''

def load_lda_model(filepath):
    # load the model from disk
    loaded_model = pickle.load(open(filepath, 'rb'))
    return loaded_model


def tfidf_transformer(transformerpath):
    tfidf_trained = models.TfidfModel.load(transformerpath)
    return tfidf_trained

def dictionary_transformer(dictionarypath):
    dictionary_trained = Dictionary.load(dictionarypath)
    return dictionary_trained


'''
Write a function to perform the pre processing steps on the entire dataset
'''
def lemmatize_stemming(text):
    w = text
    w = WordNetLemmatizer().lemmatize(w, pos='v').lower()
    w = WordNetLemmatizer().lemmatize(w, pos='n')
    w = WordNetLemmatizer().lemmatize(w, pos='a')
    return w


# Tokenize and lemmatize
def preprocess(text):
    
    result = []
    
    for token in gensim.utils.simple_preprocess(text) :
        
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            
            # TODO: Apply lemmatize_stemming() on the token, then add to the results list
            result.append(lemmatize_stemming(token))
    
    return result

dict_topics_name={0:'criminality', 1:'economy', 2: 'society',
                  3:'politics', 4:'environment', 5:'minor news',
                  6:'local politic', 7:'international news', 8:'international politic',9:'foreign criminality '}


def model_result(unknown_text):
    loaded_model=load_lda_model('lda/models/finalized_model_lda_tfidf.sav')
    tfidf_trained=tfidf_transformer('lda/models/tfid_transformer')
    dictionary_trained= dictionary_transformer('lda/models/dictionary')

    # Data preprocessing step for the unseen document
    bow_vector = dictionary_trained.doc2bow(preprocess(unknown_text))

    tfidf_sentence=tfidf_trained[bow_vector]

    counter=0

    for index, score in sorted(loaded_model[tfidf_sentence], key=lambda tup: -1*tup[1]):
        while counter < 1 :
            topics_name= dict_topics_name[index]
            result_mod = f"Score: {round(float(score),2)} \n Topic main words: {loaded_model.print_topic(index, 5)}"
            counter += 1

    return topics_name, result_mod