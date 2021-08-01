import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd

import acquire

def basic_clean(string):

    '''take in a string, lowercase all, normalize & replace non letters/numbers/whitespace or single quote'''

    # normalize string
    string = unicodedata.normalize('NFKD', string)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    
    # remove anything that is not a through z, a number, a single quote, or whitespace
    string = re.sub(r"[^\w'\s]", '', string).lower()

    return string

def tokenize(string):
    '''take in string and tokenize all words'''
    
    #create tokenizer ojbject
    tokenizer = nltk.tokenize.ToktokTokenizer()
    
    #execute on string
    string = tokenizer.tokenize(string, return_str=True)
    
    return string

def stem(string):
    '''take in string and return text after stemming all words'''
    # Create the nltk stemmer object, then use it
    ps = nltk.porter.PorterStemmer()
    
    # apply stemming transformation 
    stems = [ps.stem(word) for word in string.split()]
    string = ' '.join(stems)
    
    return string

def lemmatize(string):
    '''take in string and return text after lemmatizing each word'''
    # create lemmatizer object
    wnl = nltk.stem.WordNetLemmatizer()
    
    # apply lemmatizer to string
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    string = ' '.join(lemmas)

    return string

def remove_stopwords(string, extra_words = [], exclude_words = []):
    '''take in string and return text after removing English stopwords'''
    # define stopwords
    stopword_list = stopwords.words('english')
    
    # Remove 'exclude_words' from stopword_list to keep these in my text.
    stopword_list = set(stopword_list) - set(exclude_words)
    
     # Add in 'extra_words' to stopword_list.
    stopword_list = stopword_list.union(set(extra_words))
    
    # split article into indiv words
    words = string.split()
    # filter each word removing words in the stopword list
    filtered_words = [w for w in words if w not in stopword_list]
    
    # reconstitute article from filtered words list
    string = ' '.join(filtered_words)
    
    return string
    
def prep_df(df, feature, extra_words = [], exclude_words = []):
    df['clean'] = df[feature].apply(basic_clean).apply(tokenize).apply(remove_stopwords)
    df['stemmed'] = df[feature].apply(basic_clean).apply(tokenize).apply(stem).apply(remove_stopwords)
    df['lemmatized'] = df[feature].apply(basic_clean).apply(tokenize).apply(lemmatize).apply(remove_stopwords)
    return df[['title', feature, 'clean', 'stemmed', 'lemmatized']]