import os
import numpy as np
import re
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer

def main():
    list_of_files = sklearn.datasets.load_files('data\\BBC', encoding='latin1', load_content=True, random_state=0)
    print('Index of 1: ')
    print(len(list_of_files['target_names']))
    print(len(list_of_files['filenames']))
    print(len(list_of_files['data']))

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(list_of_files['data'])
    print(vectorizer.get_feature_names())
    #print(X.toarray())
    
if __name__ == "__main__":
    main()