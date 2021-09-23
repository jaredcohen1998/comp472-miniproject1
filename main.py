import os
import numpy as np
import re
import matplotlib.pyplot as plt
import sklearn.datasets

def main():
    list_of_files = sklearn.datasets.load_files('data\\BBC', encoding='latin1', load_content=True, random_state=0)
    print('Index of 1: ')
    print(list_of_files['target_names'][1])
    print(list_of_files['filenames'][1])
    print(list_of_files['data'][1])
    #word_frequency(list_of_files)
    
if __name__ == "__main__":
    main()