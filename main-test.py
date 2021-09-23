import os
import numpy as np
import re
import matplotlib.pyplot as plt

def listf(directory):
    files = []
    folders = []

    for f in os.scandir(directory):
        if f.is_file():
            print(f.path)
            if (f.path != "data\\BBC\\README.TXT"):
                files.append(f.path)
        elif f.is_dir():
            folders.append(f.path)

    for d in list(folders):
        f = listf(d)
        files.extend(f)

    return files

def find_files():
    basepath = "data\\BBC"

    files = listf(basepath)
    return files

def word_frequency(list_of_files):
    total_word_list = []
    words_list_for_a_class = []
    old_file_category = ''

    for i in range(len(list_of_files)):
        fileName = list_of_files[i]

        tokens = fileName.split("\\")

        file_category = tokens[len(tokens) - 2] # business, entertainment, politics, sport, or tech
        if (old_file_category == ''):
            old_file_category = file_category

        if (file_category != old_file_category):
            unique, counts = np.unique(words_list_for_a_class, return_counts=True)
            print(file_category)
            print('=================================')
            print(np.asarray((unique, counts)).T)
            print('=================================')

            old_file_category = file_category
            words_list_for_a_class = []

        with open(fileName, 'r') as f:
            for line in f:
                for word in line.split():
                    word = word.translate(str.maketrans('', '', '!.,;()"')) # remove puncuation
                    if (word != ''):
                        words_list_for_a_class.append(word)

    #plt.plot(1, 2)
    #plt.show()

def main():
    list_of_files = find_files()
    word_frequency(list_of_files)
    
if __name__ == "__main__":
    main()