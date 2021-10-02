import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Task 1 #2
def plot_bar_graph(basepath, list_of_files):
    target_names = list_of_files['target_names']
    target_freq = [0] * len(target_names)

    for i in range(len(list_of_files['filenames'])):
        filename = list_of_files['filenames'][i]
        for x in range(len(target_names)):
            if (filename[len(basepath) + 1:].startswith(target_names[x])):
                target_freq[x] = target_freq[x] + 1
                break

    data = {}
    for i in range(len(target_freq)):
        data[list_of_files['target_names'][i]] = target_freq[i]

    classes = list(data.keys())
    numberofarticles = list(data.values())
    
    fig = plt.figure(figsize=(10, 5))
    plt.bar(classes, numberofarticles, color='maroon', width=0.4)
    
    plt.xlabel("Classes")
    plt.ylabel("Number of articles")
    plt.title("Distribution of the instances in each class")
    
    fig.savefig('BBC-distribution.pdf', dpi=fig.dpi)

# Task 1 #3
def load_bbc_files(basepath):
    print("Reading files...")
    return sklearn.datasets.load_files(basepath, encoding='latin1', load_content=True, random_state=0)

# Task 1 #4
def pre_process_data_set(list_of_files):
    #print("TODO")

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(list_of_files)

    return (vectorizer.get_feature_names(), X)

# Task 1 #5
def split_dataset(X, testsize, randomstate):
    X_train, X_test = train_test_split(X, test_size=testsize, random_state=randomstate)
    return (X_train, X_test)

def main():
    basepath = "data\\BBC"
    list_of_files = load_bbc_files(basepath)

    plot_bar_graph(basepath, list_of_files)
    words, word_frequency = pre_process_data_set(list_of_files)

    #print(words)
    #print(word_frequency.toarray())

    train, test = split_dataset(word_frequency.toarray(), 0.20, None)
    
if __name__ == "__main__":
    main()