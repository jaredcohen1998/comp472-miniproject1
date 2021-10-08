import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# Task 1 #2
def plot_bar_graph(basepath, list_of_files):
    print("Plotting bar graph...")
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
    plot = plt.bar(classes, numberofarticles, color='maroon', width=0.4)
    
    # Taken from https://www.pythonprogramming.in/how-to-plot-a-very-simple-bar-chart-using-matplotlib.html
    for value in plot:
        height = value.get_height()
        plt.text(value.get_x() + value.get_width()/2,1.002*height,'%d' % int(height), ha='center', va='bottom')

    plt.title("Distribution of the instances in each class")
    plt.xlabel("Classes")
    plt.ylabel("Number of articles")
    plt.show()
    fig.savefig('BBC-distribution.pdf', dpi=fig.dpi)

# Task 1 #3
def load_bbc_files(basepath):
    print("Reading files...")
    return sklearn.datasets.load_files(basepath, encoding='latin1', load_content=True, random_state=0)

# Task 1 #4
def pre_process_data_set(list_of_files):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(list_of_files['data'])

    return (vectorizer.get_feature_names_out(), X)

# Task 1 #5
def split_dataset(X, y, testsize, randomstate):
    X_train, X_test ,y_train,y_test= train_test_split(X, y, test_size=testsize, random_state=randomstate)
    #y_train,y_test = train_test_split(y, test_size=testsize, random_state=randomstate)
    return (X_train, X_test, y_train, y_test)

def main():
    basepath = "data\\BBC"
    list_of_files = load_bbc_files(basepath)
    print(list_of_files['target'].shape)
    plot_bar_graph(basepath, list_of_files)
    words, word_frequency = pre_process_data_set(list_of_files)

    print(words)
    print(word_frequency.toarray())

    train, test, y_train, y_test = split_dataset(word_frequency.toarray(), list_of_files['target'], 0.20, None)
    print(train)
    print(train.shape)
    print(test.shape)
    testData = [1, 2, 3, 4, 5 ,6, 7, 8, 9, 10]
    testData2 = [1, 2, 3, 4, 5 ,6, 7, 8, 9, 10]
    train1, test1, train2, test2 = split_dataset(testData, testData2, 0.20, None)
    print(train1)
    print(train2)
    nb = MultinomialNB()
    nb.fit(train, y_train)
    predicted = nb.predict(test)
    print(list_of_files['target_names'])
    print(confusion_matrix(y_test, predicted))
    print(classification_report(y_test, predicted, target_names=list_of_files['target_names']))
    print(f1_score(y_test, predicted, average=None))
    print(accuracy_score(y_test, predicted))
    countArray = [0, 0, 0, 0, 0]
    for x in y_train:
        if(x == 'business') countArray[0]++
if __name__ == "__main__":
    main()