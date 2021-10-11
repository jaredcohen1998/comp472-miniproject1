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
        plt.text(value.get_x() + value.get_width()/2, 1.002*height,
                 '%d' % int(height), ha='center', va='bottom')

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testsize, random_state=randomstate)
    return (X_train, X_test, y_train, y_test)

# Task 1 #6


def nb_multinomial_classify(X_train, y_train, X_test):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    return nb.predict(X_test)


def count_zero_freq_words(c_words):
    count = 0
    for w in c_words:
        if (w == 0):
            count += 1
    return count


def main():
    basepath = "data\\BBC"
    list_of_files = load_bbc_files(basepath)
    plot_bar_graph(basepath, list_of_files)
    words, word_frequency = pre_process_data_set(list_of_files)

    # print("Words ", len(words))
    # print("Word freq ", len(word_frequency.toarray()))

    print("Target Names", list_of_files['target_names'])
    X_train, X_test, y_train, y_test = split_dataset(
        word_frequency.toarray(), list_of_files['target'], 0.20, None)
    predicted = nb_multinomial_classify(X_train, y_train, X_test)

    # Task 1 #7
    # a
    print("---------- MultinomialNB default values, try 1 ----------\n")

    # b
    print("Confusion Matrix \n\n{0}\n\n".format(
        confusion_matrix(y_test, predicted)))
    # c
    print("Classification Report \n\n{0}\n\n".format(classification_report(y_test, predicted,
          target_names=list_of_files['target_names'])))

    # d
    print("F1 Score \n\n{0}\n\n".format(list(
        zip(list_of_files["target_names"], f1_score(y_test, predicted, average=None)))))
    print("Accuracy Score \n\n{0}\n\n".format(
        accuracy_score(y_test, predicted)))

    # e
    priors = np.zeros(5)
    for x in list_of_files['target']:
        if (x == 0):
            priors[0] += 1
        if (x == 1):
            priors[1] += 1
        if (x == 2):
            priors[2] += 1
        if (x == 3):
            priors[3] += 1
        if (x == 4):
            priors[4] += 1

    priors /= len(list_of_files['target'])
    print("Prior probability of each class: {0}".format(
        list(zip(list_of_files['target_names'], priors))))

    # f
    print("Vocabulary size: {0}".format(len(words)))

    # g
    word_count = np.zeros(5, int)
    for i, w in enumerate(word_frequency.toarray()):
        if (list_of_files['target'][i] == 0):
            word_count[0] += sum(w)
        if (list_of_files['target'][i] == 1):
            word_count[1] += sum(w)
        if (list_of_files['target'][i] == 2):
            word_count[2] += sum(w)
        if (list_of_files['target'][i] == 3):
            word_count[3] += sum(w)
        if (list_of_files['target'][i] == 4):
            word_count[4] += sum(w)

    print(
        "Word-tokens in each class: {0}".format(list(zip(list_of_files['target_names'], word_count))))

    # h
    print(sum(word_count))

    # Summing word frequencies per class
    business_freq_array = np.zeros(len(words))
    entertainment_freq_array = np.zeros(len(words))
    politics_freq_array = np.zeros(len(words))
    sport_freq_array = np.zeros(len(words))
    tech_freq_array = np.zeros(len(words))
    for i, w in enumerate(word_frequency.toarray()):
        if (list_of_files['target'][i] == 0):
            business_freq_array += w
        if (list_of_files['target'][i] == 1):
            entertainment_freq_array += w
        if (list_of_files['target'][i] == 2):
            politics_freq_array += w
        if (list_of_files['target'][i] == 3):
            sport_freq_array += w
        if (list_of_files['target'][i] == 4):
            tech_freq_array += w

    # Calculating # of zero-freq words in each class
    business_zero_freq_count = count_zero_freq_words(business_freq_array)
    entertainment_zero_freq_count = count_zero_freq_words(
        entertainment_freq_array)
    politics_zero_freq_count = count_zero_freq_words(politics_freq_array)
    sport_zero_freq_count = count_zero_freq_words(sport_freq_array)
    tech_zero_freq_count = count_zero_freq_words(tech_freq_array)
    total_zero_freq_count = business_zero_freq_count + entertainment_zero_freq_count + \
        politics_zero_freq_count + sport_zero_freq_count + tech_zero_freq_count

    # Calculating % of zero-freq words in each class
    business_zero_freq_percent = business_zero_freq_count/word_count[0]
    entertainment_zero_freq_percent = entertainment_zero_freq_count / \
        word_count[1]
    politics_zero_freq_percent = politics_zero_freq_count/word_count[2]
    sport_zero_freq_percent = sport_zero_freq_count/word_count[3]
    tech_zero_freq_percent = tech_zero_freq_count/word_count[4]
    total_zero_freq_percent = business_zero_freq_percent + entertainment_zero_freq_percent + \
        politics_zero_freq_percent + sport_zero_freq_percent + tech_zero_freq_percent

    # i
    print(
        "# of zero-frequency words in Business: {:d} -- {:0.4f} percent".format(business_zero_freq_count, business_zero_freq_percent))
    print(
        "# of zero-frequency words in Entertainment: {:d} -- {:0.4f} percent".format(entertainment_zero_freq_count, entertainment_zero_freq_percent))
    print(
        "# of zero-frequency words in Politics: {:d} -- {:0.4f} percent".format(politics_zero_freq_count, politics_zero_freq_percent))
    print(
        "# of zero-frequency words in Sport: {:d} -- {:0.4f} percent".format(sport_zero_freq_count, sport_zero_freq_percent))
    print(
        "# of zero-frequency words in Tech: {:d} -- {:0.4f} percent".format(tech_zero_freq_count, tech_zero_freq_percent))

    # j
    print(
        "# of zero-frequency words in entire corpus: {:d} -- {:0.4f} percent".format(total_zero_freq_count, total_zero_freq_percent))


if __name__ == "__main__":
    main()
