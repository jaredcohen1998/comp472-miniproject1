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
    fig.savefig('bbc-distribution.pdf', dpi=fig.dpi)

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


def nb_multinomial_classify(X_train, y_train, smoothing=1):
    nb = MultinomialNB(alpha=smoothing)
    nb.fit(X_train, y_train)
    return nb

# Task 1 #7


def generate_performance_report(file_name, try_num, y_test, predicted, mode="w"):
    f = open(file_name, mode)
    print("Writing performance report %d..." % try_num)

    if (mode == "a"):
        f.write("\n\n\n")

    # a
    f.write(
        "---------- MultinomialNB default values, try %d ----------\n" % try_num)

    # b
    f.write("Confusion Matrix \n\n{0}\n\n".format(
        confusion_matrix(y_test, predicted)))
    # c
    f.write("Classification Report \n\n{0}\n\n".format(classification_report(y_test, predicted,
                                                                             target_names=list_of_files['target_names'])))

    # d
    f.write("Accuracy Score: {0}\n\n".format(
        accuracy_score(y_test, predicted)))
    f.write(
        "Macro-average F1 Score: {0}\n\n".format(f1_score(y_test, predicted, average='macro')))
    f.write("Weighted-average F1 Score: {0}\n\n".format(
        f1_score(y_test, predicted, average='weighted')))

    # e
    f.write("Prior probability of each class: {0}\n\n".format(
        list(zip(list_of_files['target_names'], np.exp(mnb.class_log_prior_)))))

    # f
    f.write("Vocabulary size: {0}\n\n".format(len(words)))

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

    f.write(
        "Word-tokens in each class: {0}\n\n".format(list(zip(list_of_files['target_names'], word_count))))

    # h
    f.write("Word-tokens in entire corpus: {0}\n\n".format(sum(word_count)))

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
    business_zero_freq_percent = (business_zero_freq_count/word_count[0])*100
    entertainment_zero_freq_percent = (entertainment_zero_freq_count /
                                       word_count[1])*100
    politics_zero_freq_percent = (politics_zero_freq_count/word_count[2])*100
    sport_zero_freq_percent = (sport_zero_freq_count/word_count[3])*100
    tech_zero_freq_percent = (tech_zero_freq_count/word_count[4])*100
    total_zero_freq_percent = business_zero_freq_percent + entertainment_zero_freq_percent + \
        politics_zero_freq_percent + sport_zero_freq_percent + tech_zero_freq_percent

    # i
    f.write(
        "# of zero-frequency words in Business: {:d} -- {:0.2f} percent".format(business_zero_freq_count, business_zero_freq_percent))
    f.write(
        "# of zero-frequency words in Entertainment: {:d} -- {:0.2f} percent".format(entertainment_zero_freq_count, entertainment_zero_freq_percent))
    f.write(
        "# of zero-frequency words in Politics: {:d} -- {:0.2f} percent".format(politics_zero_freq_count, politics_zero_freq_percent))
    f.write(
        "# of zero-frequency words in Sport: {:d} -- {:0.2f} percent".format(sport_zero_freq_count, sport_zero_freq_percent))
    f.write(
        "# of zero-frequency words in Tech: {:d} -- {:0.2f} percent".format(tech_zero_freq_count, tech_zero_freq_percent))

    # j
    f.write(
        "# of zero-frequency words in entire corpus: {:d} -- {:0.2f} percent\n\n".format(total_zero_freq_count, total_zero_freq_percent))

    # k
    fav_word_1 = 29418
    fav_word_2 = 29420
    f.write("The log probability of {0} is {1} and of {2} is {3}\n\n".format(
        words[fav_word_1], list(zip(list_of_files['target_names'], feature_log_probability(mnb.feature_log_prob_, fav_word_1))), words[fav_word_2], list(zip(list_of_files['target_names'], feature_log_probability(mnb.feature_log_prob_, fav_word_2)))))

    f.close()


def count_zero_freq_words(c_words):
    count = 0
    for w in c_words:
        if (w == 0):
            count += 1
    return count


def feature_log_probability(mnb_log_vector, feature_indx):
    exp_sum = np.zeros(
        len(list_of_files['target_names'])) + mnb.class_log_prior_
    for i, x in enumerate(mnb_log_vector):
        exp_sum[i] += x[feature_indx]
    return exp_sum


def main():
    # global vars
    global mnb
    global list_of_files
    global words
    global word_frequency

    basepath = "data\\BBC"
    # Loading the corpus files. Part 3 of task 1
    list_of_files = load_bbc_files(basepath)

    # Pre-processing the dataset. Part 4 of task 1
    words, word_frequency = pre_process_data_set(list_of_files)

    # Plotting the bar graph. Part 2 of task 1
    plot_bar_graph(basepath, list_of_files)

    # Splitting the dataset 80:20. Part 5 of task 1
    X_train, X_test, y_train, y_test = split_dataset(
        word_frequency.toarray(), list_of_files['target'], 0.20, None)

    # Train a Multinomial NB classifier on the training set and test it on the test set. Part 6 of task 1
    mnb = nb_multinomial_classify(X_train, y_train)
    predicted = mnb.predict(X_test)

    # Generate bbc performance report. Part 7 of task 1
    generate_performance_report('bbc-performance.txt', 1, y_test, predicted)

    # Repeating stpes 6 and 7. Part 8 of task 1
    mnb = nb_multinomial_classify(X_train, y_train)
    predicted = mnb.predict(X_test)
    generate_performance_report(
        'bbc-performance.txt', 2, y_test, predicted, "a")

    # Repeating stpes 6 and 7, but changing the smoothing value to 0.0001. Part 9 of task 1
    mnb = nb_multinomial_classify(X_train, y_train, 0.0001)
    predicted = mnb.predict(X_test)
    generate_performance_report(
        'bbc-performance.txt', 3, y_test, predicted, "a")

    # Repeating stpes 6 and 7, but changing the smoothing value to 0.9. Part 10 of task 1
    mnb = nb_multinomial_classify(X_train, y_train, 0.9)
    predicted = mnb.predict(X_test)
    generate_performance_report(
        'bbc-performance.txt', 4, y_test, predicted, "a")


if __name__ == "__main__":
    main()
