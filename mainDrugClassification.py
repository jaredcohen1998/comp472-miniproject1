# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from pandas.api.types import CategoricalDtype
import warnings
import statistics

warnings.filterwarnings("ignore")
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def plot_bar_graph(data):
    print("Plotting bar graph...")
    dataFrequency = [0, 0, 0, 0, 0]
    for x in data['Drug']:
        if (x == 'drugA'):
            dataFrequency[0] += 1
        if (x == 'drugB'):
            dataFrequency[1] += 1
        if (x == 'drugC'):
            dataFrequency[2] += 1
        if (x == 'drugX'):
            dataFrequency[3] += 1
        if (x == 'drugY'):
            dataFrequency[4] += 1


    fig = plt.figure(figsize=(10, 5))
    plot = plt.bar(['drugA', 'drugB', 'drugC', 'drugX', 'drugY'], dataFrequency, color='maroon', width=0.4)

    # Taken from https://www.pythonprogramming.in/how-to-plot-a-very-simple-bar-chart-using-matplotlib.html
   # for value in plot:
  #      height = value.get_height()
   #     plt.text(value.get_x() + value.get_width()/2, 1.002*height,
    #             '%d' % int(height), ha='center', va='bottom')

    plt.title("Distribution of drugs")
    plt.xlabel("Drug Type")
    plt.ylabel("Number of Occurrences")
    plt.show()
    fig.savefig('Drug-distribution.pdf', dpi=fig.dpi)

def split_dataset(X, y, randomstate):
    X_train, X_test ,y_train,y_test= train_test_split(X, y,  random_state=randomstate)
    #y_train,y_test = train_test_split(y, test_size=testsize, random_state=randomstate)
    return (X_train, X_test, y_train, y_test)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
data = pandas.read_csv('drug200.csv')

plot_bar_graph(data)

convertedData = pandas.get_dummies(data, columns=['Sex', 'BP', 'Cholesterol'])

classes = convertedData['Drug']

features = convertedData.loc[:, convertedData.columns != 'Drug']



X_train, X_test, y_train, y_test =  split_dataset(features, classes, None)


#a
print("starting GaussianNB")
nb = GaussianNB()
nb.fit(X_train, y_train)
predicted = nb.predict(X_test)
print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted))
print(f1_score(y_test, predicted, average='micro'))
print(f1_score(y_test, predicted, average='macro'))
print(f1_score(y_test, predicted, average='weighted'))
print(accuracy_score(y_test, predicted))

#8
nbAccuracyList = []
nbMacroList = []
nbWeightedList = []

for x in range(10):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    predicted = nb.predict(X_test)

    nbAccuracyList.append(f1_score(y_test,predicted,average='micro'))
    nbMacroList.append(f1_score(y_test,predicted,average='macro'))
    nbWeightedList.append(f1_score(y_test,predicted,average='weighted'))

print("Average accuracy for 10 Gaussian NB fits:" + str(statistics.mean(nbAccuracyList)))
print("Standard deviation of accuracy for 10 Gaussian NB fits:" + str(statistics.stdev(nbAccuracyList)))
print("Average macro F1-Score for 10 Gaussian NB fits:" + str(statistics.mean(nbMacroList)))
print("Standard deviation of macro F1-Score for 10 Gaussian NB fits:" + str(statistics.stdev(nbMacroList)))
print("Average weighted F1-Score for 10 Gaussian NB fits:" + str(statistics.mean(nbWeightedList)))
print("Standard deviation of weighted F1-Score for 10 Gaussian NB fits:" + str(statistics.stdev(nbWeightedList)))

#b
print("starting decision tree")
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, y_train)
decisionTreePredictions = clf.predict(X_test)
print(confusion_matrix(y_test, decisionTreePredictions))
print(classification_report(y_test, decisionTreePredictions))
print(f1_score(y_test, decisionTreePredictions, average=None))
print(accuracy_score(y_test, decisionTreePredictions))

#8
DTAccuracyList = []
DTMacroList = []
DTWeightedList = []

for x in range(10):
    clf = DecisionTreeClassifier(random_state=0)
    clf = clf.fit(X_train, y_train)
    decisionTreePredictions = clf.predict(X_test)

    DTAccuracyList.append(f1_score(y_test,decisionTreePredictions,average='micro'))
    DTMacroList.append(f1_score(y_test,decisionTreePredictions,average='macro'))
    DTWeightedList.append(f1_score(y_test,decisionTreePredictions,average='weighted'))

print("Average accuracy for 10 decision tree fits:" + str(statistics.mean(DTAccuracyList)))
print("Standard deviation of accuracy for decision tree fits:" + str(statistics.stdev(DTAccuracyList)))
print("Average macro F1-Score for 10 decision tree fits:" + str(statistics.mean(DTMacroList)))
print("Standard deviation of macro F1-Score for 10 decision tree fits:" + str(statistics.stdev(DTMacroList)))
print("Average weighted F1-Score for 10 decision tree fits:" + str(statistics.mean(DTWeightedList)))
print("Standard deviation of weighted F1-Score for 10 decision tree fits:" + str(statistics.stdev(DTWeightedList)))


#c TOP-DT?
print("Starting top decision Tree")
param_grid = {'criterion' : ['gini', 'entropy'], 'max_depth' : [3, 4], 'min_samples_split' : [3, 4, 5]}
clf = GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid, scoring='f1_macro') #using f1 for unbalanced classes.
clf = clf.fit(X_train, y_train)
print(clf.best_params_)
bestTDT = DecisionTreeClassifier(random_state=0)
bestTDT.set_params(**clf.best_params_)
topDTPredictions = clf.predict(X_test)

print(confusion_matrix(y_test, topDTPredictions))
print(classification_report(y_test, topDTPredictions))
print(f1_score(y_test, topDTPredictions, average=None))
print(accuracy_score(y_test, topDTPredictions))

#8
TDTAccuracyList = []
TDTMacroList = []
TDTWeightedList = []

for x in range(10):

    bestTDT = bestTDT.fit(X_train, y_train)
    topTDTPredictions = bestTDT.predict(X_test)

    TDTAccuracyList.append(f1_score(y_test,topTDTPredictions,average='micro'))
    TDTMacroList.append(f1_score(y_test,topTDTPredictions,average='macro'))
    TDTWeightedList.append(f1_score(y_test,topTDTPredictions,average='weighted'))

print("Average accuracy for 10 best decision tree fits:" + str(statistics.mean(TDTAccuracyList)))
print("Standard deviation of accuracy for best decision tree fits:" + str(statistics.stdev(TDTAccuracyList)))
print("Average macro F1-Score for 10 best decision tree fits:" + str(statistics.mean(TDTMacroList)))
print("Standard deviation of macro F1-Score for 10 best decision tree fits:" + str(statistics.stdev(TDTMacroList)))
print("Average weighted F1-Score for 10 best decision tree fits:" + str(statistics.mean(TDTWeightedList)))
print("Standard deviation of weighted F1-Score for 10 best decision tree fits:" + str(statistics.stdev(TDTWeightedList)))

#d PER
print("starting perceptron")
clf = Perceptron(random_state=0)
clf = clf.fit(X_train, y_train)
perceptronPredictions = clf.predict(X_test)
print(confusion_matrix(y_test, perceptronPredictions))
print(classification_report(y_test, perceptronPredictions))
print(f1_score(y_test, perceptronPredictions, average=None))
print(accuracy_score(y_test, perceptronPredictions))

#8
PERAccuracyList = []
PERMacroList = []
PERWeightedList = []

for x in range(10):
    clf = Perceptron(random_state=0)
    clf = clf.fit(X_train, y_train)
    perceptronPredictions = clf.predict(X_test)
    PERAccuracyList.append(f1_score(y_test,perceptronPredictions,average='micro'))
    PERMacroList.append(f1_score(y_test,perceptronPredictions,average='macro'))
    PERWeightedList.append(f1_score(y_test,perceptronPredictions,average='weighted'))


print("Average accuracy for 10  perceptron fits:" + str(statistics.mean(PERAccuracyList)))
print("Standard deviation of accuracy for perceptron fits:" + str(statistics.stdev(PERAccuracyList)))
print("Average macro F1-Score for 10  perceptron fits:" + str(statistics.mean(PERMacroList)))
print("Standard deviation of macro F1-Score for 10  perceptron fits:" + str(statistics.stdev(PERMacroList)))
print("Average weighted F1-Score for 10  perceptron fits:" + str(statistics.mean(PERWeightedList)))
print("Standard deviation of weighted F1-Score for 10 perceptron fits:" + str(statistics.stdev(PERWeightedList)))

#e base-MLP
print("starting MLP")
clf = MLPClassifier(random_state=0, solver='sgd',activation='logistic', max_iter=200).fit(X_train, y_train)
MLPPredictions = clf.predict(X_test)
print(confusion_matrix(y_test, MLPPredictions))
print(classification_report(y_test, MLPPredictions))
print(f1_score(y_test, MLPPredictions, average=None))
print(accuracy_score(y_test, MLPPredictions))

#8
MLPAccuracyList = []
MLPMacroList = []
MLPWeightedList = []

for x in range(10):
    clf = MLPClassifier(random_state=0, solver='sgd',activation='logistic', max_iter=200).fit(X_train, y_train)
    MLPPredictions = clf.predict(X_test)

    MLPAccuracyList.append(f1_score(y_test,MLPPredictions,average='micro'))
    MLPMacroList.append(f1_score(y_test,MLPPredictions,average='macro'))
    MLPWeightedList.append(f1_score(y_test,MLPPredictions,average='weighted'))


print("Average accuracy for 10  MLP fits:" + str(statistics.mean(MLPAccuracyList)))
print("Standard deviation of accuracy for MLP fits:" + str(statistics.stdev(MLPAccuracyList)))
print("Average macro F1-Score for 10  MLP fits:" + str(statistics.mean(MLPMacroList)))
print("Standard deviation of macro F1-Score for 10  MLP fits:" + str(statistics.stdev(MLPMacroList)))
print("Average weighted F1-Score for 10 MLP fits:" + str(statistics.mean(MLPWeightedList)))
print("Standard deviation of weighted F1-Score for 10  MLP fits:" + str(statistics.stdev(MLPWeightedList)))


#f top-MLP
print("Starting top MLP")
param_grid = {'activation' : ['logistic', 'tanh', 'relu', 'identity'], 'hidden_layer_sizes' : [(30, 50,), (10,10,10,)], 'solver' : ['adam', 'sgd']}
clf = GridSearchCV(MLPClassifier(max_iter=200),param_grid=param_grid, scoring='f1_macro') #using f1 for unbalanced classes.
clf = clf.fit(X_train, y_train)
print(clf.best_params_)
topMLPPredictions = clf.predict(X_test)

print(confusion_matrix(y_test, topMLPPredictions))
print(classification_report(y_test, topMLPPredictions))
print(f1_score(y_test, topMLPPredictions, average=None))
print(accuracy_score(y_test, topMLPPredictions))

#8
TMLPAccuracyList = []
TMLPMacroList = []
TMLPWeightedList = []

for x in range(10):
    bestMLP = MLPClassifier(max_iter=200)
    bestMLP.set_params(**clf.best_params_)
    bestMLP = bestMLP.fit(X_train, y_train)
    topMLPPredictions = bestMLP.predict(X_test)

    TMLPAccuracyList.append(f1_score(y_test,topMLPPredictions,average='micro'))
    TMLPMacroList.append(f1_score(y_test,topMLPPredictions,average='macro'))
    TMLPWeightedList.append(f1_score(y_test,topMLPPredictions,average='weighted'))

print(TMLPAccuracyList)
print(TMLPMacroList)
print(TMLPWeightedList)
print("Average accuracy for 10 best MLP fits:" + str(statistics.mean(TMLPAccuracyList)))
print("Standard deviation of accuracy for best MLP fits:" + str(statistics.stdev(TMLPAccuracyList)))
print("Average macro F1-Score for 10 best MLP fits:" + str(statistics.mean(TMLPMacroList)))
print("Standard deviation of macro F1-Score for 10 best MLP fits:" + str(statistics.stdev(TMLPMacroList)))
print("Average weighted F1-Score for 10 best MLP fits:" + str(statistics.mean(TMLPWeightedList)))
print("Standard deviation of weighted F1-Score for 10 best MLP fits:" + str(statistics.stdev(TMLPWeightedList)))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
