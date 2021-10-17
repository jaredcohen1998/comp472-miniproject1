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
print(f1_score(y_test, predicted, average=None))
print(accuracy_score(y_test, predicted))
print("starting desicion tree")
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, y_train)
decisionTreePredictions = clf.predict(X_test)
print(confusion_matrix(y_test, decisionTreePredictions))
print(classification_report(y_test, decisionTreePredictions))
print(f1_score(y_test, decisionTreePredictions, average=None))
print(accuracy_score(y_test, decisionTreePredictions))
#c TOP-DT?
print("Starting top decision Tree")
param_grid = {'criterion' : ['gini', 'entropy'], 'max_depth' : [3, 4], 'min_samples_split' : [3, 4, 5]}
clf = GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid, scoring='f1_macro') #using f1 for unbalanced classes.
clf = clf.fit(X_train, y_train)
print(clf.best_params_)
topDTPredictions = clf.predict(X_test)

print(confusion_matrix(y_test, topDTPredictions))
print(classification_report(y_test, topDTPredictions))
print(f1_score(y_test, topDTPredictions, average=None))
print(accuracy_score(y_test, topDTPredictions))
#d PER
print("starting perceptron")
clf = Perceptron(random_state=0)
clf = clf.fit(X_train, y_train)
perceptronPredictions = clf.predict(X_test)
print(confusion_matrix(y_test, perceptronPredictions))
print(classification_report(y_test, perceptronPredictions))
print(f1_score(y_test, perceptronPredictions, average=None))
print(accuracy_score(y_test, perceptronPredictions))
#e base-MLP
print("starting MLP")
clf = MLPClassifier(random_state=0, solver='sgd',activation='logistic').fit(X_train, y_train)
MLPPredictions = clf.predict(X_test)
print(confusion_matrix(y_test, MLPPredictions))
print(classification_report(y_test, MLPPredictions))
print(f1_score(y_test, MLPPredictions, average=None))
print(accuracy_score(y_test, MLPPredictions))
#f top-MLP
print("Starting top MLP")
param_grid = {'activation' : ['logistic', 'tanh', 'relu', 'identity'], 'hidden_layer_sizes' : [(30, 50,), (10,10,10,)], 'solver' : ['adam', 'sgd']}
clf = GridSearchCV(MLPClassifier(),param_grid=param_grid, scoring='f1_macro') #using f1 for unbalanced classes.
clf = clf.fit(X_train, y_train)
print(clf.best_params_)
topMLPPredictions = clf.predict(X_test)

print(confusion_matrix(y_test, topMLPPredictions))
print(classification_report(y_test, topMLPPredictions))
print(f1_score(y_test, topMLPPredictions, average=None))
print(accuracy_score(y_test, topMLPPredictions))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
