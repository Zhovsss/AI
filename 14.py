'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score



def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree

    ** Note that your implementation must follow this API**
    '''


    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape
    iter = 100
    acc_list_main = []
    acc_list_main_1 = []
    acc_list_main_3 = []

    for i in range(iter):
      list_of_indices = [0,26,52,78,104,130,156,182,208,234,267]
      # shuffle the data
      idx = np.arange(n)
      np.random.seed(13)
      np.random.shuffle(idx)
      X = X[idx]
      y = y[idx]


      # 10-folds splitting
      for i in range(len(list_of_indices)-1):
        # split the data
        Xtest = X[list_of_indices[i]:list_of_indices[i+1],:]  # train on first 100 instances
        ytest = y[list_of_indices[i]:list_of_indices[i+1],:]
        if i == 0:
          Xtrain_tmp = X[list_of_indices[1]:list_of_indices[10],:]
          ytrain_tmp = y[list_of_indices[1]:list_of_indices[10],:]  # test on remaining instances
        elif i == 9:
          Xtrain_tmp = X[list_of_indices[0]:list_of_indices[9],:]
          ytrain_tmp = y[list_of_indices[0]:list_of_indices[9],:]
        else:
          index_1 = (0, list_of_indices[i])
          index_2 = (list_of_indices[i+1], 267)
          X_train_1 = X[index_1[0]:index_1[1],:]
          X_train_2 = X[index_2[0]:index_2[1],:]
          y_train_1 = y[index_1[0]:index_1[1],:]
          y_train_2 = y[index_2[0]:index_2[1],:]
          Xtrain_tmp = np.concatenate((X_train_1, X_train_2), axis=0)
          ytrain_tmp = np.concatenate((y_train_1, y_train_2), axis=0)
        acc_list = []
        acc_list_1 = []
        acc_list_3 = []
        for j in range(1, 11, 1):
          start_index = 0
          end_index = int(len(Xtrain_tmp) * j / 10)
          Xtrain = Xtrain_tmp[start_index:end_index, :]
          ytrain = ytrain_tmp[start_index:end_index, :]
          # train the decision tree
          clf = tree.DecisionTreeClassifier()
          clf = clf.fit(Xtrain,ytrain)
          clf_1 = tree.DecisionTreeClassifier(max_depth=1)
          clf_1 = clf_1.fit(Xtrain,ytrain)
          clf_3 = tree.DecisionTreeClassifier(max_depth=3)
          clf_3 = clf_3.fit(Xtrain,ytrain)

          # output predictions on the remaining data
          y_pred = clf.predict(Xtest)
          y_pred_1 = clf_1.predict(Xtest)
          y_pred_3 = clf_3.predict(Xtest)

          # compute the training accuracy of the model
          meanDecisionTreeAccuracy = accuracy_score(ytest, y_pred)
          acc_list.append(meanDecisionTreeAccuracy)
          meanDecisionTreeAccuracy_1 = accuracy_score(ytest, y_pred_1)
          acc_list_1.append(meanDecisionTreeAccuracy_1)
          meanDecisionTreeAccuracy_3 = accuracy_score(ytest, y_pred_3)
          acc_list_3.append(meanDecisionTreeAccuracy_3)
        acc_list_main.append(acc_list)
        acc_list_main_1.append(acc_list_1)
        acc_list_main_3.append(acc_list_3)

    meanDecisionTreeAccuracy_lst = []
    stddevDecisionTreeAccuracy_lst = []
    meanDecisionStumpAccuracy_lst = []
    stddevDecisionStumpAccuracy_lst = []
    meanDT3Accuracy_lst = []
    stddevDT3Accuracy_lst = []

    percentages = np.arange(0.1, 1.1, 0.1)
    mean_accuracies = []
    std_accuracies = []

    for i in range(0, 10, 1):
      # TODO: update these statistics based on the results of your experiment
      meanDecisionTreeAccuracy = np.mean(np.array([acc_list_main[j][i]for j in range(0, iter*10)]))
      stddevDecisionTreeAccuracy = np.std(np.array([acc_list_main[j][i]for j in range(0, iter*10)]), axis=0)
      meanDecisionStumpAccuracy = np.mean(np.array([acc_list_main_1[j][i]for j in range(0, iter*10)]))
      stddevDecisionStumpAccuracy = np.std(np.array([acc_list_main_1[j][i]for j in range(0, iter*10)]), axis=0)
      meanDT3Accuracy = np.mean(np.array([acc_list_main_3[j][i]for j in range(0, iter*10)]))
      stddevDT3Accuracy = np.std(np.array([acc_list_main_3[j][i]for j in range(0, iter*10)]), axis=0)

      meanDecisionTreeAccuracy_lst.append(meanDecisionTreeAccuracy)
      stddevDecisionTreeAccuracy_lst.append(stddevDecisionTreeAccuracy)
      meanDecisionStumpAccuracy_lst.append(meanDecisionStumpAccuracy)
      stddevDecisionStumpAccuracy_lst.append(stddevDecisionStumpAccuracy)
      meanDT3Accuracy_lst.append(meanDT3Accuracy)
      stddevDT3Accuracy_lst.append(stddevDT3Accuracy)

    mean_accuracies.append(meanDecisionTreeAccuracy_lst)
    mean_accuracies.append(meanDecisionStumpAccuracy_lst)
    mean_accuracies.append(meanDT3Accuracy_lst)

    std_accuracies.append(stddevDecisionTreeAccuracy_lst)
    std_accuracies.append(stddevDecisionStumpAccuracy_lst)
    std_accuracies.append(stddevDT3Accuracy_lst)


    # Plotting
    plt.figure(figsize=(10, 6))

    classifiers = ['Unlimited Depth Tree', 'Decision Stumps', '3-Level Tree']
    for i in range(len(classifiers)):
        plt.errorbar(percentages * 100, mean_accuracies[i], yerr=std_accuracies[i], label=classifiers[i])
        # make certain that the return value matches the API specification

    # Add labels and title
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Mean Test Accuracy')
    plt.title('Learning Curve over Training Data')
    plt.legend()
    plt.show()

    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats



# Do not modify from HERE...
if __name__ == "__main__":

    stats = evaluatePerformance()
    print("Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")")
    print("Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")")
    print("3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")")
# ...to HERE.
