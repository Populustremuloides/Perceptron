from Arff import readArff, splitTrainTest, splitTrainTestNumpy
from Model import Perceptron
from MakeFakeData import makeData, getLineSample
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron as skPerceptron

from sklearn import datasets as skd
import numpy as np

debugFile = "debug_dataset.arff"
evaluationFile = "data_banknote_authentication_validation.arff"
votingFile = "voting-dataset.arff"

'''Please note that the code for splitting train/test randomly is in Aarff.py. It just made more sense with my other 
data handling functions than in main.'''

def main():

    '''
    # section 1.1.1
    df = readArff(debugFile)
    model = Perceptron(shuffle=False, deterministic=10, learningRate=0.1)
    features = list(df.columns[:2])
    target = df.columns[2]

    model.fit(df, features=features, target=target)
    score = model.score(df, features=features, target=target)

    print("debug score and weights")
    print(score)
    print(model.get_weights())

    # section 1.1.2
    df = readArff(evaluationFile)
    model = Perceptron(shuffle=False, deterministic=10, learningRate=0.1)
    features = list(df.columns[:4])
    target = df.columns[-1]

    model.fit(df, features=features, target=target)
    score = model.score(df, features=features, target=target)

    print("evaluation score and weights")
    print(score)
    print(model.get_weights())

    trainDf, testDf = splitTrainTest(df, 0)
    print(trainDf)
    print(testDf)

    # section 2.1
    separableDf = makeData([[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]], [0, 0, 1, 1], 2)
    nonSeparableDf = makeData([[-0.5, 0.5], [0.5, -0.5],[0.5, 0.5], [-0.5, -0.5]], [0, 0, 1, 1], 2)

    # save the data for later
    separableDf.to_csv("separableData.csv", index=False)
    nonSeparableDf.to_csv("nonSeparableData.csv", index=False)

    # section 2.2

    # separable *******************
    model = Perceptron(shuffle=False, deterministic=10, learningRate=0.1)
    features = list(separableDf.columns[:2])
    target = separableDf.columns[-1]
    model.fit(separableDf, features=features, target=target)
    score = model.score(separableDf, features=features, target=target)

    print("separable score and weights")
    print(score)
    print(model.get_weights())

    sDf0 = separableDf[separableDf["Y"] == 0]
    sDf1 = separableDf[separableDf["Y"] == 1]

    plt.scatter(sDf0["X0"], sDf0["X1"], label="0")
    plt.scatter(sDf1["X0"], sDf1["X1"], label="1")
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.title("Separable Data")


    # get the line from the weights:
    xs, ys = getLineSample(model.get_weights())
    plt.plot(xs, ys, label="decision line")
    plt.legend()
    plt.show()

    # non-separable *******************
    model = Perceptron(shuffle=False, deterministic=10, learningRate=0.1)
    features = list(nonSeparableDf.columns[:2])
    target = nonSeparableDf.columns[-1]
    model.fit(nonSeparableDf, features=features, target=target)
    score = model.score(nonSeparableDf, features=features, target=target)

    print("non-separable score and weights")
    print(score)
    print(model.get_weights())

    nsDf0 = nonSeparableDf[nonSeparableDf["Y"] == 0]
    nsDf1 = nonSeparableDf[nonSeparableDf["Y"] == 1]

    plt.scatter(nsDf0["X0"], nsDf0["X1"], label="0")
    plt.scatter(nsDf1["X0"], nsDf1["X1"], label="1")
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.title("non-Separable Data")

    xs, ys = getLineSample(model.get_weights())
    plt.plot(xs, ys, label="decision line")
    plt.legend()
    plt.show()



    # Section 3
    df = readArff(votingFile)

    # 3.1
    dataDict = {"trial":[], "test_accuracy":[], "train_accuracy":[], "num_epochs":[]}
    for i in range(5):
        numEpochs = i + 1
        testDf, trainDf = splitTrainTest(df, seed=i)

        model = Perceptron(shuffle=False, deterministic=numEpochs, learningRate=0.1)
        features = list(trainDf.columns[:len(trainDf.columns) - 1])
        target = df.columns[-1]

        model.fit(trainDf, features=features, target=target)

        trainScore = model.score(trainDf, features=features, target=target)
        testScore = model.score(testDf, features=features, target=target)

        dataDict["trial"].append(i)
        dataDict["train_accuracy"].append(trainScore)
        dataDict["test_accuracy"].append(testScore)
        dataDict["num_epochs"].append(numEpochs)
    outDf = pd.DataFrame.from_dict(dataDict)
    meanDf = outDf.mean()
    meanDf.to_csv("voting_scores_mean.csv", index=False)
    outDf.to_csv("voting_scores.csv", index=False)



    # 3.2
    weights = model.get_weights()
    features = features + ["bias (arbitrary)"]
    weightToFeature = dict(zip(weights, features))

    dataDict = {"feature":[],"weight/importance":[]}
    weights.sort()
    for weight in weights:
        dataDict["feature"].append(weightToFeature[weight])
        dataDict["weight/importance"].append(weight)
    outDf2 = pd.DataFrame.from_dict(dataDict)
    outDf2.to_csv("features_and_weights.csv", index=False)

    # 3.3
    missclassification = []
    for item in outDf["test_accuracy"]:
        missclassification.append(1 - item)
    plt.plot(outDf["num_epochs"], missclassification)
    plt.xlabel("number of epochs")
    plt.ylabel("misclassification rate (proportion)")
    plt.title("Perceptron Test Set Misclassification vs. Epochs")
    plt.show()


    # 4.1

    df = readArff(votingFile)
    testDf, trainDf = splitTrainTest(df, seed=0)
    features = list(trainDf.columns[:len(trainDf.columns) - 1])
    target = df.columns[-1]

    x_test = testDf[features]
    x_train = trainDf[features]

    x_test = x_test.to_numpy()
    x_train = x_train.to_numpy()

    y_test = np.asarray(list(testDf[target]))
    y_train = np.asarray(list(trainDf[target]))

    model = skPerceptron()

    model.fit(x_train, y_train)
    trainScore = model.score(x_train, y_train)
    testScore = model.score(x_test, y_test)

    print("sklearn train score")
    print(trainScore)
    print("sklearn test score")
    print(testScore)
    '''

    stuff = skd.load_iris()
    #X, y = sklearn.datasets.load_iris()
    X = stuff["data"]
    y = stuff["target"]
    Xtrain, Xtest, ytrain, ytest = splitTrainTestNumpy(X, y)
    model = skPerceptron()
    model.fit(Xtrain, ytrain)
    trainScore = model.score(Xtrain, ytrain)
    testScore = model.score(Xtest, ytest)
    print("SKlearn Iris train score: ")
    print(trainScore)
    print("SKlearn Iris test score: ")
    print(testScore)


    stuff = skd.load_breast_cancer()
    X = stuff["data"]
    y = stuff["target"]
    Xtrain, Xtest, ytrain, ytest = splitTrainTestNumpy(X, y)
    model = skPerceptron()
    model.fit(Xtrain, ytrain)
    trainScore = model.score(Xtrain, ytrain)
    testScore = model.score(Xtest, ytest)
    print("SKlearn breats cancer train score: ")
    print(trainScore)
    print("SKlearn breast cancer test score: ")
    print(testScore)

    model = skPerceptron(fit_intercept=False)
    model.fit(Xtrain, ytrain)
    trainScore = model.score(Xtrain, ytrain)
    testScore = model.score(Xtest, ytest)
    print("SKlearn breats cancer train score without intercept: ")
    print(trainScore)
    print("SKlearn breast cancer test score without intercept: ")
    print(testScore)

    model = skPerceptron(alpha=0.001)
    model.fit(Xtrain, ytrain)
    trainScore = model.score(Xtrain, ytrain)
    testScore = model.score(Xtest, ytest)
    print("SKlearn breats cancer train score without increased alpha: ")
    print(trainScore)
    print("SKlearn breast cancer test score with increased alpha: ")
    print(testScore)


if __name__ == "__main__":

    main()

