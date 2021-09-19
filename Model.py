import numpy as np
import copy

'''
please note that most +1 references are for the bias
'''

class Perceptron():
    def __init__(self, learningRate=0.1, deterministic=None, shuffle=False, initialWeights="normal"):
        # read in hyperparameters (I think these actually belong with the fit function. . .)
        self.learningRate = learningRate
        self.numEpochs = deterministic
        self.shuffle = shuffle
        self.initialWeights = initialWeights

        self.fitted = False
        self.indices = []

        np.random.seed(0)

    def predict(self, df, featureNames):

        predictions = []
        for index, row in df.iterrows():
            inputs = list(row[featureNames])
            if "bias" not in df.columns:
                inputs.append(1)
            inputs = np.asarray(inputs)
            net = np.dot(self.weights, inputs)
            if net > 0:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions


    def _initializeWeights(self, numFeatures, initialWeights=None):

        if initialWeights == None:
            self.weights = np.zeros(numFeatures)
            # self.weights = np.random.normal(0, 0.1, numFeatures)
        else:
            assert len(initialWeights) == (numFeatures)
            self.weights = np.asarray(initialWeights)


    def _getDeltaCoefficient(self, target, net):

        return self.learningRate * (target - net)


    def _train(self, df, features, targetName):

        for index in self.indices:
            row = df.iloc[index]
            inputs = np.asarray(list(row[features]))
            target = row[targetName]

            net = np.sum(np.multiply(self.weights, inputs))
            if net > 0:
                output = 1
            else:
                output = 0

            deltaCoefficient = self.learningRate * (target - output)
            for i in range(len(self.weights)):
                self.weights[i] += (deltaCoefficient * inputs[i])


    def _stoppingCriterion(self, scores):
        if len(scores) > 5:
            if abs(scores[-1] - scores[-2]) < 0.01:
                return False
            else:
                if len(scores) > 15:
                    if abs(np.mean(scores[-5:]) - np.mean(scores[-10:-5])) < 0.1: # in case we are in a bit of a cycle
                        return False
                return True



    def fit(self, df, features, target, initialWeights=None):
        # avoid issues of messing with the original data
        df = copy.copy(df)
        features = copy.copy(features)
        target = copy.copy(target)

        # make sure there isn't any funny business going on with user input
        cols = list(df.columns)
        for feature in features:
            assert feature in cols
        assert target in cols

        self.features = features

        # add the bias (must be done before initializing weights)
        df["bias"] = [1] * len(df[df.columns[0]])
        features.append("bias")

        # initialize the weights
        self._initializeWeights(len(features), initialWeights)

        # initialize the indices
        self.indices = np.asarray(list(range(len(df[list(df.columns)[0]]))))

        # print(len(self.indices))
        if self.numEpochs != None: # train with a set number of epochs
            for epoch in range(self.numEpochs):
                if self.shuffle:
                    np.random.shuffle(self.indices)
                self._train(df, features, target)

        else: # train till done
            keepGoing = True
            scores = []
            while keepGoing:
                if self.shuffle:
                    np.random.shuffle(self.indices)
                self._train(df, features, target)
                score = self.score(df, features, target)
                scores.append(score)
                keepGoing = self._stoppingCriterion(scores)

        # let the rest of the model know that the weights have been _trained
        self.fitted = True

    def score(self, df, features, target):
        df = copy.copy(df)
        features = copy.copy(features)
        target = copy.copy(target)

        yHat = self.predict(df, features)

        # simple metric of accuracy
        targets = list(df[target])
        targets = np.asarray([int(x) for x in targets])
        yHat = np.asarray(yHat)
        numCorrect = float(np.sum(yHat == targets))
        numTotal = float(len(yHat))

        return numCorrect / numTotal

    def get_weights(self):
        assert self.fitted == True
        return self.weights

