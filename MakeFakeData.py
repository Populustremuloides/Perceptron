import numpy as np
import pandas as pd


def makeData(centroids, categories, numRepeats):
    np.random.seed(0)

    assert len(centroids) == len(categories)

    dataDict = {}
    numDimensions = len(centroids[0])
    for dimension in range(numDimensions):
        dataDict["X" + str(dimension)] = []
    dataDict["Y"] = []

    for i in range(len(centroids)):

        centroid = np.asarray(centroids[i])
        category = categories[i]

        for repeat in range(numRepeats):
            noise = np.random.normal(0, 0.1, numDimensions)
            newDataPoint = np.add(centroid, noise)

            for j in range(len(newDataPoint)): # add each dimension to a column

                # safeguard against really extreme values
                if newDataPoint[j] > 1:
                    newDataPoint[j] == 1
                if newDataPoint[j] < -1:
                    newDataPoint[j] == -1

                dataDict["X" + str(j)].append(newDataPoint[j])
            dataDict["Y"].append(category) # add the category
    df = pd.DataFrame.from_dict(dataDict)
    return df


def returnPoint(m, b, x):
    return (m * x) + b

def getLineSample(coefficients):
    m = -coefficients[0] / coefficients[1]
    b = -coefficients[2] / coefficients[1]

    xs = [-1, 0, 1]
    ys = []
    for x in xs:
        ys.append(returnPoint(m, b, x))
    return xs, ys



