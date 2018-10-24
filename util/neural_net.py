from csv import writer

from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA, NMF, PCA, TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.random_projection import GaussianRandomProjection

import loadData

import time

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def oneHotLabels(labels, i=10):
    encoding = []
    for label in labels:
        zeroVector = [0]*i
        zeroVector[int(label)] = 1
        encoding.append(zeroVector)

    return encoding

def writeToCsv(data, fileName="OptDigitsNN.csv"):
    with open(fileName, 'w+') as csvfile:
        testWriter = writer(csvfile)
        for row in data:
            testWriter.writerow(row)

def performExperiment():
    trExampleVectorOrig, trTargetVector = loadData.DigitsData("../data/optdigits.tra").getData()[0]
    trFormattedTargetsVector = oneHotLabels(trTargetVector)
    teExampleVectorOrig, teTargetVector = loadData.DigitsData("../data/optdigits.tes").getData()[0]
    teFormattedTargetsVector = oneHotLabels(teTargetVector)
    numTestInstances = len(teExampleVectorOrig)

    results = []
    for i in range(4, 64, 4):

        pca = NMF(n_components=i, random_state=0)
        trExampleVector = pca.fit_transform(trExampleVectorOrig)
        teExampleVector = pca.fit_transform(teExampleVectorOrig)

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1, hidden_layer_sizes=(20,20,20), verbose=True)
        fitStartTime = time.time()
        clf.fit(trExampleVector, trFormattedTargetsVector)
        fitElapsedTime = time.time() - fitStartTime

        prediction = clf.predict(teExampleVector)
        numCorrect = 0

        for ndx in range(len(teExampleVector)):
            if prediction[ndx][teTargetVector[ndx]] == 1:
                numCorrect += 1

        accuracy = float(numCorrect) / numTestInstances
        result = [i, accuracy, fitElapsedTime, fitElapsedTime / 60]
        results.append(result)

    writeToCsv(data=results)

def performKMeansClusterExperiment():
    trExampleVector, trTargetVector = loadData.DigitsData("../data/optdigits.tra").getData()[0]
    trFormattedTargetsVector = oneHotLabels(trTargetVector)
    teExampleVector, teTargetVector = loadData.DigitsData("../data/optdigits.tes").getData()[0]
    teFormattedTargetsVector = oneHotLabels(teTargetVector)
    numTestInstances = len(teExampleVector)


    # trFormattedTargetsVector = oneHotLabels(kmeansLabelsTr)
    # teFormattedTargetsVector = oneHotLabels(kmeansLabelsTe)

    # for i in range(len(trExampleVector)):
    #     trExampleVector[i] = kmeansLabelsTr[i]
    # for i in range(len(teExampleVector)):
    #     teExampleVector[i] = kmeansLabelsTe[i]

    results = [["Number of clusters", "Accuracy", "Elapsed time (seconds)", "Elapsed time (minutes)"]]
    for i in range(10, 25, 5):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(trExampleVector)
        kmeansLabelsTr = oneHotLabels(kmeans.predict(trExampleVector), i)
        kmeansLabelsTe = oneHotLabels(kmeans.predict(teExampleVector), i)

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1, hidden_layer_sizes=(20,20,20), verbose=True)
        fitStartTime = time.time()
        clf.fit(kmeansLabelsTr, trFormattedTargetsVector)
        fitElapsedTime = time.time() - fitStartTime

        prediction = clf.predict(kmeansLabelsTe)
        numCorrect = 0

        for ndx in range(len(kmeansLabelsTe)):
            if prediction[ndx][teTargetVector[ndx]] == 1:
                numCorrect += 1

        accuracy = float(numCorrect) / numTestInstances
        results.append([i, accuracy, fitElapsedTime, fitElapsedTime / 60])

    writeToCsv(fileName="KMeansDigitsNN.csv", data=results)

def performEMClusterExperiment():
    trExampleVector, trTargetVector = loadData.DigitsData("../data/optdigits.tra").getData()[0]
    trFormattedTargetsVector = oneHotLabels(trTargetVector)
    teExampleVector, teTargetVector = loadData.DigitsData("../data/optdigits.tes").getData()[0]
    teFormattedTargetsVector = oneHotLabels(teTargetVector)
    numTestInstances = len(teExampleVector)

    em = GaussianMixture(n_components=14, random_state=0).fit(trExampleVector)
    emLabelsTr = em.predict_proba(trExampleVector)
    emLabelsTe = em.predict_proba(teExampleVector)

    results = [["Number of clusters", "Accuracy", "Elapsed time (seconds)", "Elapsed time (minutes)"]]
    for i in range(5, 25, 5):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1, hidden_layer_sizes=(20,20,20), verbose=True)
        fitStartTime = time.time()
        clf.fit(emLabelsTr, trFormattedTargetsVector)
        fitElapsedTime = time.time() - fitStartTime

        prediction = clf.predict(emLabelsTe)
        numCorrect = 0

        for ndx in range(len(teExampleVector)):
            if prediction[ndx][teTargetVector[ndx]] == 1:
                numCorrect += 1

        accuracy = float(numCorrect) / numTestInstances
        results.append([accuracy, fitElapsedTime, fitElapsedTime / 60])

    writeToCsv(fileName="EMDigitsNN.csv", data=results)

def main():
    columnNames = ["Hidden layers configuration",
                   "Accuracy",
                   "Fit time in seconds", 
                   "Fit time in minutes"]

    with open('OptDigitsNN.csv', 'w') as csvfile:
        testWriter = writer(csvfile)
        testWriter.writerow(columnNames)

    # performExperiment()
    # performKMeansClusterExperiment()
    performEMClusterExperiment()

main()
# writeToCsv()