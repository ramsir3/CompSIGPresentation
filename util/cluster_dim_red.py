import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import homogeneity_score, silhouette_score, silhouette_samples
from sklearn.decomposition import FastICA, PCA, NMF
from sklearn.random_projection import GaussianRandomProjection

from collections import Counter
import time, csv

import load_data

RANDOM_STATE = 0

def timerDecorator(function):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = function(*args, **kwargs)
        t1 = time.time()
        print("Function \"%s\" took %f s" % (function.__name__, (t1 - t0)))
        return out
    return wrapper

def countClusters(kmeans_labels, labels):
    counts = dict()
    for a,b in zip(kmeans_labels, labels):
        a = str(a)
        if a in counts:
            counts[a].update([str(b)])
        else:
            counts[a] = Counter({str(b):1})
    return counts

def varodist_kmeans(data, centroids, labels):
    mse = 0
    num_el = data.shape[0]*data.shape[1]
    for i in range(data.shape[0]):
        mse += np.sum((data[i] - centroids[labels[i]])**2)
    mse /= num_el

    dist = 0
    for i in range(centroids.shape[0]):
        for j in range(centroids.shape[0]):
            dist += np.sum((centroids[i] - centroids[j])**2)**0.5
    dist /= (centroids.shape[0]*centroids.shape[0])

    return (mse, dist)

def varodist_em(data, centroids, weights):
    mse = 0
    num_el = data.shape[0]*data.shape[1]
    for i in range(data.shape[0]):
        for j in range(centroids.shape[0]):
            mse += np.sum((data[i] - centroids[j])**2) * weights[i, j]
    mse /= num_el

    dist = 0
    for i in range(centroids.shape[0]):
        for j in range(centroids.shape[0]):
            dist += np.sum((centroids[i] - centroids[j])**2)**0.5
    dist /= (centroids.shape[0]*centroids.shape[0])

    return (mse, dist)

@timerDecorator
def runKMeansRoutine(data, labels, filename=None, max_clusters=101):
    with open(filename+".csv" if filename else "KMeans_Metrics.csv", 'w') as f:
        w = csv.writer(f)
        # w.writerow(("num_clusters", "HS", "cluster_var", "cluster_dist", "varodist", "SS"))
        w.writerow(("num_clusters", "HS", "SS"))
        for nc in range(2, max_clusters):
            kmeans = KMeans(n_clusters=nc, random_state=RANDOM_STATE).fit(data)
            kmeans_labels = kmeans.predict(data)
            # cv, cd = varodist_kmeans(data, kmeans.cluster_centers_, kmeans_labels)
            # vod = cv/cd
            hs = homogeneity_score(labels, kmeans_labels)
            ss = silhouette_score(data, kmeans_labels)
            # w.writerow((nc, hs, cv, cd, vod, ss))
            w.writerow((nc, hs, ss))

@timerDecorator
def runEMRoutine(data, labels, filename=None, max_clusters=101):
    with open(filename+".csv" if filename else "EM_Metrics.csv", 'w') as f:
        w = csv.writer(f)
        # w.writerow(("num_clusters", "HS", "cluster_var", "cluster_dist", "varodist", "LL"))
        # w.writerow(("num_clusters", "HS", "LL"))
        w.writerow(("num_clusters", "HS", "SS"))

        for nc in range(2, max_clusters):
            em = GaussianMixture(n_components=nc, random_state=RANDOM_STATE).fit(data)
            em_weights = em.predict_proba(data)
            em_labels = em.predict(data)
            # cv, cd = varodist_em(data, em.means_, em_weights)
            # vod = cv/cd
            hs = homogeneity_score(labels, em_labels)
            ll = em.score(data)
            ss = silhouette_score(data, em_labels)
            # w.writerow((nc, hs, cv, cd, vod, ll))
            # w.writerow((nc, hs, ll))
            w.writerow((nc, hs, ss))

def plotReconstructionError(results, filename=None, save_csv=True):
    N = len(results) + 1
    ind = np.arange(1, N)
    width = 0.5
    fig, ax = plt.subplots()
    thr1 = 0.1
    thr2 = 0.25

    # resultsMax = max(results)
    # results = [n / resultsMax for n in results]

    rects = ax.bar(ind, height=results, width=width, color='b')
    ax.set_title(filename)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Average Reconstruction Error")
    ax.set_xticks(range(0, N, 4))
    ax.plot((0, N), (thr1, thr1), 'r-')
    ax.plot((0, N), (thr2, thr2), 'm-')
    ax.grid()
    if filename is not None:
        plt.savefig(filename+"")
    else:
        plt.show()
    if save_csv:
        with open(filename+".csv", 'w') as f:
            w = csv.writer(f)
            w.writerow(("included", "average error"))
            for e in zip(range(1, N), results):
                w.writerow(e)

@timerDecorator
def runPCARoutine(data, name):
    dn = data.shape[0]*data.shape[1]
    pca = PCA(n_components=data.shape[1], random_state=RANDOM_STATE)
    pca.fit(data)

    results = []
    for i in range(data.shape[1]):
        curVariances = pca.components_[:i+1]
        transform = np.dot(data - pca.mean_, curVariances.T)
        transformInverse = np.dot(transform, curVariances) + pca.mean_
        mse = np.sum(np.abs(data - transformInverse)) / dn
        results.append(mse)

    plotReconstructionError(results, filename="PCA_"+name)

@timerDecorator
def runICARoutine(data, name):
    dn = data.shape[0]*data.shape[1]

    results = []
    for i in range(data.shape[1]):
        ica = FastICA(n_components=i+1, random_state=RANDOM_STATE, max_iter=1000)
        transform = ica.fit_transform(data)
        transformInverse = ica.inverse_transform(transform)
        mse = np.sum(np.abs(data - transformInverse)) / dn
        # print(i, mse)
        results.append(mse)

    plotReconstructionError(results, filename="ICA_"+name)

@timerDecorator
def runRPRoutine(data, name):
    dn = data.shape[0]*data.shape[1]

    results = []
    for i in range(data.shape[1]):
        rp = GaussianRandomProjection(n_components=i+1, random_state=RANDOM_STATE)
        rp.fit_transform(data)
        curVariances = rp.components_
        transform = np.dot(data, curVariances.T)
        transformInverse = np.dot(transform, curVariances)
        mse = np.sum(np.abs(data - transformInverse)) / dn
        results.append(mse)

    plotReconstructionError(results, filename="RPR_"+name)

@timerDecorator
def runNMFoutine(data, name):
    dn = data.shape[0]*data.shape[1]

    results = []
    for i in range(data.shape[1]):
        pg = NMF(n_components=i+1, random_state=RANDOM_STATE)
        transform = pg.fit_transform(data)
        transformInverse = pg.inverse_transform(transform)
        mse = np.sum(np.abs(data - transformInverse)) / dn
        # print(i, mse)
        results.append(mse)

    plotReconstructionError(results, filename="NMF_"+name)


digits = load_data.DigitsData("data/digits/optdigits.tra").getData()[0]
adult = np.genfromtxt("data/adult/ohe_adult.csv", skip_footer=29000, delimiter=",", dtype=int)
adult = (adult[:3000, :-1], adult[:3000, -1])

# adult = loadData.AdultData("adult/adult.data").getData(split=0.01)[0]
datas = [(adult, "adult"), (digits, "digits")]
for d in [(digits, "digits")]:
    data, labels = d[0]
    print(labels.shape)
    name = d[1]

    runKMeansRoutine(data, labels, filename=name+"_KMeans_metrics")
    runEMRoutine(data, labels, filename=name+"_EM_metrics")

    # runPCARoutine(data, name)
    # runICARoutine(data, name)
    # runRPRoutine(data, name)
    # runNMFoutine(data, name)
