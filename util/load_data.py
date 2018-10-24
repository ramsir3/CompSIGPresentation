import numpy as np
from sklearn.preprocessing import OneHotEncoder

class DigitsData(object):
    """docstring for DigitsData"""
    def __init__(self, filename):
        super(DigitsData, self).__init__()
        self.filename = filename
        # self.example = self.load(filename)
        # print(self.example)

    def load(self, filename):
        raw = np.genfromtxt(filename, delimiter=",")
        # labels = np.copy(raw[:, -1]).reshape(-1, 1)
        # raw = np.copy(raw[:, :-1])
        # ohe = OneHotEncoder(sparse=False)
        # labels = ohe.fit_transform(labels)
        # raw = np.concatenate((raw, labels), axis=1)
        return raw


    def getData(self, split=1, random=False):
        example = self.load(self.filename)

        if split <= 0 or split > 1:
            raise ValueError("invalid split value")

        if split is not 1:
            idx = range(example.shape[0])
            if random:
                idx = np.random.permutation(example.shape[0])

            idx1 = idx[int(example.shape[0]*split):]
            idx2 = idx[:int(example.shape[0]*split)]

            labels1 = example[idx1, -1]
            data1 = example[idx1, :-1]
            labels2 = example[idx2, -1]
            data2 = example[idx2, :-1]

            return [(data1, labels1), (data2, labels2)]
        else:
            labels = example[:, -1]
            data = example[:, :-1]
            return [(data, labels)]

class AdultData(DigitsData):
    """docstring for AdultData"""
    def __init__(self, filename):
        self.mask = np.array([False, True, False, True, False, True, True, True, True, True, False, False, False, True, True])
        # self.mask2 = np.array([True, False, True, False, True, False, False, False, False, False, True, True, True, False, False])
        super(AdultData, self).__init__(filename)

    def load(self, filename):
        raw = np.genfromtxt(filename, dtype=None, delimiter=",", autostrip=True)
        # print(raw)
        raw = self.label2numeric(raw)
        # self.raw = raw
        labels = np.copy(raw[:, -1]).reshape(-1, 1)
        raw = np.copy(raw[:, :-1])

        ohe = OneHotEncoder(categorical_features=self.mask[:-1], sparse=False)
        out = ohe.fit_transform(raw)
        # ohe = OneHotEncoder(sparse=False)
        # labels = ohe.fit_transform(labels)
        out = np.concatenate((out, labels), axis=1)

        np.savetxt("ohe_adult.csv", out, delimiter=",")
        # print("load")
        return out

    def label2numeric(self, raw):
        numEx = raw.shape[0]
        cats = dict()
        for i in range(numEx):
            for j in range(self.mask.shape[0]):
                if self.mask[j]:
                    if j in cats.keys():
                        cats[j].add(raw[i][j])
                    else:
                        cats[j] = set()
                        cats[j].add(raw[i][j])
                        # print(cats)
        for key in cats.keys():
            cats[key] = sorted(list(cats[key]))
            # print(key, len(cats[key]))
        # print(cats)

        out = list()
        for i in range(numEx):
            out.append(list(raw[i]))
            for j in range(self.mask.shape[0]):
                if self.mask[j]:
                    out[i][j] = cats[j].index(out[i][j])

        return np.array(out, dtype=int)

if __name__ == "__main__":
    a = AdultData("adult/adult.data").getData()