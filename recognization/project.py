import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from hmmlearn import hmm


def nb_train(train_data):
    # for index_i, row_i in train_data.iterrows():
    #     prior[table.get(row_i["letter"])] += 1
    #     for i in range(128):
    #         con_pro[table.get(row_i["letter"]), i, row_i.iloc[6 + i]] += 1
    prior = train_data.groupby(["letter"]).count()["id"] / len(train_data)
    con_pro = train_data.groupby(['letter']).sum().iloc[:, 5:134] / train_data.groupby(["letter"]).count().iloc[:,5:134]
    return prior, con_pro


def nb_accuracy(test_data, train_parameter):
    table = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
             "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24,
             "z": 25}
    predict_result = np.zeros(len(test_data))
    id = 0
    count = 0
    for index, row in test_data.iterrows():
        predict_result[id] = nb_predict(row, train_parameter)
        if predict_result[id] == table.get(row["letter"]):
            count += 1
        id +=1
    accuracy = count / len(test_data)
    return accuracy, predict_result


def nb_predict(data, train_parameter):
    prior, con_pro = train_parameter
    result_pro = np.zeros(26)
    for i in range(26):
        result_pro[i] = np.sum([np.log(con_pro.iloc[i, j]) if data.iloc[6 + j] == 1 else np.log(1 - con_pro.iloc[i, j]) for j in range(128)]) + np.log(prior[i])
    return np.argmax(result_pro)


def hmm_train(hmm_train_data):
    table = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
             "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24,
             "z": 25}
    trasfer_mat = np.zeros((26, 26))
    startprob = np.zeros(26)
    emissionprob = np.zeros((26, 26))
    for index, row in hmm_train_data.iterrows():
        startprob[table.get(row["letter"])] += 1
        emissionprob[table.get(row["letter"]), table.get(row["bayes"])] += 1
        if row["next_id"] == -1:
            continue
        next_letter = hmm_train_data.loc[row["next_id"] - 1, "letter"]
        trasfer_mat[table.get(row["letter"]), table.get(next_letter)] += 1

    trasfer_mat = trasfer_mat / np.sum(trasfer_mat, axis=1).reshape(-1, 1)
    emissionprob = emissionprob / np.sum(emissionprob, axis=1).reshape(-1, 1)
    startprob = startprob / np.sum(startprob)
    return trasfer_mat, emissionprob, startprob


def hmm_accuracy(hmm_test_data, hmm_train_result):
    table = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
             "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24,
             "z": 25}
    hmm_model = hmm.MultinomialHMM(n_components=26)

    hmm_model.transmat_, hmm_model.emissionprob_, hmm_model.startprob_ = hmm_train_result
    querySequence = []
    resultSet = []
    for index, row in hmm_test_data.iterrows():
        querySequence.append(table.get(hmm_test_data.loc[index, "letter"]))
        if (hmm_test_data.loc[index, "next_id"] == -1):
            resultSet.extend(hmm_model.predict(np.array(querySequence).reshape(-1, 1)))
            querySequence = []
    # print(resultSet)
    accuracy = 0
    table2 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
              "u",
              "v", 'w', "x", "y", "z"]
    expect_array = []
    hmm_test_data_ite = hmm_test_data.iterrows()
    for j in range(len(resultSet)):
        if table2[resultSet[j]] == hmm_test_data.loc[next(hmm_test_data_ite)[0], "letter"]:
            accuracy += 1
        expect_array.append(table2[resultSet[j]])
    accuracy /= len(resultSet)
    return accuracy


def main():
    header = pd.read_csv("letter.names", header=None)
    # print(header.values.reshape(1, -1)[0])
    data = pd.read_csv("letter.data", sep="\s+", names=header.values.reshape(1, -1)[0])

    '''
    10 cross fold for naive bayes 
    '''
    bayes_result = np.zeros(10)
    for i in range(10):
        train_data = data[data["fold"] != i]
        test_data = data[data["fold"] == i]
        train_data.index = range(len(train_data))
        test_data.index = range(len(test_data))
        clf = BernoulliNB()
        clf.fit(train_data.iloc[:, 6:134], train_data.iloc[:, 1])
        bayes_result[i] = clf.score(test_data.iloc[:, 6:134], test_data.iloc[:, 1])
        # accuracy, predic_result = nb_accuracy(test_data, nb_train(train_data))
        # print(accuracy)
        # print(predic_result)


    '''
    combine the naive bayes predictions to dataset
    '''
    beyess_predicts = clf.predict(data.iloc[:, 6:134])
    bayes = pd.DataFrame(beyess_predicts.reshape(-1, 1), columns=["bayes"])
    # print(bayes.values.shape)
    hmm_data = pd.concat([data, bayes], axis=1)
    table = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
             "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24,
             "z": 25}

    '''
    10 cross fold for HMM
    '''
    hmm_result = np.zeros(10)
    for i in range(10):
        hmm_train_data = hmm_data[hmm_data["fold"] != i]
        hmm_test_data = hmm_data[hmm_data["fold"] == i]
        hmm_result[i] = hmm_accuracy(hmm_test_data, hmm_train(hmm_train_data))

    print("The accuracy for bayes without adding HMM")
    print(bayes_result)
    print("The average accuracy: ")
    print(np.average(bayes_result))
    print("The accuracy for bayes after adding HMM")
    print(hmm_result)
    print("The average accuracy: ")
    print(np.average(hmm_result))


if __name__ == '__main__':
    main()
