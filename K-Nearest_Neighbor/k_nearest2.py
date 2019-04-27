import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import random

style.use('fivethirtyeight')

# euclidean_distance = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color=i)

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups.')
    distances = []

    for grp in data:
        for features in data[grp]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, grp])

    votes = [i[1] for i in sorted(distances)[:k]]
    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][0] / k

    return vote_result, confidence


accuracies = []
for i in range(25):
    df = pd.read_csv('./DataSet/breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()

    random.shuffle(full_data)

    test_size = 0.4
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    # first 20% of data for training
    train_data = full_data[:-int(test_size*len(full_data))]
    # last 20% of data for testing
    test_data = full_data[-int(test_size*len(full_data)):]

    # for i in train_data:
    #     train_set[i[-1]].append(i[:-1])
    [train_set[i[-1]].append(i[:-1]) for i in train_data]

    # for i in test_data:
    #     test_set[i[-1]].append(i[:-1])
    [test_set[i[-1]].append(i[:-1]) for i in test_data]

    correct = 0
    total = 0

    for grp in test_set:
        for data in test_set[grp]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if grp == vote:
                correct += 1
            total += 1

    # print(f'Accuracy: {correct/total}')
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))
