# -*- coding: utf-8 -*-

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.special import comb
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D

def toy_dataset(n1, n2):
    # class 1 data set
    rs = np.random.random(n1)
    r = 1.0
    X1 = np.c_[r * np.cos(2*np.pi*rs), r * np.sin(2*np.pi*rs)]
    Y1 = ['class1' for _ in range(n1)]

    # class 2 data set
    rs = np.random.random(n2)
    r = 2.0
    X2 = np.c_[r * np.cos(2*np.pi*rs), r * np.sin(2*np.pi*rs)]
    Y2 = ['class2' for _ in range(n2)]

    # concat
    X = np.r_[X1, X2]
    Y = np.r_[Y1, Y2]
    x_df = pd.DataFrame(data = X, columns = ['x', 'y'])
    y_series = pd.Series(Y, name = 'class')
    df = pd.concat([x_df, y_series], axis=1)
    return df

np.random.seed(0)

# train dataset
train_num = 100
df = toy_dataset(n1=train_num, n2=train_num)

# test dataset
test_num = 30
test_df = toy_dataset(n1=test_num, n2=test_num)

# train dataset scatter
c1_df = df[df['class'] == 'class1']
c2_df = df[df['class'] == 'class2']
sns.scatterplot(c1_df['x'], c1_df['y'], label='class1')
sns.scatterplot(c2_df['x'], c2_df['y'], label='class2')
plt.savefig('train_dataset.png')
plt.close()

ds = np.arange(1, 11)
accs = np.zeros(10)

print('even degree')
for d in range(2, 11, 2):
    clf = SVC(gamma = 'auto', kernel='poly', degree=d)
    clf.fit(df[['x', 'y']], df['class'])
    y_pred = clf.predict(test_df[['x', 'y']])
    acc = accuracy_score(test_df['class'], y_pred)
    print('degree =', d, 'acc:', acc)
    accs[d-1] = acc

print('odd degree')
for d in range(1, 10, 2):
    clf = SVC(gamma = 'auto', kernel='poly', degree=d)
    clf.fit(df[['x', 'y']], df['class'])
    y_pred = clf.predict(test_df[['x', 'y']])
    acc = accuracy_score(test_df['class'], y_pred)
    print('degree =', d, 'acc:', acc)
    accs[d-1] = acc

sns.lineplot(ds, accs)
plt.xlabel('degree')
plt.ylabel('Accuracy')
plt.savefig('acc.png')
plt.close()


def poly_feature(x, d):
    n = x.shape[0]
    Z = np.zeros((n, d+1))
    for i in range(d+1):
        a = np.sqrt(comb(d, i, exact=True))
        Z[:,i] = a * (x[:,0]**(d-i)) * (x[:,1]**(i))
    return Z

for d in range(1, 11):
    
    z = poly_feature(df[['x', 'y']].values, d)
    columns = [ 'feature' + str(i) for i in range(d+1)]
    feature_df = pd.DataFrame(data = z, columns = columns)
    feature_df = pd.concat([feature_df, df['class']], axis=1)
    g = sns.pairplot(feature_df, hue='class', vars=feature_df.columns[:-1])
    g.fig.suptitle('degree = ' + str(d))
    plt.savefig('pair_d' + str(d) + '.png')
    plt.close()

    # 3d plot
    '''
    for n, m, l in combinations(range(d+1), 3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        c1 = feature_df[feature_df['class'] == 'class1']
        ax.scatter3D(c1['feature' + str(n)], c1['feature' + str(m)], c1['feature' + str(l)], label='class1')
        c2 = feature_df[feature_df['class'] == 'class2']
        ax.scatter3D(c2['feature' + str(n)], c2['feature' + str(m)], c2['feature' + str(l)], label='class2')
        ax.set_xlabel('feature' + str(n))
        ax.set_ylabel('feature' + str(m))
        ax.set_zlabel('feature' + str(l))
        plt.legend()
        plt.show()
    '''