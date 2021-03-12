import streamlit as st
import numpy as np
from itertools import product

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from matplotlib import colors
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.metrics import accuracy_score
from scipy.special import logit

st.title('Explore different classifier and datasets')


dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Linear', 'Non-Linear', 'Moon', 'XOR')
)

scale = st.sidebar.selectbox(
    'Standardize',
    ('Yes', 'No')
)

model_type = st.sidebar.selectbox(
    'model',
    ('Logistics Regression', 'Decision Tree',
     'Random Forest', 'Ensemble')
)

if model_type == 'Logistics Regression':
    poly_feats = st.sidebar.slider(
        'polynomial feature level',
        1, 20
    )
    reg = st.sidebar.selectbox(
        'Regularization',
        ('0.0001', '0.001', '0.01', '0.1', '1', '10', '100', '1000')
    )

if model_type == 'Decision Tree':
    depth = st.sidebar.slider(
        'Max depth',
        1, 20
    )

if model_type == 'Random Forest':
    depth = st.sidebar.slider(
        'Max depth',
        1, 20
    )
    tree_num = st.sidebar.slider(
        'Number of trees',
        min_value=1, max_value=500, step=20
    )

train_rate = st.sidebar.slider(
    'training set ratio',
    min_value=0.1, max_value=0.9, step=0.1
)

@st.cache
def get_dataset(name):
    data = None
    if name == 'Linear':
        data = np.loadtxt('data/ml-ex2/ex2data1.txt', delimiter=',')
        X, y = np.hsplit(data, np.array([2]))
        y = y.ravel()
    elif name == 'Non-Linear':
        data = np.loadtxt('data/ml-ex2/ex2data2.txt', delimiter=',')
        X, y = np.hsplit(data, np.array([2]))
        y = y.ravel()
    elif name == 'Moon':
        X, y = make_moons(n_samples=1000, noise=0.3)
    elif name == 'XOR':
        n_samples = 1000
        n_bins = 4  # use 4 bins as we have 4 clusters here

        # Generate 4 blobs with 2 classes where the second blob contains
        # half positive samples and half negative samples. Probability in this
        # blob is therefore 0.5.
        centers = [(-3, -3), (3, 3), (-3, 3), (3, -3)]
        X, y = make_blobs(n_samples=n_samples, cluster_std=1.5, centers=centers, shuffle=False,
                          random_state=42)

        y[:n_samples // 2] = 0
        y[n_samples // 2:] = 1
    return X, y


X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))
st.write('number of positive:', sum(y))

if scale == 'Yes':
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_rate, random_state=1234)


def LR_pipe_line(X, y, poly_feats, reg):
    reg = float(reg)
    poly = PolynomialFeatures(poly_feats, include_bias=False)

    # Logistic regression classifier.
    # - C=1.0 will result in good fit
    # - C=1e4 will result in overfit (to little regularization)
    # - C=1e-2 will result in underfit (to much regularization)
    clf = LogisticRegression(C=reg, solver='lbfgs', max_iter=10000)
    model = Pipeline([  # ('poly', poly),
        ('poly', poly),
        ('clf', clf)])
    # Fit data to model
    model.fit(X, y)
    return model


def Tree_pipe_line(X, y, depth):
    clf = DecisionTreeClassifier(max_depth=depth)
    # Fit data to model
    clf.fit(X, y)
    return clf


def RF_pipe_line(X, y, depth, count):
    clf = RandomForestClassifier(max_depth=depth, n_estimators=count)
    # Fit data to model
    clf.fit(X, y)
    return clf


def EN_pipe_line(X, y):
    clf_rf = RandomForestClassifier(max_depth=6, n_estimators=200)
    clf_tree = DecisionTreeClassifier(max_depth=4)
    # clf_lr = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
    clf_lr = LR_pipe_line(X, y, 3, 1)
    clf_knn = KNeighborsClassifier(n_neighbors=5)
    # Fit data to model
    clf_rf.fit(X, y)
    clf_tree.fit(X, y)
    # clf_lr.fit(X, y)
    clf_knn.fit(X, y)
    return [clf_rf, clf_tree, clf_lr, clf_knn]


if model_type == 'Logistics Regression':
    model = LR_pipe_line(X_train, y_train, poly_feats, reg)
elif model_type == 'Decision Tree':
    model = Tree_pipe_line(X_train, y_train, depth)
elif model_type == 'Random Forest':
    model = RF_pipe_line(X_train, y_train, depth, tree_num)
elif model_type == 'Ensemble':
    models = EN_pipe_line(X_train, y_train)


# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(14, 7))

if model_type != 'Ensemble':
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    score_train = np.round(model.score(X_train, y_train), 3)
    score_test = np.round(model.score(X_test, y_test), 3)
else:
    buf = np.zeros(len(xx.ravel()))
    preds_train = np.zeros(len(y_train))
    preds_test = np.zeros(len(y_test))
    for model in models:
        buf += model.predict(np.c_[xx.ravel(), yy.ravel()])
        preds_train += model.predict(X_train)
        preds_test += model.predict(X_test)
    Z = np.rint(buf.reshape(xx.shape) / 4)
    preds_train = np.rint(preds_train / 4)
    preds_test = np.rint(preds_test / 4)
    score_train = np.round(accuracy_score(y_train, preds_train), 3)
    score_test = np.round(accuracy_score(y_test, preds_test), 3)


axarr[0].contourf(xx, yy, Z, alpha=0.4)
axarr[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
axarr[0].set_title('Training accuracy:{}'.format(score_train))
axarr[0].axis('equal')

axarr[1].contourf(xx, yy, Z, alpha=0.4)
axarr[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, edgecolor='k')
axarr[1].set_title('Test accuracy:{}'.format(score_test))
axarr[1].axis('equal')
plt.show()
st.pyplot(f)