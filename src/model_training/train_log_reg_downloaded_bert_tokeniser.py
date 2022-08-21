import argparse
import os
import pickle
import random
import re
import sys
import time

sys.path.append("../src")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from train_log_reg import analyse, set_seed

from data_utils import HotelReviewsProcessor, ReviewsDataset
from parsing_utils import init_parser


def get_labelled_data(file):
    data = []
    label = []
    for line in file:
        line1 = line.split("\t")
        data.append(line1[1])
        label.append(int(line1[0]))
    return (np.array(data), np.array(label))


def main():
    args = init_parser().parse_args()
    set_seed(args.seed)
    file_path = "../datasets/hotel_reviews/additional_downloads/"
    train_file = open(file_path + "train.txt", "r")
    (train_data, train_label) = get_labelled_data(train_file)
    valid_file = open(file_path + "dev.txt", "r")
    (valid_data, valid_label) = get_labelled_data(valid_file)
    test_file = open(file_path + "test.txt", "r")
    (test_data, test_label) = get_labelled_data(test_file)
    vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"\S+")

    train_X = vectorizer.fit_transform(train_data)
    test_X = vectorizer.transform(test_data)

    # Grid search
    grid = {
        "C": np.logspace(-3, 3, 7),
        "penalty": ["l1", "l2"],
        "random_state": [0],
        "class_weight": ["balanced"],
        "solver": ["liblinear"],
    }  # l1 lasso l2 ridge
    logreg = LogisticRegression()
    clf = GridSearchCV(logreg, grid, cv=10)
    best_clf = clf.fit(train_X, train_label)

    # Accuracy
    print("Train_acc: %.2f" % best_clf.score(train_X, train_label))
    test_pred = best_clf.predict(test_X)
    p, r, f1, _ = precision_recall_fscore_support(
        test_label, test_pred, average="binary"
    )
    accuracy = accuracy_score(test_label, test_pred)
    print("Test Performance")
    print(
        "Acc: %.2f | P: %.2f | R: %.2f | F1: %.2f"
        % (100.0 * accuracy, 100.0 * p, 100.0 * r, 100.0 * f1)
    )
    print("--------------------------------------------------------------")
    best_clf_model = best_clf.best_estimator_
    if args.save is not None:
        pickle.dump([vectorizer, best_clf_model], open(args.save, "wb"))
    if args.load is not None:
        [vectorizer, best_clf_model] = pickle.load(open(args.load, "rb"))
    analyse(vectorizer, best_clf_model, args.top_k)


if __name__ == "__main__":
    main()
