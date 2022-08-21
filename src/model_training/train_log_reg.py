import argparse
import os
import pickle
import random
import sys
import time

sys.path.append("../src")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

from data_utils import HotelReviewsProcessor, ReviewsDataset
from parsing_utils import init_parser


def set_seed(seed):
    # set seed for all possible avenues of stochasticity
    np.random.seed(seed=seed)
    random.seed(seed)


def analyse(vectorizer, best_clf, top_k=20):
    feat_name = np.array(vectorizer.get_feature_names())
    weights = best_clf.coef_.reshape(-1)
    print("True words")  # get k highest scoring words
    for index in weights.argsort()[-top_k:][::-1]:
        print(str(feat_name[index]) + " : " + str(round(weights[index], 4)))
    print("Fake words")  # get k lowest scoring words
    for index in weights.argsort()[:top_k]:
        print(str(feat_name[index]) + " : " + str(round(weights[index], 4)))


def main():
    args = init_parser().parse_args()
    set_seed(args.seed)

    # load data
    dataset_processor = HotelReviewsProcessor()
    train_examples = dataset_processor.get_train_examples(args.data_dir)
    train_examples = train_examples[: args.num_train_examples]
    dev_examples = dataset_processor.get_dev_examples(args.data_dir)
    test_examples = dataset_processor.get_test_examples(args.data_dir)

    train_data = []
    train_label = []
    for example in train_examples:
        train_data.append(example.content)
        train_label.append(example.truthfulness)
    valid_data = []
    valid_label = []
    for example in dev_examples:
        valid_data.append(example.content)
        valid_label.append(example.truthfulness)
    test_data = []
    test_label = []
    for example in test_examples:
        test_data.append(example.content)
        test_label.append(example.truthfulness)

    # compute tfidf features
    vectorizer = TfidfVectorizer()
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

    # Print top k words
    analyse(vectorizer, best_clf_model, args.top_k)


if __name__ == "__main__":
    main()
