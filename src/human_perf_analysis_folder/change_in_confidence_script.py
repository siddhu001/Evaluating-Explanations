import argparse
import pickle
import sys
import time

sys.path.append("../src")
import numpy as np
from nltk.metrics import edit_distance

from analysis_utils import Utils
from user_study_id_files import (played_array_global,
                                 played_array_global_ablation, played_array_IG,
                                 played_array_lime, played_array_logreg,
                                 played_array_logreg_none, played_array_none)
from utils import compute_bootstrapped_confidence_interval, find_time_diff


def change_in_confidence(example):
    max_change_score_val = 0
    current_logs = example.log_array
    max_change_score_str = current_logs[0].output_text_no_tags
    for log_val in current_logs:
        prob = log_val.prob
        if example.original_score > 0.5:
            if prob < example.original_score:
                if (example.original_score - prob) > max_change_score_val:
                    max_change_score_val = example.original_score - prob
                    max_change_score_str = log_val.output_text_no_tags
        else:
            if example.original_score < prob:
                if (prob - example.original_score) > max_change_score_val:
                    max_change_score_val = prob - example.original_score
                    max_change_score_str = log_val.output_text_no_tags
    return max_change_score_val


def compute_avg_confidence_change(exp_type="lime"):
    if exp_type == "lime":
        played_array = played_array_lime
    elif exp_type == "log_reg":
        played_array = played_array_logreg
    elif exp_type == "log_reg_control":
        played_array = played_array_logreg_none
    elif exp_type == "integrated_gradients":
        played_array = played_array_IG
    elif exp_type == "global_exp":
        played_array = played_array_global
    elif exp_type == "global_exp_ablation":
        played_array = played_array_global_ablation
    else:
        played_array = played_array_none

    index_dict = {}
    train_arr = []
    test_arr = []

    for k in played_array:
        user_array = []
        user_train_arr = []
        user_name = k
        database = pickle.load(open("save_database_files/database_" + k + ".pkl", "rb"))
        for example in database.train_examples_arr:
            if len(example.log_array) < 2:
                continue
            max_change_conf = change_in_confidence(example)
            train_arr.append(max_change_conf)
            user_train_arr.append(max_change_conf)
        for example in database.test_examples_arr:
            if len(example.log_array) < 2:
                continue
            max_change_conf = change_in_confidence(example)
            user_array.append(max_change_conf)
            test_arr.append(max_change_conf)

    print("Training -")
    print("Max change in confidence -")
    print(round(np.mean(train_arr) * 100, 1))
    compute_bootstrapped_confidence_interval(np.array(train_arr) * 100)

    print("Testing -")
    print("Max change in confidence -")
    print(round(np.mean(test_arr) * 100, 1))
    compute_bootstrapped_confidence_interval(np.array(test_arr) * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_type",
        required=True,
        choices=[
            "lime",
            "log_reg",
            "log_reg_control",
            "integrated_gradients",
            "none",
            "global_exp",
            "global_exp_ablation",
        ],
        help="type of explanation to display",
    )
    args = parser.parse_args()
    compute_avg_confidence_change(args.exp_type)
