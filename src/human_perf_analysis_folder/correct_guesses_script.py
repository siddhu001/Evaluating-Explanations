import argparse
import pickle
import sys
import time

sys.path.append("../src")

import numpy as np
from sklearn.metrics import classification_report

from string_names import *
from user_study_id_files import (played_array_global,
                                 played_array_global_ablation, played_array_IG,
                                 played_array_lime, played_array_logreg,
                                 played_array_logreg_none, played_array_none)
from utils import compute_bootstrapped_confidence_interval, find_time_diff


def compute_correct_guesses(exp_type="lime"):
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
    time_dict = {}
    correct_guess_train = []
    correct_guess_test = []
    correct_guess = []
    predicted_arr = []
    actual_arr = []
    for k in played_array:
        user_name = k
        database = pickle.load(open("save_database_files/database_" + k + ".pkl", "rb"))
        train_count = 0
        for j in database.train_examples_arr:
            if len(j.log_array) > 0:
                train_count += 1
        correct_guess_train.append(database.correct_guesses / train_count)

        test_count = 0
        for j in database.test_examples_arr:
            if len(j.log_array) > 0:
                test_count += 1
            else:
                continue
            if j.log_array[0].prob > 0.5:
                actual_arr.append(1)
                if j.user_guesses[0] == genuine_correct_str:
                    predicted_arr.append(1)
                else:
                    predicted_arr.append(0)
            else:
                actual_arr.append(0)
                if j.user_guesses[0] == fake_correct_str:
                    predicted_arr.append(0)
                else:
                    predicted_arr.append(1)
        correct_guess_test.append(database.test_correct_guesses / test_count)
        correct_guess.append(
            (database.correct_guesses + database.test_correct_guesses)
            / (train_count + test_count)
        )

    print("All")
    print(round(np.mean(correct_guess) * 100, 1))
    compute_bootstrapped_confidence_interval(correct_guess)


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
    compute_correct_guesses(args.exp_type)
