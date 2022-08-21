import argparse
import pickle
import sys
import time

sys.path.append("../src")

import numpy as np

from user_study_id_files import (played_array_global,
                                 played_array_global_ablation, played_array_IG,
                                 played_array_lime, played_array_logreg,
                                 played_array_logreg_none, played_array_none)
from utils import compute_bootstrapped_confidence_interval, find_time_diff


def compute_flipped_examples(exp_type="lime"):
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
    examples_flipped_train = []
    examples_flipped_test = []
    for k in played_array:
        user_name = k
        database = pickle.load(open("save_database_files/database_" + k + ".pkl", "rb"))
        train_count = 0
        for j in database.train_examples_arr:
            if len(j.log_array) > 1:
                train_count += 1
        examples_flipped_train.append(len(database.examples_flipped_dict) / train_count)
        test_count = 0
        for j in database.test_examples_arr:
            if len(j.log_array) > 1:
                test_count += 1
        examples_flipped_test.append(
            len(database.test_examples_flipped_dict) / test_count
        )

    print("Train")
    print(round(np.mean(examples_flipped_train) * 100, 1))
    compute_bootstrapped_confidence_interval(np.array(examples_flipped_train) * 100)

    print("Test")
    print(round(np.mean(examples_flipped_test) * 100, 1))
    compute_bootstrapped_confidence_interval(np.array(examples_flipped_test) * 100)


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
    compute_flipped_examples(args.exp_type)
