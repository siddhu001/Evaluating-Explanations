import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# setting path
sys.path.append("../src")

from user_study_id_files import (played_array_global,
                                 played_array_global_ablation, played_array_IG,
                                 played_array_lime, played_array_none)
from utils import compute_bootstrapped_confidence_interval, find_time_diff


def convert_to_one_hot_vector(a):
    a = np.array(a)
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b.tolist()


def check_cohort(user_database, cohort_dict):
    test_examples = [example.text for example in user_database.test_examples_arr]
    for cohort_id in cohort_dict:
        cohort_text_arr = cohort_dict[cohort_id]
        if np.array_equal(cohort_text_arr, test_examples):
            return cohort_id
    cohort_id = len(cohort_dict)
    cohort_dict[cohort_id] = test_examples
    return cohort_id


def change_in_confidence(example_arr):
    max_change_score_val_arr = []
    for example in example_arr:
        if len(example.log_array) < 2:
            continue
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
        max_change_score_val_arr.append(max_change_score_val)
    return np.mean(max_change_score_val_arr)


def create_user_study_arr(
    played_array,
    explanation_type,
    correct_guess_train=[],
    correct_guess_test=[],
    correct_guess=[],
    confidence_change_train=[],
    confidence_change_test=[],
    eg_flip_train=[],
    eg_flip_test=[],
    exp_type_arr=[],
    user_arr=[],
    user_dict={},
    data_cohort_arr=[],
    data_cohort_dict={},
):
    for k in played_array:
        user_name = k
        database = pickle.load(open("save_database_files/database_" + k + ".pkl", "rb"))
        train_count = 0
        for j in database.train_examples_arr:
            if len(j.log_array) > 1:
                train_count += 1
        test_count = 0
        for j in database.test_examples_arr:
            if len(j.log_array) > 1:
                test_count += 1
        correct_guess_train.append(database.correct_guesses / train_count)
        correct_guess_test.append(database.test_correct_guesses / test_count)
        correct_guess.append(
            (database.correct_guesses + database.test_correct_guesses)
            / (train_count + test_count)
        )
        eg_flip_train.append(len(database.examples_flipped_dict) / train_count)
        eg_flip_test.append(len(database.test_examples_flipped_dict) / test_count)
        confidence_change_train.append(
            change_in_confidence(database.train_examples_arr)
        )
        confidence_change_test.append(change_in_confidence(database.test_examples_arr))
        exp_type_arr.append(explanation_type)
        if k not in user_dict:
            user_dict[k] = len(user_dict)
        user_arr.append(user_dict[k])
        data_cohort_arr.append(check_cohort(database, data_cohort_dict))


def prepare_data():
    index_dict = {}
    correct_guess_train = []
    correct_guess_test = []
    correct_guess = []
    confidence_change_train = []
    confidence_change_test = []
    eg_flip_train = []
    eg_flip_test = []
    exp_type_arr = []
    user_arr = []
    user_dict = {}
    data_cohort_arr = []
    data_cohort_dict = {}
    played_arr_dict = {}
    played_arr_dict["lime"] = played_array_lime
    played_arr_dict["IG"] = played_array_IG
    played_arr_dict["global"] = played_array_global
    played_arr_dict["global_ablation"] = played_array_global_ablation
    played_arr_dict["none"] = played_array_none
    for k in played_arr_dict:
        create_user_study_arr(
            played_arr_dict[k],
            k,
            correct_guess_train,
            correct_guess_test,
            correct_guess,
            confidence_change_train,
            confidence_change_test,
            eg_flip_train,
            eg_flip_test,
            exp_type_arr,
            user_arr,
            user_dict,
            data_cohort_arr,
            data_cohort_dict,
        )

    lst = list(
        zip(
            confidence_change_train,
            confidence_change_test,
            correct_guess_train,
            correct_guess_test,
            correct_guess,
            eg_flip_train,
            eg_flip_test,
            [int(exp_type == "lime") for exp_type in exp_type_arr],
            [int(exp_type == "IG") for exp_type in exp_type_arr],
            [int(exp_type == "global") for exp_type in exp_type_arr],
            [int(exp_type == "global_ablation") for exp_type in exp_type_arr],
            user_arr,
            data_cohort_arr,
        )
    )
    column_names = [
        "Train_Confidence_Change",
        "Test_Confidence_Change",
        "Train_Guess",
        "Test_Guess",
        "Guess_All",
        "Train_Flips",
        "Test_Flips",
        "Explanation_Type_LIME",
        "Explanation_Type_IG",
        "Explanation_Type_global",
        "Explanation_Type_global_ablation",
        "User",
        "Cohort",
    ]
    data_cohort_arr = convert_to_one_hot_vector(data_cohort_arr)
    for lst_id in range(len(lst)):
        lst[lst_id] = list(lst[lst_id]) + data_cohort_arr[lst_id]
    column_names += list(data_cohort_dict.keys())
    df = pd.DataFrame(lst, columns=column_names)
    return (df, list(data_cohort_dict))


def run_mixed_effects_model(data, cohorts, explanation_arr):
    data["Intercept"] = 1
    for col in [
        "Guess_All",
        "Train_Flips",
        "Test_Flips",
        "Train_Confidence_Change",
        "Test_Confidence_Change",
    ]:
        md = sm.MixedLM(
            data[col],
            data[
                [
                    "Intercept",
                ]
                + explanation_arr
            ],
            groups=data["Cohort"],
        )
        mdf = md.fit(method=["lbfgs"])
        print(mdf.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    (df, cohorts) = prepare_data()
    explanation_arr = [
        "Explanation_Type_LIME",
        "Explanation_Type_IG",
        "Explanation_Type_global",
        "Explanation_Type_global_ablation",
    ]
    run_mixed_effects_model(df, cohorts, explanation_arr)
