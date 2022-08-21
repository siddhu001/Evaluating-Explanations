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

from run_mixed_effects_model import (change_in_confidence, check_cohort,
                                     convert_to_one_hot_vector,
                                     create_user_study_arr,
                                     run_mixed_effects_model)

from user_study_id_files import played_array_logreg, played_array_logreg_none
from utils import compute_bootstrapped_confidence_interval, find_time_diff


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
    played_arr_dict["logreg"] = played_array_logreg
    played_arr_dict["none"] = played_array_logreg_none
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
            [int(exp_type == "logreg") for exp_type in exp_type_arr],
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
        "Explanation_Type_Logreg",
        "User",
        "Cohort",
    ]
    data_cohort_arr = convert_to_one_hot_vector(data_cohort_arr)
    for lst_id in range(len(lst)):
        lst[lst_id] = list(lst[lst_id]) + data_cohort_arr[lst_id]
    column_names += list(data_cohort_dict.keys())
    df = pd.DataFrame(lst, columns=column_names)
    return (df, list(data_cohort_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    (df, cohorts) = prepare_data()
    run_mixed_effects_model(df, cohorts, ["Explanation_Type_Logreg"])
