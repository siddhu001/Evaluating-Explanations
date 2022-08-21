import argparse
import operator
import pickle
import sys
import time

sys.path.append("../src")
import numpy as np
from nltk.metrics import edit_distance
from word_edit_distance_highlighted_first_script import (
    generate_higlighted_dict, get_topk_words)

from analysis_utils import Utils
from user_study_id_files import (played_array_global_ablation, played_array_IG,
                                 played_array_lime, played_array_logreg)
from utils import compute_bootstrapped_confidence_interval, find_time_diff


def generate_edit_dist_graph(example):
    max_change_score_val = 0
    current_logs = example.log_array
    max_change_score_str = current_logs[0].output_text_no_tags
    max_change_score_exp = current_logs[0].output_text
    ins_ex_arr = []
    ins_high_ex_arr = []
    ins_ex_conf_arr = []
    ins_high_ex_conf_arr = []
    sub_high_high_ex_conf_arr = []
    sub_high_nonhigh_ex_conf_arr = []
    sub_nonhigh_high_ex_conf_arr = []
    sub_nonhigh_nonhigh_ex_conf_arr = []
    sub_ex_conf_arr = []
    sub_high_high_ex_arr = []
    sub_high_nonhigh_ex_arr = []
    sub_nonhigh_high_ex_arr = []
    sub_nonhigh_nonhigh_ex_arr = []
    sub_ex_arr = []
    del_ex_arr = []
    del_high_ex_arr = []
    del_ex_conf_arr = []
    del_high_ex_conf_arr = []
    for log_val_index in range(len(current_logs) - 1):
        num_words = len(current_logs[log_val_index].output_text_no_tags.split(" "))
        original_text_topk = get_topk_words(current_logs[log_val_index].output_text)
        edit_text_topk = get_topk_words(current_logs[log_val_index + 1].output_text)
        if current_logs[log_val_index].prob > 0.5:
            prob_change = max(
                current_logs[log_val_index].prob - current_logs[log_val_index + 1].prob,
                0,
            )
        else:
            prob_change = max(
                current_logs[log_val_index + 1].prob - current_logs[log_val_index].prob,
                0,
            )
        (ins_edit, subs_edit, del_edit) = Utils.editops(
            current_logs[log_val_index].output_text_no_tags.split(" "),
            current_logs[log_val_index + 1].output_text_no_tags.split(" "),
        )
        del_highlight_edit = 0
        ins_highlight_edit = 0
        sub_ins_high_del_high_edit = 0
        sub_ins_high_del_nonhigh_edit = 0
        sub_ins_nonhigh_del_high_edit = 0
        sub_ins_nonhigh_del_nonhigh_edit = 0
        for del_attempt in del_edit:
            if (int(del_attempt[1].split("->")[0])) in original_text_topk:
                del_highlight_edit += 1

        for ins_attempt in ins_edit:
            if (int(ins_attempt[1].split("->")[1])) in edit_text_topk:
                ins_highlight_edit += 1

        for sub_attempt in subs_edit:
            if (int(sub_attempt[1].split("->")[1])) in edit_text_topk:
                sub_highlighted_ins_word = True
            else:
                sub_highlighted_ins_word = False
            if (int(sub_attempt[1].split("->")[0])) in original_text_topk:
                sub_highlighted_del_word = True
            else:
                sub_highlighted_del_word = False
            if sub_highlighted_ins_word and sub_highlighted_del_word:
                sub_ins_high_del_high_edit += 1
            elif sub_highlighted_ins_word:
                sub_ins_high_del_nonhigh_edit += 1
            elif sub_highlighted_del_word:
                sub_ins_nonhigh_del_high_edit += 1
            else:
                sub_ins_nonhigh_del_nonhigh_edit += 1
        ins_edit_score = len(ins_edit) / num_words
        sub_edit_score = len(subs_edit) / num_words
        del_edit_score = len(del_edit) / num_words
        del_highlight_edit = del_highlight_edit / num_words
        ins_highlight_edit = ins_highlight_edit / num_words
        sub_ins_high_del_high_edit = sub_ins_high_del_high_edit / num_words
        sub_ins_high_del_nonhigh_edit = sub_ins_high_del_nonhigh_edit / num_words
        sub_ins_nonhigh_del_high_edit = sub_ins_nonhigh_del_high_edit / num_words
        sub_ins_nonhigh_del_nonhigh_edit = sub_ins_nonhigh_del_nonhigh_edit / num_words
        ins_ex_arr.append(ins_edit_score)
        ins_ex_conf_arr.append(ins_edit_score * prob_change)
        ins_high_ex_arr.append(ins_highlight_edit)
        ins_high_ex_conf_arr.append(ins_highlight_edit * prob_change)
        del_ex_arr.append(del_edit_score)
        del_ex_conf_arr.append(del_edit_score * prob_change)
        del_high_ex_arr.append(del_highlight_edit)
        del_high_ex_conf_arr.append(del_highlight_edit * prob_change)
        sub_ex_arr.append(sub_edit_score)
        sub_high_high_ex_arr.append(sub_ins_high_del_high_edit)
        sub_high_nonhigh_ex_arr.append(sub_ins_nonhigh_del_high_edit)
        sub_nonhigh_high_ex_arr.append(sub_ins_high_del_nonhigh_edit)
        sub_nonhigh_nonhigh_ex_arr.append(sub_ins_nonhigh_del_nonhigh_edit)
        sub_ex_conf_arr.append(sub_edit_score * prob_change)
        sub_high_high_ex_conf_arr.append(sub_ins_high_del_high_edit * prob_change)
        sub_high_nonhigh_ex_conf_arr.append(sub_ins_nonhigh_del_high_edit * prob_change)
        sub_nonhigh_high_ex_conf_arr.append(sub_ins_high_del_nonhigh_edit * prob_change)
        sub_nonhigh_nonhigh_ex_conf_arr.append(
            sub_ins_nonhigh_del_nonhigh_edit * prob_change
        )
    return_dict = {}
    return_dict["ins"] = {}
    return_dict["sub"] = {}
    return_dict["del"] = {}

    return_dict["ins"]["All"] = np.mean(ins_ex_arr)
    return_dict["ins"]["Highlight"] = np.mean(ins_high_ex_arr)
    return_dict["ins"]["All_Conf"] = np.mean(ins_ex_conf_arr)
    return_dict["ins"]["Highlight_Conf"] = np.mean(ins_high_ex_conf_arr)

    return_dict["sub"]["All"] = np.mean(sub_ex_arr)
    return_dict["sub"]["Highlight->Highlight"] = np.mean(sub_high_high_ex_arr)
    return_dict["sub"]["Highlight->No Highlight"] = np.mean(sub_high_nonhigh_ex_arr)
    return_dict["sub"]["No Highlight->Highlight"] = np.mean(sub_nonhigh_high_ex_arr)
    return_dict["sub"]["No Highlight->No Highlight"] = np.mean(
        sub_nonhigh_nonhigh_ex_arr
    )
    return_dict["sub"]["All_Conf"] = np.mean(sub_ex_conf_arr)
    return_dict["sub"]["Highlight->Highlight_Conf"] = np.mean(sub_high_high_ex_conf_arr)
    return_dict["sub"]["Highlight->No Highlight_Conf"] = np.mean(
        sub_high_nonhigh_ex_conf_arr
    )
    return_dict["sub"]["No Highlight->Highlight_Conf"] = np.mean(
        sub_nonhigh_high_ex_conf_arr
    )
    return_dict["sub"]["No Highlight->No Highlight_Conf"] = np.mean(
        sub_nonhigh_nonhigh_ex_conf_arr
    )

    return_dict["del"]["All"] = np.mean(del_ex_arr)
    return_dict["del"]["Highlight"] = np.mean(del_high_ex_arr)
    return_dict["del"]["All_Conf"] = np.mean(del_ex_conf_arr)
    return_dict["del"]["Highlight_Conf"] = np.mean(del_high_ex_conf_arr)

    return return_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_type",
        required=True,
        choices=[
            "lime",
            "log_reg",
            "integrated_gradients",
            "global_exp_ablation",
        ],
        help="type of explanation to display",
    )
    args = parser.parse_args()
    exp_type = args.exp_type
    if exp_type == "lime":
        played_array = played_array_lime
    elif exp_type == "log_reg":
        played_array = played_array_logreg
    elif exp_type == "integrated_gradients":
        played_array = played_array_IG
    elif exp_type == "global_exp_ablation":
        played_array = played_array_global_ablation
    index_dict = {}
    train_arr = []
    train_ins_arr = []
    train_ins_nonhigh_arr = []
    train_ins_high_arr = []
    train_ins_nonhigh_conf_arr = []
    train_ins_high_conf_arr = []
    train_sub_high_high_arr = []
    train_sub_high_nonhigh_arr = []
    train_sub_high_arr = []
    train_sub_nonhigh_high_arr = []
    train_sub_nonhigh_nonhigh_arr = []
    train_sub_nonhigh_arr = []
    train_sub_high_high_conf_arr = []
    train_sub_high_nonhigh_conf_arr = []
    train_sub_high_conf_arr = []
    train_sub_nonhigh_high_conf_arr = []
    train_sub_nonhigh_nonhigh_conf_arr = []
    train_sub_nonhigh_conf_arr = []
    train_sub_arr = []
    train_del_arr = []
    train_del_high_arr = []
    train_del_nonhigh_arr = []
    train_del_high_conf_arr = []
    train_del_nonhigh_conf_arr = []
    for k in played_array:
        user_name = k
        database = pickle.load(open("save_database_files/database_" + k + ".pkl", "rb"))
        for example in database.train_examples_arr:
            if len(example.log_array) < 2:
                continue
            return_dict = generate_edit_dist_graph(example)
            edit_dist = (
                return_dict["ins"]["All"]
                + return_dict["sub"]["All"]
                + return_dict["del"]["All"]
            )
            train_arr.append(edit_dist)
            train_ins_arr.append(return_dict["ins"]["All"])
            if return_dict["ins"]["All"] != 0:
                train_ins_high_arr.append(
                    return_dict["ins"]["Highlight"] / return_dict["ins"]["All"]
                )
                train_ins_nonhigh_arr.append(
                    1 - (return_dict["ins"]["Highlight"] / return_dict["ins"]["All"])
                )
            if return_dict["ins"]["All_Conf"] != 0:
                train_ins_high_conf_arr.append(
                    return_dict["ins"]["Highlight_Conf"]
                    / return_dict["ins"]["All_Conf"]
                )
                train_ins_nonhigh_conf_arr.append(
                    1
                    - (
                        return_dict["ins"]["Highlight_Conf"]
                        / return_dict["ins"]["All_Conf"]
                    )
                )
            train_sub_arr.append(return_dict["sub"]["All"])
            if return_dict["sub"]["All"] != 0:
                train_sub_high_high_arr.append(
                    return_dict["sub"]["Highlight->Highlight"]
                    / return_dict["sub"]["All"]
                )
                train_sub_high_nonhigh_arr.append(
                    return_dict["sub"]["Highlight->No Highlight"]
                    / return_dict["sub"]["All"]
                )
                train_sub_high_arr.append(
                    (
                        return_dict["sub"]["Highlight->Highlight"]
                        + return_dict["sub"]["Highlight->No Highlight"]
                    )
                    / return_dict["sub"]["All"]
                )
                train_sub_nonhigh_high_arr.append(
                    return_dict["sub"]["No Highlight->Highlight"]
                    / return_dict["sub"]["All"]
                )
                train_sub_nonhigh_nonhigh_arr.append(
                    return_dict["sub"]["No Highlight->No Highlight"]
                    / return_dict["sub"]["All"]
                )
                train_sub_nonhigh_arr.append(
                    (
                        return_dict["sub"]["No Highlight->Highlight"]
                        + return_dict["sub"]["No Highlight->No Highlight"]
                    )
                    / return_dict["sub"]["All"]
                )
            if return_dict["sub"]["All_Conf"] != 0:
                train_sub_high_high_conf_arr.append(
                    return_dict["sub"]["Highlight->Highlight_Conf"]
                    / return_dict["sub"]["All_Conf"]
                )
                train_sub_high_nonhigh_conf_arr.append(
                    return_dict["sub"]["Highlight->No Highlight_Conf"]
                    / return_dict["sub"]["All_Conf"]
                )
                train_sub_high_conf_arr.append(
                    (
                        return_dict["sub"]["Highlight->Highlight_Conf"]
                        + return_dict["sub"]["Highlight->No Highlight_Conf"]
                    )
                    / return_dict["sub"]["All_Conf"]
                )
                train_sub_nonhigh_high_conf_arr.append(
                    return_dict["sub"]["No Highlight->Highlight_Conf"]
                    / return_dict["sub"]["All_Conf"]
                )
                train_sub_nonhigh_nonhigh_conf_arr.append(
                    return_dict["sub"]["No Highlight->No Highlight_Conf"]
                    / return_dict["sub"]["All_Conf"]
                )
                train_sub_nonhigh_conf_arr.append(
                    (
                        return_dict["sub"]["No Highlight->Highlight_Conf"]
                        + return_dict["sub"]["No Highlight->No Highlight_Conf"]
                    )
                    / return_dict["sub"]["All_Conf"]
                )
            train_del_arr.append(return_dict["del"]["All"])
            if return_dict["del"]["All"] != 0:
                train_del_high_arr.append(
                    return_dict["del"]["Highlight"] / return_dict["del"]["All"]
                )
                train_del_nonhigh_arr.append(
                    1 - (return_dict["del"]["Highlight"] / return_dict["del"]["All"])
                )
            if return_dict["del"]["All_Conf"] != 0:
                train_del_high_conf_arr.append(
                    return_dict["del"]["Highlight_Conf"]
                    / return_dict["del"]["All_Conf"]
                )
                train_del_nonhigh_conf_arr.append(
                    1
                    - (
                        return_dict["del"]["Highlight_Conf"]
                        / return_dict["del"]["All_Conf"]
                    )
                )

    print("Training -")
    print("Word delete highlight -")
    print(round(np.mean(train_del_high_arr) * 100, 2))
    compute_bootstrapped_confidence_interval(np.array(train_del_high_arr) * 100)
    print("Word delete highlight confidence weighted -")
    print(round(np.mean(train_del_high_conf_arr) * 100, 2))
    compute_bootstrapped_confidence_interval(np.array(train_del_high_conf_arr) * 100)
    print("Word substitute with deleting highlighted -")
    print(round(np.mean(train_sub_high_arr) * 100, 2))
    compute_bootstrapped_confidence_interval(np.array(train_sub_high_arr) * 100)
    print("Word substitute with deleting highlighted confidence weighted -")
    print(round(np.mean(train_sub_high_conf_arr) * 100, 2))
    compute_bootstrapped_confidence_interval(np.array(train_sub_high_conf_arr) * 100)
