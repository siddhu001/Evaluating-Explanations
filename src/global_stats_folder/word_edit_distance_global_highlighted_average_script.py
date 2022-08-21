import operator
import pickle
import sys
import time

sys.path.append("../src")

import numpy as np
from nltk.metrics import edit_distance

from analysis_utils import Utils
from user_study_id_files import played_array_global
from utils import compute_bootstrapped_confidence_interval, find_time_diff

genuine_top_words = [
    "Location",
    "floor",
    "elevators",
    "location",
    "elevator",
    "bell",
    "large",
    "bar",
    "2nd",
    "rate",
    "/",
    ".",
    "2",
    "(",
    "River",
    "upgraded",
    "straight",
    "Book",
    ")",
    "upgrade",
]
fake_top_words = [
    "Regency",
    "luxury",
    "Chicago",
    "luxurious",
    "I",
    "uneven",
    "welcomed",
    "definitely",
    "People",
    "sleep",
    "Hotel",
    "heart",
    "And",
    "personally",
    "supposed",
    "seemed",
    "securing",
    "managed",
    "Hilton",
    "Help",
]


def generate_higlighted_dict(output_text, genuine=True):
    highlight_dict = {}
    count_id = 0
    for text_span in output_text.split("<span")[1:]:
        word = text_span.replace("</span>", "").split('">')[1]
        if count_id not in highlight_dict:
            highlight_dict[count_id] = []
        highlight = 0
        if genuine:
            if word.strip() in genuine_top_words:
                highlight = 1
        else:
            if word.strip() in fake_top_words:
                highlight = 1
        highlight_dict[count_id].append((word, highlight))
        if word[-1] == " ":
            count_id += 1
    return highlight_dict


def get_topk_words(output_text, topk_val=0.2, genuine=True):
    word_highlight_dict = generate_higlighted_dict(output_text, genuine)
    score_dict = {}
    word_dict = {}
    for word_count in word_highlight_dict:
        max_absolute_score = 0
        final_word = ""
        for (word, highlight) in word_highlight_dict[word_count]:
            final_word += word
            absolute_score = highlight
            if absolute_score > max_absolute_score:
                max_absolute_score = absolute_score
        score_dict[word_count] = max_absolute_score
        word_dict[word_count] = final_word
    topk_dict = {}
    for word_count, max_absolute_score in score_dict.items():
        if max_absolute_score == 1:
            topk_dict[word_count] = word_dict[word_count]
    return topk_dict


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
        if current_logs[log_val_index].prob > 0.5:
            original_text_topk = get_topk_words(current_logs[log_val_index].output_text)
            edit_text_topk = get_topk_words(
                current_logs[log_val_index + 1].output_text, genuine=False
            )
            prob_change = max(
                current_logs[log_val_index].prob - current_logs[log_val_index + 1].prob,
                0,
            )
        else:
            original_text_topk = get_topk_words(
                current_logs[log_val_index].output_text, genuine=False
            )
            edit_text_topk = get_topk_words(current_logs[log_val_index + 1].output_text)
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
    played_array = played_array_global
    index_dict = {}
    train_arr = []
    train_ins_arr = []
    train_ins_high_arr = []
    train_del_arr = []
    train_del_high_arr = []
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
            if (return_dict["ins"]["All"] + return_dict["sub"]["All"]) != 0:
                train_ins_high_arr.append(
                    (
                        return_dict["ins"]["Highlight"]
                        + return_dict["sub"]["Highlight->Highlight"]
                        + return_dict["sub"]["No Highlight->Highlight"]
                    )
                    / (return_dict["ins"]["All"] + return_dict["sub"]["All"])
                )
            train_del_arr.append(return_dict["del"]["All"])
            if return_dict["del"]["All"] != 0:
                train_del_high_arr.append(
                    (
                        return_dict["del"]["Highlight"]
                        + return_dict["sub"]["Highlight->Highlight"]
                        + return_dict["sub"]["Highlight->No Highlight"]
                    )
                    / (return_dict["del"]["All"] + return_dict["sub"]["All"])
                )

    print("Training -")
    print("Word insert highlighted -")
    print(round(np.mean(train_ins_high_arr) * 100, 2))
    print(round(np.std(train_ins_high_arr) * 100, 2))
    compute_bootstrapped_confidence_interval(np.array(train_ins_high_arr) * 100)
    print("Word delete highlight -")
    print(round(np.mean(train_del_high_arr) * 100, 2))
    print(round(np.std(train_del_high_arr) * 100, 2))
    compute_bootstrapped_confidence_interval(np.array(train_del_high_arr) * 100)
