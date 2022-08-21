import argparse
import operator
import pickle
import sys
import time

sys.path.append("../src")
import numpy as np
from nltk.metrics import edit_distance

from analysis_utils import Utils
from user_study_id_files import (played_array_global_ablation, played_array_IG,
                                 played_array_lime, played_array_logreg)
from utils import compute_bootstrapped_confidence_interval, find_time_diff


def generate_higlighted_dict(output_text):
    highlight_dict = {}
    count_id = 0
    for text_span in output_text.split("<span")[1:]:
        word = text_span.replace("</span>", "").split('">')[1]
        highlight = text_span.split("rgb(")[1].split(")")[0]
        if count_id not in highlight_dict:
            highlight_dict[count_id] = []
        highlight_dict[count_id].append((word, highlight))
        if word[-1] == " ":
            count_id += 1
    return highlight_dict


def get_topk_words(output_text, topk_val=0.2):
    word_highlight_dict = generate_higlighted_dict(output_text)
    score_dict = {}
    word_dict = {}
    for word_count in word_highlight_dict:
        max_absolute_score = 0
        final_word = ""
        for (word, highlight) in word_highlight_dict[word_count]:
            final_word += word
            pos_highlight = int(highlight.strip().split(",")[2].strip())
            neg_highlight = int(highlight.strip().split(",")[1].strip())
            absolute_score = 0
            if pos_highlight != 255:
                absolute_score = (255 - pos_highlight) / 255
            elif neg_highlight != 255:
                absolute_score = (255 - neg_highlight) / 255
            if absolute_score > max_absolute_score:
                max_absolute_score = absolute_score
        score_dict[word_count] = max_absolute_score
        word_dict[word_count] = final_word
    topk_dict = {}
    sorted_x = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)
    for word_count, max_absolute_score in sorted_x[: int(topk_val * len(score_dict))]:
        topk_dict[word_count] = word_dict[word_count]

    if len(topk_dict) != int(topk_val * len(score_dict)):
        print("whhhatt")
        print(score_dict)
        print(topk_dict)
        exit()
    return topk_dict


def generate_edit_dist_graph(example):
    max_change_score_val = 0
    current_logs = example.log_array
    max_change_score_str = current_logs[0].output_text_no_tags
    max_change_score_exp = current_logs[0].output_text
    log_val_index = 0
    num_words = len(current_logs[log_val_index].output_text_no_tags.split(" "))
    original_text_topk = get_topk_words(current_logs[log_val_index].output_text)
    edit_text_topk = get_topk_words(current_logs[log_val_index + 1].output_text)
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
    return (
        ins_edit_score,
        sub_edit_score,
        del_edit_score,
        del_highlight_edit,
        ins_highlight_edit,
        sub_ins_high_del_high_edit,
        sub_ins_high_del_nonhigh_edit,
        sub_ins_nonhigh_del_high_edit,
        sub_ins_nonhigh_del_nonhigh_edit,
    )


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
    train_sub_high_high_arr = []
    train_sub_high_nonhigh_arr = []
    train_sub_high_arr = []
    train_sub_nonhigh_high_arr = []
    train_sub_nonhigh_nonhigh_arr = []
    train_sub_nonhigh_arr = []
    train_sub_arr = []
    train_del_arr = []
    train_del_high_arr = []
    train_del_nonhigh_arr = []
    for k in played_array:
        user_name = k
        database = pickle.load(open("save_database_files/database_" + k + ".pkl", "rb"))
        for example in database.train_examples_arr:
            if len(example.log_array) < 2:
                continue
            (
                ins_edit,
                subs_edit,
                del_edit,
                del_high_edit,
                ins_high_edit,
                sub_ins_high_del_high_edit,
                sub_ins_high_del_nonhigh_edit,
                sub_ins_nonhigh_del_high_edit,
                sub_ins_nonhigh_del_nonhigh_edit,
            ) = generate_edit_dist_graph(example)
            edit_dist = ins_edit + subs_edit + del_edit
            train_arr.append(edit_dist)
            train_ins_arr.append(ins_edit)
            if ins_edit != 0:
                train_ins_high_arr.append(ins_high_edit / ins_edit)
                train_ins_nonhigh_arr.append(1 - (ins_high_edit / ins_edit))
            train_sub_arr.append(subs_edit)
            if subs_edit != 0:
                train_sub_high_high_arr.append(sub_ins_high_del_high_edit / subs_edit)
                train_sub_high_nonhigh_arr.append(
                    sub_ins_nonhigh_del_high_edit / subs_edit
                )
                train_sub_high_arr.append(
                    (sub_ins_high_del_high_edit + sub_ins_nonhigh_del_high_edit)
                    / subs_edit
                )
                train_sub_nonhigh_high_arr.append(
                    sub_ins_high_del_nonhigh_edit / subs_edit
                )
                train_sub_nonhigh_nonhigh_arr.append(
                    sub_ins_nonhigh_del_nonhigh_edit / subs_edit
                )
                train_sub_nonhigh_arr.append(
                    (sub_ins_high_del_nonhigh_edit + sub_ins_nonhigh_del_nonhigh_edit)
                    / subs_edit
                )
            train_del_arr.append(del_edit)
            if del_edit != 0:
                train_del_high_arr.append(del_high_edit / del_edit)
                train_del_nonhigh_arr.append(1 - (del_high_edit / del_edit))

    print("Training -")
    print("Word delete highlight -")
    print(round(np.mean(train_del_high_arr) * 100, 2))
    compute_bootstrapped_confidence_interval(np.array(train_del_high_arr) * 100)
    print("Word substitute with deleting highlighted -")
    print(round(np.mean(train_sub_high_arr) * 100, 2))
    compute_bootstrapped_confidence_interval(np.array(train_sub_high_arr) * 100)
