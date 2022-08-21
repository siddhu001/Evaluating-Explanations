# ----------------------------------------------------------------------------#
# Imports
# ----------------------------------------------------------------------------#

import argparse
# from flask.ext.sqlalchemy import SQLAlchemy
import logging
import os
import os.path
import pickle
import random
import re
import string
import time
from collections import defaultdict
from datetime import datetime
from logging import FileHandler, Formatter

import numpy as np
import torch
import torch.optim as optim
from flask import (Flask, flash, redirect, render_template, request, session,
                   url_for)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils import data
from tqdm import tqdm
from transformers import (AdamW, BertForSequenceClassification, BertTokenizer,
                          get_linear_schedule_with_warmup)

import importance
import importance_utils
import string_names
from data_utils import HotelReviewsProcessor, Review, input_to_features
from database import User
from forms import *
from model_training.main import set_seed
from temperature_scaling import ModelWithTemperature
from utils import compute_edit_dist, filter_reviews, strip_tags

parser = argparse.ArgumentParser()
# This option specify explanation type
parser.add_argument(
    "--explanation_type",
    default="attention",
    choices=[
        "attention",
        "integrated_gradients",
        "grad_times_inp",
        "grad_norm",
        "lime",
        "log_reg",
        "none",
        "global",
    ],
    help="type of explanation to display",
)
# This option specify explanation type
parser.add_argument(
    "--seed", default=123, type=int, help="random seed for torch, numpy, random"
)
# This option specify dataset directory
parser.add_argument("--data_dir", default="", type=str, help="data directory")
# This option specify number of example in training phase
parser.add_argument(
    "--num_train_examples", default=20, type=int, help="number of testing examples"
)
# This option specify number of example in train phase
parser.add_argument(
    "--num_test_examples",
    default=10,
    type=int,
    help="number of examples in prediction phase",
)
# This option specify number of example in test phase
parser.add_argument("--task", default="reviews", type=str, help="name of task")
# This option runs experiment in easy phase i.e.
# get examples only with confidence betweem 0.1 and 0.9
parser.add_argument(
    "--check_easy", default=False, type=str, help="Look at only examples easier to flip"
)
# This option filter review by word length
parser.add_argument(
    "--filter", default=False, type=str, help="Look at only shorter reviews to flip"
)
parser.add_argument(
    "--topk", default=False, type=str, help="Look at only top 10%  explanations"
)
parser.add_argument(
    "--global_exp", default=False, type=str, help="Provide global explanations"
)

parser.add_argument("--small", default=False, type=str, help="Prepare recruitment test")
parser.add_argument(
    "--control", default=False, type=str, help="Do not show explanation type"
)
parser.add_argument(
    "--no_global_words",
    default=False,
    type=str,
    help="Do not show global explanation words",
)
args = parser.parse_args()


# ----------------------------------------------------------------------------#
# App Config.
# ----------------------------------------------------------------------------#


data_cohort_pickle_dict = pickle.load(open("data_cohort_dict.pkl", "rb"))

now = datetime.now()


random.seed(5)
app = Flask(__name__)
app.config.from_object("config")
dataset_processor = HotelReviewsProcessor()
if args.check_easy:
    test_examples = dataset_processor.get_error_examples(args.data_dir)
else:
    test_examples = dataset_processor.get_test_examples(args.data_dir)
    if args.filter:
        # print(len(test_examples))
        test_examples = filter_reviews(test_examples)
random.shuffle(test_examples)

print([k.truthfulness for k in test_examples[: args.num_train_examples]])
# This is dictionary which save information of all users
user_dict = dict()

### Load your BERT model over here ###
set_seed(args.seed)
if args.global_exp:
    [vectorizer, best_clf_model] = pickle.load(open("logreg_bert_tokeniser.pkl", "rb"))
    feat_name = np.array(vectorizer.get_feature_names())
    weights = best_clf_model.coef_.reshape(-1).copy()  # feature weights
    weights /= np.max(np.abs(weights))

if args.explanation_type == "log_reg":
    [vectorizer, best_clf_model] = pickle.load(open("logreg_model.pth", "rb"))
    feat_name = np.array(vectorizer.get_feature_names())
    weights = best_clf_model.coef_.reshape(-1).copy()  # feature weights
    weights /= np.max(np.abs(weights))  # Normalise feature weights
else:
    orig_model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased", output_attentions=True
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = ModelWithTemperature(orig_model)
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if is_cuda else torch.device("cpu")
    print("Device in use: ", device)
    if args.task == "reviews":
        model.load_state_dict(
            torch.load("model_dumps/bert_cased_calibrated.pth", map_location=device)
        )
    model.to(device)
    model.eval()

### Load your Logistic regression model over here ###
def global_exp(tokens):
    input_X = vectorizer.transform([" ".join(tokens)])
    found_scores = {}
    rows, cols = input_X.nonzero()
    # this loop stores weights of feature in input text
    for k in cols:
        found_scores[feat_name[k]] = weights[k]
    scores = []
    for token in tokens:
        if token in found_scores:  # if word id one of the feature
            scores.append(found_scores[token])
        else:
            print("not found")
            print(token)
            scores.append(0)
    return scores


def logreg_predict(input1, test=False):
    """Computes the explanation and prediction for logistic regression.

    Args:
      input1: the text for which prediction need to be computed
      test: If user is in test phase or training phase

    Returns:
      prob: Probability of genuine class label
      output_html : Model explanation for prediction
    """
    input1 = input1.strip().replace(u"\xa0", u" ")
    print(input1)
    input_X = vectorizer.transform([input1])  # Run vectoriser on input
    # print(input_X)
    prob = best_clf_model.predict_proba(input_X)[0][1]  # Generate model probabilities
    # print(prob.shape)
    found_scores = {}
    rows, cols = input_X.nonzero()
    # this loop stores weights of feature in input text
    for k in cols:
        found_scores[feat_name[k]] = weights[k]
    tokens = input1.split(" ")
    print(tokens)
    print(found_scores)
    scores = []
    tokens_final = []
    # This loop find all tokens in put and assign them to corresponding weight
    for token in tokens:
        if token.lower() in found_scores:  # if word id one of the feature
            scores.append(found_scores[token.lower()])
            tokens_final.append(token)
        else:
            subtoken_array = re.findall(
                r"\w+|[^\w\s]", token, re.UNICODE
            )  # split the word into punctuation & words
            for word_count in range(len(subtoken_array)):
                if subtoken_array[word_count].lower() in found_scores:
                    scores.append(found_scores[subtoken_array[word_count].lower()])
                else:
                    scores.append(0.0)  # punctuation are removed in vectoriser
                if word_count == 0:
                    tokens_final.append(subtoken_array[word_count])
                else:
                    tokens_final.append(
                        "##" + subtoken_array[word_count]
                    )  # this option point that it is subword
    # print(tokens_final)
    # print(scores)
    # output_html = "Will add explanation"
    predicted_output = (
        string_names.genuine_str if prob > 0.5 else string_names.fake_str
    )  # saves model prediction
    out_prob = prob if prob > 0.5 else (1.0 - prob)  # saves model confidence
    if test:  # user in test phase
        output_html = input1  # just return input text
    else:
        output_html = importance_utils.add_highlights(
            tokens_final, scores, logreg=True, topk=args.topk
        )  # generate explanation based on feature weights
    return prob, output_html


def predict(input1, explanation_type, test=False):
    """Computes the explanation and prediction for BERT model.

    Args:
      input1: the text for which prediction need to be computed
      explanation_type: Type of model explanation
      test: If user is in test phase or training phase

    Returns:
      prob: Probability of genuine class label
      output_html : Model explanation for prediction
    """
    input1 = input1.strip().replace(u"\xa0", u" ")
    review_example = Review(content=input1, truthfulness=0, sentiment=0)
    feature = input_to_features(
        review_example, tokenizer, 512
    )  # generate features to pass through BERT
    long_type = torch.cuda.LongTensor if is_cuda else torch.LongTensor
    input_ids = torch.tensor([feature.input_ids]).type(long_type)
    attention_mask = torch.tensor([feature.attention_mask]).type(long_type)
    # print(input_ids.shape)
    # print(attention_mask.shape)
    user_dict[session.get("name")].input_ids = input_ids
    user_dict[session.get("name")].attention_mask = attention_mask
    assert (
        user_dict[session.get("name")].input_ids.shape
        == user_dict[session.get("name")].attention_mask.shape
    )
    output = model(
        user_dict[session.get("name")].input_ids,
        user_dict[session.get("name")].attention_mask,
    )  # computes output from BERT
    pred_scores = output[0][0]
    prediction = (
        torch.argmax(pred_scores, dim=-1).cpu().numpy()
    )  # gets model prediction
    predicted_output = string_names.genuine_str if prediction else string_names.fake_str
    prob = torch.nn.functional.softmax(pred_scores, dim=0)[prediction].item()
    prob = round(prob, 4)
    if test or explanation_type == "global":  # user in test phase
        output_html = input1
    else:
        # generate model explanation
        if explanation_type == "attention":
            explainer = importance.AttentionExplainer()
        elif explanation_type == "integrated_gradients":
            explainer = importance.IntegratedGradientExplainer()
        elif explanation_type == "grad_times_inp":
            explainer = importance.GradientBasedImportanceScores("grad_times_inp")
        elif explanation_type == "grad_norm":
            explainer = importance.GradientBasedImportanceScores("grad_norm")
        elif explanation_type == "lime":
            explainer = importance.LimeExplainer()
        else:
            raise ValueError("explanation type not supported")

        tokens = ["[CLS]"] + tokenizer.tokenize(input1)[: 512 - 2] + ["[SEP]"]

        if explanation_type == "lime":
            score_list = explainer.get_html_explanations(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                class_names=[string_names.fake_str, string_names.genuine_str],
            )
            scores = explainer.get_token_level_score(score_list[0], tokens)
        else:
            user_dict[session.get("name")].input_ids = input_ids
            user_dict[session.get("name")].attention_mask = attention_mask
            scores = explainer.get_scores(model, user_dict, session)
            print(scores)
            scores = scores[0]
            print(scores)
            print(len(scores))
            print(len(tokens))
            if (
                predicted_output == string_names.fake_str
            ):  # make weights negative if model prediction fake
                # if explanation_type == "attention" or explanation_type == "grad_norm":
                scores = [x * -1 for x in scores]
                print(len(scores))
        assert len(tokens) == len(scores)
        output_html = importance_utils.add_highlights(
            tokens, scores, topk=args.topk
        )  # add span colour to text based on model explanation
    if predicted_output == string_names.fake_str:
        return (1.0 - prob), output_html
    else:
        return prob, output_html


def global_predict(input1, test=False):
    """Computes the explanation and prediction for BERT model.

    Args:
      input1: the text for which prediction need to be computed
      explanation_type: Type of model explanation
      test: If user is in test phase or training phase

    Returns:
      prob: Probability of genuine class label
      output_html : Model explanation for prediction
    """
    input1 = input1.strip().replace(u"\xa0", u" ")
    review_example = Review(content=input1, truthfulness=0, sentiment=0)
    feature = input_to_features(
        review_example, tokenizer, 512
    )  # generate features to pass through BERT
    long_type = torch.cuda.LongTensor if is_cuda else torch.LongTensor
    input_ids = torch.tensor([feature.input_ids]).type(long_type)
    attention_mask = torch.tensor([feature.attention_mask]).type(long_type)
    # print(input_ids.shape)
    # print(attention_mask.shape)
    output = model(input_ids, attention_mask)  # computes output from BERT
    pred_scores = output[0][0]
    prediction = (
        torch.argmax(pred_scores, dim=-1).cpu().numpy()
    )  # gets model prediction
    predicted_output = string_names.genuine_str if prediction else string_names.fake_str
    prob = torch.nn.functional.softmax(pred_scores, dim=0)[prediction].item()
    prob = round(prob, 4)
    if test:  # user in test phase
        output_html = input1
    else:
        tokens = feature.tokens
        scores = global_exp(tokens)
        assert len(tokens) == len(scores)
        output_html = importance_utils.add_highlights(
            tokens, scores, topk=args.topk
        )  # add span colour to text based on model explanation
    if predicted_output == string_names.fake_str:
        return (1.0 - prob), output_html
    else:
        return prob, output_html


def show_train_example(name, flipped=False, edit_reject=False):
    """Show the example in training phase.

    Args:
      name: name of user
      flipped: True if user has already flipped model prediction

    Returns:
      template: html page shown on browser
    """
    # Compute example id of user
    cur_id = user_dict[name].train_index + user_dict[name].test_index + 1
    # Computes number of correct guess
    correct_guesses = (
        user_dict[name].test_correct_guesses + user_dict[name].correct_guesses
    )
    # Computes number of examples correctly flipped
    correct_flips = len(user_dict[name].examples_flipped_dict) + len(
        user_dict[name].test_examples_flipped_dict
    )
    user_dict[name].calculate_bonus()
    if edit_reject:
        incorrect_edit = string_names.incorrect_edit_str
    else:
        incorrect_edit = ""
    if args.global_exp and user_dict[name].explanation_type == "log_reg":
        global_explanation_str = string_names.global_logreg_explanation_str
    elif args.global_exp and (not (args.no_global_words)):
        global_explanation_str = string_names.global_bert_explanation_str
    else:
        global_explanation_str = ""
    if not (user_dict[name].train_guess_state()):
        # User is in edit example phase
        # Add caret showing model confidence
        attention_explain_str1 = (
            "<div style = 'left: "
            + str(7 + (user_dict[name].get_train_log_array().prob * 80))
            + "%;position: relative;'> <b>Confidence</b></div><div style = 'left: "
            + str(9 + (user_dict[name].get_train_log_array().prob * 80))
            + "%;position: relative;'> &#94;</div><div style = 'text-align: center;'> "
            + string_names.attention_explain_image
            + "</div><div style = 'left: 49%;\
    position: relative;'> &#94;</div><div style = 'left: 47%;\
    position: relative;'> <span style = 'color: rgb(0, 255, 0)'>\
    <b>Target</b></span></div>"
        )
        if flipped:
            # If user has correctly flipped model prediction
            textarea_final_str = (
                string_names.div_readonly_str
            )  # text area no longer editable
            edit_review_final = (
                user_dict[name].get_train_log_array().user_feedback
            )  # congrats user for flipping prediction
            next_final_str = string_names.next_viewable_str  # show next button
        else:
            # If user has correctly flipped model prediction
            textarea_final_str = string_names.div_editable_str  # text area is editable
            edit_review_final = user_dict[name].get_train_user_guesses()[
                2
            ]  # give user feedback
            next_final_str = string_names.next_hidden_str  # not show next button
        template = render_template(
            "start.html",
            example_index=cur_id,
            output_text=user_dict[name].get_train_log_array().output_text,
            flips=correct_flips,
            guess_text=user_dict[name].get_train_user_guesses()[0],
            edit_review=edit_review_final,
            correct_guesses=str(correct_guesses) + "/" + str(cur_id),
            attention_explanation=attention_explain_str1,
            time_left=user_dict[name].get_train_time_left(),
            time_left_sec=user_dict[name].get_train_time_left_sec(),
            div_str=textarea_final_str,
            heading_str=string_names.heading_edit_str,
            overall_feedback=user_dict[name].get_train_log_array().overall_feedback,
            next_str=next_final_str,
            bonus=user_dict[name].bonus,
            incorrect_edit=incorrect_edit,
            global_explanation=global_explanation_str,
        )
        # example_index - index of the example user has to be shown
        # output_text - example text that is displayed in textbox
        # flips - number of examples correctly flipped
        # guess_text -  feedback about user guess
        return template
    else:
        # User is in guessing model prediction phase
        return render_template(
            "start.html",
            example_index=cur_id,
            output_text=user_dict[name].get_train_input(),
            flips=correct_flips,
            radio_str=string_names.original_radio_str,
            correct_guesses=str(correct_guesses) + "/" + str(cur_id - 1),
            time_left=user_dict[name].get_train_time_left(),
            time_left_sec=user_dict[name].get_train_time_left_sec(),
            div_str=string_names.div_readonly_str,
            heading_str=string_names.heading_guess_str,
            next_str=string_names.next_hidden_str,
            bonus=user_dict[name].bonus,
        )


def check_flip_model_prediction(curr_example, prob, dt_now, examples_dict):
    """Checks if model has flipped prediction

    Args:
      curr_example: object of example class user is currently editing
      prob: Model prediction for current edit
      dt_now: Time at which the edit was made
      examples_dict: Dictionary of flipped examples

    Returns:
      boolean variable indicating if the prediction flipped
    """
    if curr_example.original_score > 0.5:  # Example is originally genuine
        if prob < curr_example.original_score:  # If user has decreased probability
            if (curr_example.original_score - prob) > curr_example.max_change_score:
                # if this edit is maximum decrease in model confidence
                curr_example.max_change_score = curr_example.original_score - prob
        if prob < 0.5:  # if user has also flipped prediction
            examples_dict[curr_example.index] = dt_now  # save time when example flipped
            return True
    elif curr_example.original_score < 0.5:  # Example is originally fake
        # rest is same as before
        if prob > curr_example.original_score:
            if (prob - curr_example.original_score) > curr_example.max_change_score:
                curr_example.max_change_score = prob - curr_example.original_score
        if prob > 0.5:
            examples_dict[curr_example.index] = dt_now
            return True
    return False


def show_test_example(name, flipped=False, edit_reject=False):
    """Show the example in test phase.

    Args:
      name: name of user
      predict: True if user has already flipped model prediction

    Returns:
      template: html page shown on browser
    """
    # For documentation please refer to show_train_example, functions are identical
    cur_id = user_dict[name].train_index + user_dict[name].test_index + 1
    correct_guesses = (
        user_dict[name].test_correct_guesses + user_dict[name].correct_guesses
    )
    correct_flips = len(user_dict[name].examples_flipped_dict) + len(
        user_dict[name].test_examples_flipped_dict
    )
    user_dict[name].calculate_bonus()
    if edit_reject:
        incorrect_edit = string_names.incorrect_edit_str
    else:
        incorrect_edit = ""
    if not (user_dict[name].test_guess_state()):
        attention_explain_str1 = (
            "<div style = 'left: "
            + str(7 + (user_dict[name].get_test_log_array().prob * 80))
            + "%;position: relative;'> <b>Confidence</b></div><div style = 'left: "
            + str(9 + (user_dict[name].get_test_log_array().prob * 80))
            + "%;position: relative;'> &#94;</div><div style = 'text-align: center;'> "
            + string_names.attention_explain_image
            + "</div><div style = 'left: 49%;position: relative;'> \
    &#94;</div><div style = 'left: 47%;position: relative;'> \
    <span style = 'background-color: rgb(0, 255, 0)'><b>Target</b></span></div>"
        )
        total_id = cur_id
        if flipped:
            textarea_final_str = string_names.div_readonly_str
            edit_review_final = user_dict[name].get_test_log_array().user_feedback
            next_final_str = string_names.next_viewable_str
        else:
            textarea_final_str = string_names.div_editable_str
            edit_review_final = user_dict[name].get_test_user_guesses()[2]
            next_final_str = string_names.next_hidden_str
        template = render_template(
            "predict.html",
            example_index=cur_id,
            output_text=user_dict[name].get_test_log_array().output_text,
            flips=correct_flips,
            guess_text=user_dict[name].get_test_user_guesses()[0],
            edit_review=edit_review_final,
            correct_guesses=str(correct_guesses) + "/" + str(total_id),
            time_left=user_dict[name].get_test_time_left(),
            time_left_sec=user_dict[name].get_test_time_left_sec(),
            div_str=textarea_final_str,
            heading_str=string_names.heading_edit_str,
            overall_feedback=user_dict[name].get_test_log_array().overall_feedback,
            next_str=next_final_str,
            attention_explanation=attention_explain_str1,
            bonus=user_dict[name].bonus,
            incorrect_edit=incorrect_edit,
        )
        return template
    else:
        total_id = cur_id - 1
        return render_template(
            "predict.html",
            example_index=cur_id,
            output_text=user_dict[name].test_input(),
            flips=correct_flips,
            radio_str=string_names.original_radio_str,
            correct_guesses=str(correct_guesses) + "/" + str(total_id),
            time_left=user_dict[name].get_test_time_left(),
            time_left_sec=user_dict[name].get_test_time_left_sec(),
            div_str=string_names.div_readonly_str,
            heading_str=string_names.heading_guess_str,
            next_str=string_names.next_hidden_str,
            bonus=user_dict[name].bonus,
        )


def get_feedback(curr_example, prob, dt_now, examples_dict, cur_change):
    """Generates feedback about change in prediction with user's last edit.

    Args:
      curr_example: object of example class user is currently editing
      prob: Model prediction for current edit
      dt_now: Time at which the edit was made
      examples_dict: Dictionary of flipped examples
      cur_change: Change in prediction made by the current edit

    Returns:
      overall_feedback: Generates feedback string showing max change with
                        the originial model prediction and the prediction change
                        made by last attempt
      user_feedback: Generates feedback string deisplaying if user has flipped model
                      prediction
    """
    if curr_example.index not in examples_dict:  # if example has not been flipped yet
        if check_flip_model_prediction(curr_example, prob, dt_now, examples_dict):
            # if edit flips model prediction
            user_feedback = string_names.successful_flip_attempt
        else:
            print("ooookkkk")
            user_feedback = string_names.unsuccessful_flip_attempt
    else:
        user_feedback = string_names.already_successful_flip_attempt
    if round(curr_example.max_change_score * 100, 1) == 0:
        # if model confidence has not reduced by any of edits
        if curr_example.original_score > 0.5:
            overall_feedback = "<p>" + string_names.progress_str + "No Decrease</p>"
        else:
            overall_feedback = "<p>" + string_names.progress_str + "No Decrease</p>"
    else:
        # show by how much model confidence has reduced
        if curr_example.original_score > 0.5:
            overall_feedback = (
                "<p>"
                + string_names.progress_str
                + str(round(curr_example.max_change_score * 100, 1))
                + "%</p>"
            )
        else:
            overall_feedback = (
                "<p>"
                + string_names.progress_str
                + str(round(curr_example.max_change_score * 100, 1))
                + "%</p>"
            )
    # print(cur_change)
    # Generate change in model confidence made by current attempt
    if curr_example.original_score > 0.5:
        if cur_change > 0:
            overall_feedback += (
                "<p> "
                + string_names.last_step_str
                + "-"
                + str(round(abs(cur_change), 1))
                + "%</p>"
            )
        elif cur_change < 0:
            overall_feedback += (
                "<p> "
                + string_names.last_step_str
                + str(round(abs(cur_change), 1))
                + "%</p>"
            )
        else:
            overall_feedback += "<p> " + string_names.last_step_str + "No Change</p>"
    else:
        if cur_change > 0:
            overall_feedback += (
                "<p> "
                + string_names.last_step_str
                + str(round(abs(cur_change), 1))
                + "%</p>"
            )
        elif cur_change < 0:
            overall_feedback += (
                "<p> "
                + string_names.last_step_str
                + "-"
                + str(round(abs(cur_change), 1))
                + "%</p>"
            )
        else:
            overall_feedback += "<p> " + string_names.last_step_str + "No Change</p>"
    return (overall_feedback, user_feedback)


def add_current_pred_to_feedback(overall_feedback, prob):
    """Generates feedback about model confidence for user's last edit.

    Args:
      overall_feedback: Feedback string generated so far
      prob: Probability of current edit

    Returns:
      overall_feedback: Generates feedback string showing max change with
                        the originial model prediction,the prediction change
                        made by last attempt, current prediction and current model
                        confidence
    """
    if prob > 0.5:
        # If model predicted genuine
        overall_feedback += (
            '<p> Current prediction: <span style = "background-color: rgb(255, 255, 0)">'
            + string_names.genuine_str
            + "</span></p>"
        )
        overall_feedback += (
            "<p> Current confidence: " + str(round(prob * 100, 1)) + "%</p>"
        )
    else:
        overall_feedback += (
            '<p> Current prediction: <span style = "background-color: rgb(255, 0, 255)">'
            + string_names.fake_str
            + "</span></p>"
        )
        overall_feedback += (
            "<p> Current confidence: " + str(round((1 - prob) * 100, 1)) + "%</p>"
        )
    # Generate box around overall feedback given to user
    overall_feedback = (
        '<div class = "box" id = "includedContent" style = "border : 2px solid; margin : 0 auto;"><p> '
        + overall_feedback
        + " </p></div><br>"
    )
    return overall_feedback


# ----------------------------------------------------------------------------#
# Controllers.
# ----------------------------------------------------------------------------#


@app.after_request
def after_request(response):
    """Avoid caching of previous response
    to avoid user to go to previous example by clicking back
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.route("/survey", methods=["GET", "POST"])
def survey():
    """Shows survey page and collects
    feedback from users
    """
    if request.method == "GET":
        return render_template("survey.html")
    if request.method == "POST":
        name = session.get("name")
        print(request.form)
        if (
            "user_acc_belief" not in request.form
            or "ai_acc_belief" not in request.form
            or "assistance_a" not in request.form
            or "comprehensive_a" not in request.form
        ):
            # User cannot submit without answering compulsory questions
            return render_template(
                "survey.html", error_message=string_names.end_page_error_message
            )
        user_dict[name].survey_response_dict["explanation_agree"] = request.form[
            "user_acc_belief"
        ]
        user_dict[name].survey_response_dict["explanation_frequency"] = request.form[
            "ai_acc_belief"
        ]
        user_dict[name].survey_response_dict["Time"] = request.form["assistance_a"]
        if "assistance_b" in request.form:
            user_dict[name].survey_response_dict["Time more explain"] = request.form[
                "assistance_b"
            ]
        user_dict[name].survey_response_dict["Comprehensive"] = request.form[
            "comprehensive_a"
        ]
        if "comprehensive_b" in request.form:
            user_dict[name].survey_response_dict[
                "Comprehensive more explain"
            ] = request.form["comprehensive_b"]
        if "feedback" in request.form:
            user_dict[name].survey_response_dict["Suggestions"] = request.form[
                "feedback"
            ]
        pickle.dump(
            user_dict[name], open("database_" + name + ".pkl", "wb")
        )  # save to database
        return redirect(url_for("end"))


@app.route("/instruction")
def instruction():
    """Shows instructions popup window"""
    return render_template("instruction.html")


@app.route("/end")
def end():
    """Shows end page with amazon coupon code"""
    if request.method == "GET":
        name = session.get("name")
        if user_dict[name].coupon_code is None:
            rand_int = random.randint(10000, 99999)
            user_dict[name].coupon_code = rand_int
            pickle.dump(user_dict[name], open("database_" + name + ".pkl", "wb"))
        else:
            rand_int = user_dict[name].coupon_code
        return render_template("end.html", coupon=rand_int)


@app.route("/", methods=["GET", "POST"])
def start():
    """Shows start page with instructions and asking user
    to enter his username
    """
    if request.method == "GET":
        name = session.get("name")
        if name is None:
            name = ""
        return render_template("index.html", name=name)
    if request.method == "POST":
        if "options2" not in request.form:
            # is user does not accept terms of condition
            return render_template(
                "index.html", error_message=string_names.start_page_error_message
            )
        name = request.form["text"]
        print(app.secret_key)
        session["name"] = name
        print("Name")
        print(name)
        print(request.form)
        testing_inputs = data_cohort_pickle_dict[int(len(user_dict) % 5)]
        if name not in user_dict:
            # initialise object of User class
            if args.small:
                user_dict[name] = User(
                    name,
                    test_examples[21:23],
                    test_examples[args.num_train_examples :],
                    args,
                    testing_inputs,
                )
            else:
                user_dict[name] = User(
                    name,
                    test_examples[: args.num_train_examples],
                    test_examples[args.num_train_examples :],
                    args,
                    testing_inputs,
                )
            pickle.dump(user_dict[name], open("database_" + name + ".pkl", "wb"))
            return redirect(url_for("submit"))
        else:
            # User restarting game
            # loads user information from database
            user_dict[name] = pickle.load(open("database_" + name + ".pkl", "rb"))
            if user_dict[name].coupon_code is not None:
                rand_int = user_dict[name].coupon_code
                return render_template("end.html", coupon=rand_int)
            if user_dict[name].show_test:
                # if user in test phase
                return redirect(url_for("test_submit"))
            else:
                # if user in train phase
                return redirect(url_for("submit"))


@app.route("/formsubmit", methods=["GET", "POST"])
def submit():
    """
    Shows the browser page for training example and
    collects response receved from user during training phase
    """
    name = session.get("name")
    if request.method == "POST":
        # if user has clicked next/ guesses prediction or made edits
        user_dict[name].set_train_time_left(
            request.form["timer"]
        )  # checks time left in timer and update databse
        print(user_dict[name].get_train_time_left())
        if "m" in user_dict[name].get_train_time_left():
            minutes = int(user_dict[name].get_train_time_left().split("m")[0])
        else:
            minutes = 0
        if "s" in user_dict[name].get_train_time_left():
            seconds = int(
                user_dict[name].get_train_time_left().split("m")[-1].replace("s", "")
            )
        else:
            seconds = 0
        user_dict[name].set_train_time_left_sec(
            60 * minutes + seconds
        )  # update database with seconds left
        print(user_dict[name].get_train_time_left_sec())
        str_now = time.time()
        if (
            request.form["submit_btn"] == "Submit"
        ):  # User has either guessed prediction or edited example
            ins_err, sub_err, del_err = compute_edit_dist(
                user_dict[name].train_examples_arr[user_dict[name].train_index].text,
                strip_tags(request.form["text"]),
            )
            if (ins_err + sub_err + del_err) > 0.9 or (del_err > 0.5):
                return show_train_example(name, edit_reject=True)
            now = datetime.now()
            dt_now = now.strftime("%d/%m/%Y %H:%M:%S")
            if (
                user_dict[name].explanation_type == "log_reg"
            ):  # Logistic Regression model
                if args.control:
                    prob, output_text = logreg_predict(
                        strip_tags(request.form["text"]), test=True
                    )
                else:
                    prob, output_text = logreg_predict(strip_tags(request.form["text"]))
            elif args.global_exp:
                prob, output_text = global_predict(strip_tags(request.form["text"]))
            elif user_dict[name].explanation_type == "none":
                prob, output_text = predict(
                    strip_tags(request.form["text"]),
                    explanation_type=user_dict[name].explanation_type,
                    test=True,
                )
            else:
                # BERT based exaplantion
                prob, output_text = predict(
                    strip_tags(request.form["text"]),
                    explanation_type=user_dict[name].explanation_type,
                )

            if user_dict[name].train_guess_state():
                # Checks if this example is appearing the first time
                # Hence we need to check if user made correct guess
                print(request.form["options"])
                if request.form["options"] == "genuine":
                    # if user selected genuine
                    if prob > 0.5:
                        # correct guess
                        user_dict[name].correct_guesses = (
                            user_dict[name].correct_guesses + 1
                        )
                        user_dict[name].set_train_user_guesses(
                            [
                                string_names.genuine_correct_str,
                                string_names.genuine_str,
                                string_names.edit_review_fake,
                            ]
                        )  # store user guess
                    else:
                        # incorrect guess
                        user_dict[name].set_train_user_guesses(
                            [
                                string_names.genuine_incorrect_str,
                                string_names.genuine_str,
                                string_names.edit_review_genuine,
                            ]
                        )
                elif request.form["options"] == "fake":
                    # if user selected fake
                    if prob > 0.5:
                        # correct guess
                        user_dict[name].set_train_user_guesses(
                            [
                                string_names.fake_incorrect_str,
                                string_names.fake_str,
                                string_names.edit_review_fake,
                            ]
                        )
                    else:
                        # incorrect guess
                        user_dict[name].correct_guesses = (
                            user_dict[name].correct_guesses + 1
                        )
                        user_dict[name].set_train_user_guesses(
                            [
                                string_names.fake_correct_str,
                                string_names.fake_str,
                                string_names.edit_review_genuine,
                            ]
                        )
                user_dict[name].set_train_original_score(
                    prob
                )  # save original model confidence
                user_dict[name].set_train_max_change_score(
                    0
                )  # initialise change in model confidence
                user_dict[name].set_train_previous_score(
                    prob
                )  # initialise confidence of previous attempt
                user_feedback = ""
                overall_feedback = ""
            else:
                # calculate change in model confidence
                cur_change = round(
                    (prob - user_dict[name].get_train_previous_score()) * 100, 1
                )
                print(cur_change)
                user_dict[name].set_train_previous_score(
                    prob
                )  # update confidence of previous attempt in database
                # get feedback of decrease in confidence by user attempt
                (overall_feedback, user_feedback) = get_feedback(
                    user_dict[name].train_examples_arr[user_dict[name].train_index],
                    prob,
                    dt_now,
                    user_dict[name].examples_flipped_dict,
                    cur_change,
                )
            # get feedback about decrease in confidence by current attempt
            overall_feedback = add_current_pred_to_feedback(overall_feedback, prob)
            cur_time = time.time()
            # save log of attempts for current example
            user_dict[name].set_train_log_array(
                (dt_now, prob, user_feedback, overall_feedback, output_text)
            )
            start_time = user_dict[
                name
            ].get_train_starttime()  # time at which user started viewing eg
            # compute seconds left for current example
            user_dict[name].set_train_time_left_sec(
                180 - float(cur_time) + float(start_time)
            )
            timerMin = int(user_dict[name].get_train_time_left_sec() / 60)
            timerSec = int(user_dict[name].get_train_time_left_sec() - (60 * timerMin))
            user_dict[name].set_train_time_left(
                str(timerMin) + "m" + str(timerSec) + "s"
            )

            if user_feedback == string_names.successful_flip_attempt:
                # if user has flipped model prediction
                template = show_train_example(
                    name, flipped=True
                )  # create html page that should be displayed
            else:
                template = show_train_example(
                    name
                )  # create html page that should be displayed
            pickle.dump(
                user_dict[name], open("database_" + name + ".pkl", "wb")
            )  # save in database
            return template
        elif request.form["submit_btn"] == "Next":
            # user has pressed next button
            if user_dict[name].train_index < (args.num_train_examples):
                # update example index in database
                user_dict[name].train_index = user_dict[name].train_index + 1
            if user_dict[name].train_index % 2 == 0:
                # if user has seen 2 training examples, move to example in test phase
                user_dict[name].show_test = True
                pickle.dump(user_dict[name], open("database_" + name + ".pkl", "wb"))
                if args.small:
                    return redirect(url_for("end"))
                else:
                    return redirect(
                        url_for("test_submit")
                    )  # moves to example in test phase
            user_dict[name].set_train_time_left("3m")  # initialises time left
            user_dict[name].set_train_time_left_sec(180)
            user_dict[name].set_train_starttime(time.time())
            return show_train_example(name)

    if request.method == "GET":
        # User just wants to view the example
        if user_dict[name].train_index >= (args.num_train_examples):
            return redirect(url_for("test_submit"))
        if user_dict[name].train_not_initialised():
            # User is viewing example for first time
            user_dict[name].set_train_time_left("3m")  # initialise time left
            user_dict[name].set_train_time_left_sec(180)
            user_dict[name].set_train_starttime(time.time())  # initialise start time
        else:
            # User is resuming an example after maybe a break
            start_time = user_dict[name].get_train_starttime()
            cur_time = time.time()
            # computes time left and updates database
            user_dict[name].set_train_time_left_sec(
                180 - float(cur_time) + float(start_time)
            )
            timerMin = int(user_dict[name].get_train_time_left_sec() / 60)
            timerSec = int(user_dict[name].get_train_time_left_sec() - (60 * timerMin))
            user_dict[name].set_train_time_left(
                str(timerMin) + "m" + str(timerSec) + "s"
            )
        return show_train_example(
            name,
            flipped=user_dict[name].train_index
            in user_dict[name].examples_flipped_dict,
        )


@app.route("/formtest", methods=["GET", "POST"])
def test_submit():
    """
    Shows the browser page for test example and
    collects response receved from user during test phase
    """
    # this function is identical to submit()
    name = session.get("name")
    if request.method == "POST":
        # if user has clicked next/ guesses prediction or made edits
        user_dict[name].set_test_time_left(
            request.form["timer"]
        )  # checks time left in timer and update databse
        print(user_dict[name].get_test_time_left())
        if "m" in user_dict[name].get_test_time_left():
            minutes = int(user_dict[name].get_test_time_left().split("m")[0])
        else:
            minutes = 0
        if "s" in user_dict[name].get_test_time_left():
            seconds = int(
                user_dict[name].get_test_time_left().split("m")[-1].replace("s", "")
            )
        else:
            seconds = 0
        user_dict[name].set_test_time_left_sec(
            60 * minutes + seconds
        )  # update database with seconds left
        print(user_dict[name].get_test_time_left_sec())
        str_now = time.time()
        if request.form["submit_btn"] == "Submit":
            # User has either guessed prediction or edited example
            ins_err, sub_err, del_err = compute_edit_dist(
                user_dict[name].test_examples_arr[user_dict[name].test_index].text,
                strip_tags(request.form["text"]),
            )
            if (ins_err + sub_err + del_err) > 0.9 or (del_err > 0.5):
                return show_test_example(name, edit_reject=True)
            now = datetime.now()
            dt_now = now.strftime("%d/%m/%Y %H:%M:%S")
            if user_dict[name].explanation_type == "log_reg":
                prob, _ = logreg_predict(strip_tags(request.form["text"]), test=True)
            elif args.global_exp:
                prob, output_text = global_predict(
                    strip_tags(request.form["text"]), test=True
                )
            else:
                prob, _ = predict(
                    strip_tags(request.form["text"]),
                    explanation_type=user_dict[name].explanation_type,
                    test=True,
                )
            output_text = request.form["text"]
            print(output_text)
            if user_dict[name].test_guess_state():
                # Checks if this example is appearing the first time
                # Hence we need to check if user made correct guess
                print(request.form["options"])
                if request.form["options"] == "genuine":
                    if prob > 0.5:
                        user_dict[name].test_correct_guesses = (
                            user_dict[name].test_correct_guesses + 1
                        )
                        user_dict[name].set_test_user_guesses(
                            [
                                string_names.genuine_correct_str,
                                string_names.genuine_str,
                                string_names.edit_review_fake,
                            ]
                        )
                    else:
                        user_dict[name].set_test_user_guesses(
                            [
                                string_names.genuine_incorrect_str,
                                string_names.genuine_str,
                                string_names.edit_review_genuine,
                            ]
                        )
                elif request.form["options"] == "fake":
                    if prob > 0.5:
                        user_dict[name].set_test_user_guesses(
                            [
                                string_names.fake_incorrect_str,
                                string_names.fake_str,
                                string_names.edit_review_fake,
                            ]
                        )
                    else:
                        user_dict[name].test_correct_guesses = (
                            user_dict[name].test_correct_guesses + 1
                        )
                        user_dict[name].set_test_user_guesses(
                            [
                                string_names.fake_correct_str,
                                string_names.fake_str,
                                string_names.edit_review_genuine,
                            ]
                        )
                user_dict[name].set_test_original_score(prob)
                print("oookkk")
                print(
                    user_dict[name]
                    .test_examples_arr[user_dict[name].test_index]
                    .original_score
                )
                user_dict[name].set_test_max_change_score(0)
                user_dict[name].set_test_previous_score(prob)
                user_feedback = ""
                overall_feedback = ""
            else:
                cur_change = round(
                    (prob - user_dict[name].get_test_previous_score()) * 100, 1
                )
                user_dict[name].set_test_previous_score(prob)
                (overall_feedback, user_feedback) = get_feedback(
                    user_dict[name].test_examples_arr[user_dict[name].test_index],
                    prob,
                    dt_now,
                    user_dict[name].test_examples_flipped_dict,
                    cur_change,
                )
            overall_feedback = add_current_pred_to_feedback(overall_feedback, prob)
            user_dict[name].set_test_log_array(
                (dt_now, prob, user_feedback, overall_feedback, output_text)
            )
            cur_time = time.time()
            start_time = user_dict[name].get_test_starttime()
            user_dict[name].set_test_time_left_sec(
                180 - float(cur_time) + float(start_time)
            )
            timerMin = int(user_dict[name].get_test_time_left_sec() / 60)
            timerSec = int(user_dict[name].get_test_time_left_sec() - (60 * timerMin))
            user_dict[name].set_test_time_left(
                str(timerMin) + "m" + str(timerSec) + "s"
            )
            if user_feedback == string_names.successful_flip_attempt:
                template = show_test_example(name, flipped=True)
            else:
                template = show_test_example(name)
            pickle.dump(user_dict[name], open("database_" + name + ".pkl", "wb"))
            return template
        elif request.form["submit_btn"] == "Next":
            # user has pressed next button
            if user_dict[name].test_index < (args.num_test_examples - 1):
                # go back to training phase
                user_dict[name].test_index = user_dict[name].test_index + 1
                user_dict[name].show_test = False
                pickle.dump(user_dict[name], open("database_" + name + ".pkl", "wb"))
                return redirect(url_for("submit"))
            else:
                # this is last example, take to survey page
                user_dict[name].test_index = user_dict[name].test_index + 1
                return redirect(url_for("survey"))
            # return show_test_example(user_dict[name].test_index, name)

    if request.method == "GET":
        # User wants to view example in test phase
        # Please refer to submit function
        if user_dict[name].test_not_initialised():
            user_dict[name].set_test_time_left("3m")
            user_dict[name].set_test_time_left_sec(180)
            user_dict[name].set_test_starttime(time.time())
        else:
            start_time = user_dict[name].get_test_starttime()
            cur_time = time.time()
            user_dict[name].set_test_time_left_sec(
                180 - float(cur_time) + float(start_time)
            )
            timerMin = int(user_dict[name].get_test_time_left_sec() / 60)
            timerSec = int(user_dict[name].get_test_time_left_sec() - (60 * timerMin))
            user_dict[name].set_test_time_left(
                str(timerMin) + "m" + str(timerSec) + "s"
            )
        return show_test_example(
            name,
            flipped=user_dict[name].test_index
            in user_dict[name].test_examples_flipped_dict,
        )


if not app.debug:
    file_handler = FileHandler("error.log")
    file_handler.setFormatter(
        Formatter("%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]")
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info("errors")

# ----------------------------------------------------------------------------#
# Launch.
# ----------------------------------------------------------------------------#

# Default port:
if __name__ == "__main__":
    # app.run()
    if args.explanation_type == "integrated_gradients":
        port = int(os.environ.get("PORT", 4000))
    elif args.explanation_type == "log_reg":
        if args.control:
            port = int(os.environ.get("PORT", 3050))
        else:
            port = int(os.environ.get("PORT", 3000))
    elif args.explanation_type == "none":
        port = int(os.environ.get("PORT", 3050))
    else:
        port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, threaded=True)


# Or specify port manually:
"""
if __name__ == '__main__':
  port = int(os.environ.get('PORT', 5000))
  app.run(host = '0.0.0.0', port = port)
"""
