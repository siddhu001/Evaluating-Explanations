import random
from datetime import datetime

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from temperature_scaling import ModelWithTemperature
from utils import strip_tags


class Log:
    def __init__(self, val):
        (dt_now, prob, user_feedback, overall_feedback, output_text) = val
        self.time = dt_now
        self.prob = prob
        self.user_feedback = user_feedback
        self.overall_feedback = overall_feedback
        self.output_text = output_text
        self.output_text_no_tags = (
            strip_tags(output_text).strip().replace(u"\xa0", u" ")
        )


class Example:
    def __init__(self, text, index, string_input=False):
        if string_input:
            self.text = text
        else:
            self.text = text.content
        self.index = index
        self.original_score = None
        self.max_change_score = None
        self.previous_score = None
        self.user_guesses = None
        self.log_array = []
        self.time_left = "3m"
        self.time_left_sec = "180"
        self.starttime = None


class User:
    def __init__(self, name, train_examples, test_examples, args, testing_inputs):
        self.name = name
        now = datetime.now()
        dt_now = now.strftime("%d/%m/%Y %H:%M:%S")
        self.dt_game_start = dt_now
        self.train_examples_arr = []
        for eg_index in range(len(train_examples)):
            eg = train_examples[eg_index]
            self.train_examples_arr.append(Example(eg, eg_index))
        self.test_examples_arr = []
        for eg_index in range(len(testing_inputs)):
            eg = testing_inputs[eg_index]
            self.test_examples_arr.append(Example(eg, eg_index, string_input=True))
        self.train_index = 0
        self.test_index = 0
        self.correct_guesses = 0
        self.test_correct_guesses = 0
        self.examples_flipped_dict = {}
        self.test_examples_flipped_dict = {}
        self.show_test = False
        self.explanation_type = args.explanation_type
        self.survey_response_dict = dict()
        self.coupon_code = None
        self.bonus = "0.0"
        self.input_ids = None
        self.attention_mask = None
        self.input_example = None
        self.baseline_example = None
        self.attention_mask_example = None
        if args.explanation_type != "log_reg":

            orig_model = BertForSequenceClassification.from_pretrained(
                "bert-base-cased", output_attentions=True
            )
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            self.model = ModelWithTemperature(orig_model)
            is_cuda = torch.cuda.is_available()
            device = torch.device("cuda") if is_cuda else torch.device("cpu")
            print("Device in use: ", device)
            self.model.load_state_dict(
                torch.load("model_dumps/bert_cased_calibrated.pth", map_location=device)
            )
            self.model.to(device)
            self.model.eval()

    def get_train_time_left(self):
        return self.train_examples_arr[self.train_index].time_left

    def set_train_time_left(self, time):
        self.train_examples_arr[self.train_index].time_left = time

    def get_train_time_left_sec(self):
        return self.train_examples_arr[self.train_index].time_left_sec

    def set_train_time_left_sec(self, time):
        self.train_examples_arr[self.train_index].time_left_sec = time

    def get_train_starttime(self):
        return self.train_examples_arr[self.train_index].starttime

    def set_train_starttime(self, time):
        self.train_examples_arr[self.train_index].starttime = time

    def train_guess_state(self):
        return self.train_examples_arr[self.train_index].original_score is None

    def train_not_initialised(self):
        return self.train_examples_arr[self.train_index].starttime is None

    def get_train_user_guesses(self):
        return self.train_examples_arr[self.train_index].user_guesses

    def set_train_user_guesses(self, val):
        self.train_examples_arr[self.train_index].user_guesses = val

    def get_train_original_score(self):
        return self.train_examples_arr[self.train_index].original_score

    def set_train_original_score(self, val):
        self.train_examples_arr[self.train_index].original_score = val

    def get_train_max_change_score(self):
        return self.train_examples_arr[self.train_index].max_change_score

    def set_train_max_change_score(self, val):
        self.train_examples_arr[self.train_index].max_change_score = val

    def get_train_previous_score(self):
        return self.train_examples_arr[self.train_index].previous_score

    def set_train_previous_score(self, val):
        self.train_examples_arr[self.train_index].previous_score = val

    def get_train_log_array(self):
        return self.train_examples_arr[self.train_index].log_array[-1]

    def set_train_log_array(self, val):
        self.train_examples_arr[self.train_index].log_array.append(Log(val))

    def get_train_input(self):
        return self.train_examples_arr[self.train_index].text

    def get_test_time_left(self):
        return self.test_examples_arr[self.test_index].time_left

    def set_test_time_left(self, time):
        self.test_examples_arr[self.test_index].time_left = time

    def get_test_time_left_sec(self):
        return self.test_examples_arr[self.test_index].time_left_sec

    def set_test_time_left_sec(self, time):
        self.test_examples_arr[self.test_index].time_left_sec = time

    def get_test_starttime(self):
        return self.test_examples_arr[self.test_index].starttime

    def set_test_starttime(self, time):
        self.test_examples_arr[self.test_index].starttime = time

    def test_guess_state(self):
        return self.test_examples_arr[self.test_index].original_score is None

    def test_not_initialised(self):
        return self.test_examples_arr[self.test_index].starttime is None

    def get_test_user_guesses(self):
        return self.test_examples_arr[self.test_index].user_guesses

    def set_test_user_guesses(self, val):
        self.test_examples_arr[self.test_index].user_guesses = val

    def get_test_original_score(self):
        return self.test_examples_arr[self.test_index].original_score

    def set_test_original_score(self, val):
        self.test_examples_arr[self.test_index].original_score = val

    def get_test_max_change_score(self):
        return self.test_examples_arr[self.test_index].max_change_score

    def set_test_max_change_score(self, val):
        self.test_examples_arr[self.test_index].max_change_score = val

    def get_test_previous_score(self):
        return self.test_examples_arr[self.test_index].previous_score

    def set_test_previous_score(self, val):
        self.test_examples_arr[self.test_index].previous_score = val

    def get_test_log_array(self):
        return self.test_examples_arr[self.test_index].log_array[-1]

    def set_test_log_array(self, val):
        self.test_examples_arr[self.test_index].log_array.append(Log(val))

    def test_input(self):
        return self.test_examples_arr[self.test_index].text

    def calculate_bonus(self):
        correct_guesses = self.test_correct_guesses + self.correct_guesses
        correct_flips = len(self.examples_flipped_dict) + len(
            self.test_examples_flipped_dict
        )
        self.bonus = str(
            round((0.1 * int(correct_guesses)) + (0.2 * int(correct_flips)), 1)
        )
