from html.parser import HTMLParser
from io import StringIO

import numpy as np

from analysis_utils import Utils


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def filter_reviews(reviews, limit=125):
    # filter dataset to only reviews with word length
    # less than limit
    new_reviews = []
    for review in reviews:
        if len(review.content.split(" ")) < limit:
            new_reviews.append(review)
    return new_reviews


def find_time_diff(starting_time, editing_time):
    # print(starting_time)
    # print(editing_time)
    starting_day = starting_time.split(" ")[0]
    editing_day = editing_time.split(" ")[0]
    minutes = 0
    starting_onlytime = starting_time.split(" ")[1].strip().split(":")
    editing_onlytime = editing_time.split(" ")[1].strip().split(":")
    starting_hour = int(starting_onlytime[0])
    editing_hour = int(editing_onlytime[0])
    starting_minute = int(starting_onlytime[1])
    editing_minute = int(editing_onlytime[1])
    starting_sec = int(starting_onlytime[2])
    editing_sec = int(editing_onlytime[2])
    if starting_hour != editing_hour:
        minutes = (editing_hour - starting_hour) * 60
    if starting_minute != editing_minute:
        minutes += editing_minute - starting_minute
    seconds = minutes * 60
    if starting_sec != editing_sec:
        seconds += editing_sec - starting_sec
    if starting_day != editing_day:
        print(starting_day)
        print(editing_day)
        print("error")
        seconds += 3600 * 24
    return seconds


def compute_edit_dist(input_text, edited_text):
    num_words = len(input_text.split(" "))
    (ins_edit, subs_edit, del_edit) = Utils.editops(
        input_text.split(" "), edited_text.split(" ")
    )
    return (
        len(ins_edit) / num_words,
        len(subs_edit) / num_words,
        len(del_edit) / num_words,
    )


def compute_bootstrapped_confidence_interval(
    results_arr, num_bootstraps=10000, round_index=2
):
    bootstrap_results_arr = []
    for bootstrap_index in range(num_bootstraps):
        bootstrap_results_arr.append(
            np.mean(np.random.choice(results_arr, len(results_arr)))
        )
    print("Lower Bound")
    print(round(np.percentile(bootstrap_results_arr, 2.5), round_index))
    print("Upper Bound")
    print(round(np.percentile(bootstrap_results_arr, 97.5), round_index))
