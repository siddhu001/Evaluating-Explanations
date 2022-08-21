import json
import os
import pickle

import numpy as np
import torch
from datasets import load_dataset
from torch.utils import data


class Review:
    def __init__(self, content, sentiment, truthfulness):
        """
        init function

        Parameters
        ----------
        content : str
            content of the review
        sentiment : int
            1 for positive, and 0 for negative
        truthfulness: int
            1 for genuine and 0 for fake
        """
        self.content = content
        self.sentiment = sentiment
        self.truthfulness = truthfulness


class InputFeatures(object):
    """A single set of features corresponding to an example in the dataset."""

    def __init__(self, tokens, input_ids, attention_mask, label_id):
        """
        init function

        Parameters
        ----------
        tokens : list of tokens
        input_ids : list
            input ids of tokens
        attention_mask : list
            attention mask of input which tells us which tokens to ignore for self attention.
            attention_mask[i] = 1 denotes that token participates in self-attention,
            attention_mask[i] = 0 implies otherwise
        label_id : int
            associated label of the sentence
        """
        self.tokens = tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_id = label_id


def collect_reviews(path="../datasets/hotel_reviews/original_version"):
    """
    aggregates all the hotel reviews from all the dataset folders

    Parameters
    ----------
    path : str, optional
        folder base path, by default "../datasets/hotel_reviews/original_version"

    Returns
    -------
    list
        list of all the reviews, each of which is an object of the class Review
    """

    all_reviews = []
    for polarity in ["negative_polarity", "positive_polarity"]:
        for status in ["truthful_from_Web", "deceptive_from_MTurk"]:
            for fold in range(1, 5 + 1):
                current_path = os.path.join(path, polarity, status, "fold" + str(fold))
                # TODO: check if this file actually exists
                files = os.listdir(current_path)
                for file in files:
                    file_path = os.path.join(current_path, file)
                    content = open(file_path).readlines()[0].strip()
                    sentiment = 1 if "positive" in polarity else 0
                    truthfulness = 1 if "truthful" in status else 0
                    all_reviews.append(Review(content, sentiment, truthfulness))
    return all_reviews


def collect_amazon_reviews():
    """
    aggregates all the hotel reviews from all the dataset folders

    Parameters
    ----------
    path : str, optional
        folder base path, by default "../datasets/hotel_reviews/original_version"

    Returns
    -------
    list
        list of all the reviews, each of which is an object of the class Review
    """
    dataset = load_dataset("amazon_polarity")
    content_arr = dataset["train"]["content"]
    label_arr = dataset["train"]["label"]
    all_reviews = []
    for count in range(len(content_arr)):
        print(count)
        content = content_arr[count].strip()
        sentiment = label_arr[count]
        truthfulness = sentiment
        all_reviews.append(Review(content, sentiment, truthfulness))
    content_arr = dataset["test"]["content"]
    label_arr = dataset["test"]["label"]
    for count in range(len(content_arr)):
        print(count)
        content = content_arr[count].strip()
        sentiment = label_arr[count]
        truthfulness = sentiment
        all_reviews.append(Review(content, sentiment, truthfulness))
    return all_reviews


def collect_bios(path="/projects/ogma2/users/siddhana/bias_in_bios/bias_in_bios/"):
    """
    aggregates all the hotel reviews from all the dataset folders

    Parameters
    ----------
    path : str, optional
        folder base path, by default "../datasets/hotel_reviews/original_version"

    Returns
    -------
    list
        list of all the reviews, each of which is an object of the class Review
    """
    bio_dict = pickle.load(open(path + "BIOS.pkl", "rb"))
    all_reviews = []
    for bio in bio_dict:
        # print(count)
        if bio["title"] == "attorney" or bio["title"] == "paralegal":
            content = bio["bio"].strip()
            sentiment = 1 if bio["title"] == "attorney" else 0
            truthfulness = 1 if bio["title"] == "attorney" else 0
            all_reviews.append(Review(content, sentiment, truthfulness))
    return all_reviews


def collect_headlines(path="../datasets/Sarcasm_Headlines_Dataset.json"):
    """
    aggregates all the hotel reviews from all the dataset folders

    Parameters
    ----------
    path : str, optional
        folder base path, by default "../datasets/hotel_reviews/original_version"

    Returns
    -------
    list
        list of all the reviews, each of which is an object of the class Review
    """
    all_reviews = []
    for line in open(path, "r"):
        headline = json.loads(line)
        content = headline["headline"].strip()
        sentiment = 1
        truthfulness = 1 if headline["is_sarcastic"] else 0
        all_reviews.append(Review(content, sentiment, truthfulness))
    return all_reviews


class DataProcessor(object):
    """Base class for data converters for data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of examples for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of examples for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_data(self, filename):
        """Reads the data"""
        raise NotImplementedError()


class HotelReviewsProcessor(DataProcessor):
    def __init__(self):
        super(HotelReviewsProcessor, self).__init__()
        pass

    def get_train_examples(self, data_dir):
        return self.get_data(os.path.join(data_dir, "train.txt"))

    def get_dev_examples(self, data_dir):
        return self.get_data(os.path.join(data_dir, "dev.txt"))

    def get_test_examples(self, data_dir):
        return self.get_data(os.path.join(data_dir, "test.txt"))

    def get_download_examples(self, data_dir):
        return self.get_data(os.path.join(data_dir, "download.txt"))

    def get_download_with_prediction_examples(self, data_dir):
        return self.get_data(os.path.join(data_dir, "download_pred_filter.txt"))

    def get_error_examples(self, data_dir):
        return self.get_data(os.path.join(data_dir, "error.txt"))

    def get_correct_examples(self, data_dir):
        return self.get_data(os.path.join(data_dir, "correct_review.txt"))

    def get_labels(self):
        return ["0", "1"]

    def _convert_input_to_review(self, line):
        truthfulness, sentiment, content = line.strip().split(" ||| ")
        truthfulness = 1 if "truthful" in truthfulness else 0
        sentiment = 1 if "positive" in sentiment else 0
        return Review(content, sentiment, truthfulness)

    def get_data(self, filename):
        # get all the lines from the file and convert them to reviews
        raw_lines = open(filename).readlines()
        reviews = [self._convert_input_to_review(line) for line in raw_lines]
        return reviews


def input_to_features(example, tokenizer, max_seq_len):
    """
    converts the input review to input features

    Parameters
    ----------
    example : Review
        an example Review
    tokenizer : ?
        tokenizer which has some basic functionality to tokenize, etc.
    max_seq_len : int
        maximum sequence length

    Returns
    -------
    InputFeature
        extracted features from the input
    """

    content = example.content
    label = (
        example.truthfulness
    )  # the label can also be sentiment (in case we are doing sentiment)

    assert max_seq_len >= 2
    tokens = ["[CLS]"] + tokenizer.tokenize(content)[: max_seq_len - 2] + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(tokens)

    return InputFeatures(tokens, input_ids, attention_mask, label)


class ReviewsDataset(data.Dataset):
    def __init__(self, examples, tokenizer, max_seq_len):
        super(ReviewsDataset, self).__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        features = input_to_features(
            self.examples[idx], self.tokenizer, self.max_seq_len
        )

        return features.input_ids, features.attention_mask, features.label_id

    @classmethod
    def pad(cls, batch):
        # class method to pad sequences

        is_cuda = torch.cuda.is_available()
        long_type = torch.cuda.LongTensor if is_cuda else torch.LongTensor

        seq_lens = [len(sample[0]) for sample in batch]
        max_seq_len = max(seq_lens)

        # lambda function to pad the batch with zeros
        f_pad = lambda x, seqlen: [
            sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch
        ]
        # lambda function to index the batch
        f_index = lambda x: [sample[x] for sample in batch]

        input_ids_list = torch.Tensor(f_pad(0, max_seq_len)).type(long_type)
        attention_mask_list = torch.Tensor(f_pad(1, max_seq_len)).type(long_type)
        label_ids_list = torch.Tensor(f_index(2)).type(long_type)

        return input_ids_list, attention_mask_list, label_ids_list
