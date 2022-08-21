import argparse
import os
import random
import sys
import time

sys.path.append("../src")
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils import data
from tqdm import tqdm
from transformers import (AdamW, BertForSequenceClassification, BertTokenizer,
                          get_linear_schedule_with_warmup)

import wandb
from data_utils import HotelReviewsProcessor, ReviewsDataset
from parsing_utils import init_parser


def set_seed(seed):
    # set seed for all possible avenues of stochasticity
    np.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_tokenizer(args):
    suffix = "cased" if args.case_sensitive else "uncased"
    return BertTokenizer.from_pretrained("bert-base-" + suffix)


def load_model(args):
    suffix = "cased" if args.case_sensitive else "uncased"
    # FIXME: we need BertForSequenceClassification here
    return BertForSequenceClassification.from_pretrained(
        "bert-base-" + suffix, output_attentions=True
    )


def init_optimizer(args, model, num_training_steps):
    # initializes and returns the optimizer & scheduler
    named_params = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in named_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay_finetune,
        },
        {
            "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


def evaluate(model, predict_dataloader, epoch, dataset_name, args):
    """
    evaluate the model

    Parameters
    ----------
    model : torch.nn.Module
        Bert-type model
    predict_dataloader : torch.utils.data.DataLoader
        iterator for the dataset
    epoch: int
        the epoch value after which we evaluate
    dataset_name : str
        which split: train/dev/test
    args : ArgumentParser
        argument parser

    Returns
    -------
    dict
        evaluation statistics, precision, recall, f1, accuracy
    """

    model.eval()

    # prediction stats
    pred_labels = []
    gold_labels = []
    st_time = time.time()

    for batch in predict_dataloader:
        input_ids, attention_mask, label_ids = batch
        output = model(input_ids, attention_mask, labels=label_ids)
        pred_scores = output[1]  # predictions: batch_size x num_labels
        predictions = torch.argmax(pred_scores, dim=-1)
        pred_labels.extend(predictions.tolist())
        gold_labels.extend(label_ids.tolist())

    # print prediction results ...
    p, r, f1, _ = precision_recall_fscore_support(
        gold_labels, pred_labels, average="binary"
    )
    accuracy = accuracy_score(gold_labels, pred_labels)

    et_time = time.time()
    print(
        "Epoch: %d | dataset: %s | Acc: %.2f | P: %.2f | R: %.2f | F1: %.2f | T: %.3f mins"
        % (
            epoch,
            dataset_name,
            100.0 * accuracy,
            100.0 * p,
            100.0 * r,
            100.0 * f1,
            (et_time - st_time) / 60.0,
        )
    )
    print("--------------------------------------------------------------")

    # for name, val in [("precision", p), ("recall", r), ("accuracy", accuracy), ("F1", f1)]:

    results = {
        "precision": p,
        "recall": r,
        "f1": f1,
        "accuracy": accuracy,
    }

    for key, val in results.items():
        wandb.log({dataset_name + " " + key: val})

    return results


def main():
    args = init_parser().parse_args()
    wandb.init(project="games-for-interpretability", entity=args.user)
    wandb.config.update(args)
    set_seed(args.seed)
    tokenizer = load_tokenizer(args)

    # load data
    dataset_processor = HotelReviewsProcessor()
    train_examples = dataset_processor.get_train_examples(args.data_dir)
    train_examples = train_examples[: args.num_train_examples]
    dev_examples = dataset_processor.get_dev_examples(args.data_dir)
    test_examples = dataset_processor.get_test_examples(args.data_dir)

    # instantiate datasets
    train_dataset = ReviewsDataset(train_examples, tokenizer, args.max_seq_len)
    dev_dataset = ReviewsDataset(dev_examples, tokenizer, args.max_seq_len)
    test_dataset = ReviewsDataset(test_examples, tokenizer, args.max_seq_len)
    # pickle.dump(test_dataset, open("test_dataloader.pkl","wb"))

    # instantiate data loaders
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=ReviewsDataset.pad,
        worker_init_fn=np.random.seed(args.seed),
    )

    dev_dataloader = data.DataLoader(
        dataset=dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ReviewsDataset.pad,
        worker_init_fn=np.random.seed(args.seed),
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ReviewsDataset.pad,
        worker_init_fn=np.random.seed(args.seed),
    )

    model = load_model(args)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model.cuda()

    num_training_steps = (
        len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    )
    optimizer, scheduler = init_optimizer(args, model, num_training_steps)
    best_dev_accuracy = 0.0

    for epoch in range(args.epochs):
        # start of training loop
        st_time = time.time()
        for step, batch in enumerate(train_dataloader):
            input_ids, attention_mask, label_ids = batch
            output = model(input_ids, attention_mask, labels=label_ids)
            # output is a 2-tuple, where the first item is loss, and the second is
            # predictions: batch_size x num_labels
            loss = output[0]  # negative log likelihood loss (averaged over batch)

            loss /= args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # time to update weights ...
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            wandb.log({"Train Loss": loss.item()})

        print("Training time = %.3f mins" % ((time.time() - st_time) / 60.0))
        evaluate(model, train_dataloader, epoch + 1, "train", args)
        dev_results = evaluate(model, dev_dataloader, epoch + 1, "dev", args)
        evaluate(model, test_dataloader, epoch + 1, "test", args)

        if args.save is not None and dev_results["accuracy"] > best_dev_accuracy:
            best_dev_accuracy = dev_results["accuracy"]
            torch.save(model.state_dict(), args.save)
            print("Model saved after epoch ", epoch + 1)


if __name__ == "__main__":
    main()
