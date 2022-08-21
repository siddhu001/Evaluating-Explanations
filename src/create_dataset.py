import os
from random import shuffle

from tqdm import tqdm

from data_utils import (collect_amazon_reviews, collect_bios,
                        collect_headlines, collect_reviews)
from parsing_utils import init_parser


def write_reviews(reviews, filename):
    with open(filename, "w") as fw:
        for review in tqdm(reviews):
            sentiment = "positive" if review.sentiment else "negative"
            truthful = "truthful" if review.truthfulness else "fake"
            content = review.content
            delimiter = " ||| "
            fw.write(truthful + delimiter + sentiment + delimiter + content + "\n")
    return


def main():
    args = init_parser().parse_args()
    print(args.task)
    if args.task == "review":
        reviews = collect_reviews()
        shuffle(reviews)
        assert len(reviews) == 1600
        train_reviews = reviews[:1000]
        dev_reviews = reviews[1000:1200]
        test_reviews = reviews[1200:]

        path = "../datasets/hotel_reviews/"
    elif args.task == "amazon":
        reviews = collect_amazon_reviews()
        shuffle(reviews)
        # assert len(reviews) == 1600
        train_reviews = reviews[: (int(0.6 * len(reviews)))]
        dev_reviews = reviews[(int(0.6 * len(reviews))) : (int(0.8 * len(reviews)))]
        test_reviews = reviews[(int(0.8 * len(reviews))) :]
        path = "../datasets/amazon_reviews/"
    elif args.task == "sarcasm":
        reviews = collect_headlines()
        shuffle(reviews)
        assert len(reviews) == 28619
        train_reviews = reviews[: (int(0.6 * len(reviews)))]
        dev_reviews = reviews[(int(0.6 * len(reviews))) : (int(0.8 * len(reviews)))]
        test_reviews = reviews[(int(0.8 * len(reviews))) :]
        path = "../datasets/sarcastic_headlines/"
    elif args.task == "bios":
        reviews = collect_bios()
        shuffle(reviews)
        print(len(reviews))
        train_reviews = reviews[: (int(0.6 * len(reviews)))]
        dev_reviews = reviews[(int(0.6 * len(reviews))) : (int(0.8 * len(reviews)))]
        test_reviews = reviews[(int(0.8 * len(reviews))) :]
        path = "../datasets/bios/"

    write_reviews(train_reviews, os.path.join(path, "train.txt"))
    write_reviews(dev_reviews, os.path.join(path, "dev.txt"))
    write_reviews(test_reviews, os.path.join(path, "test.txt"))


if __name__ == "__main__":
    main()
