import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils import data
from transformers import BertForSequenceClassification, BertTokenizer

sys.path.append("../src")

from data_utils import HotelReviewsProcessor, ReviewsDataset
from parsing_utils import init_parser
from temperature_scaling import ModelWithTemperature


def load_tokenizer(args):
    suffix = "cased" if args.case_sensitive else "uncased"
    return BertTokenizer.from_pretrained("bert-base-" + suffix)


def load_model(args):
    suffix = "cased" if args.case_sensitive else "uncased"
    model = BertForSequenceClassification.from_pretrained("bert-base-" + suffix)
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if is_cuda else torch.device("cpu")
    model.load_state_dict(torch.load(args.load, map_location=device))
    model.to(device)
    return model


def main():
    args = init_parser().parse_args()

    # load data
    dataset_processor = HotelReviewsProcessor()
    dev_examples = dataset_processor.get_dev_examples(args.data_dir)
    tokenizer = load_tokenizer(args)
    dev_dataset = ReviewsDataset(dev_examples, tokenizer, args.max_seq_len)

    # instantiate data loaders
    dev_dataloader = data.DataLoader(
        dataset=dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ReviewsDataset.pad,
        worker_init_fn=np.random.seed(args.seed),
    )

    # load model
    model = load_model(args)
    model.eval()

    # calibrate model
    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(dev_dataloader)
    # save model
    if args.save is not None:
        torch.save(scaled_model.state_dict(), args.save)


if __name__ == "__main__":
    main()
