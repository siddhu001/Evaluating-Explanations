import argparse


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", default=123, type=int, help="random seed for torch, numpy, random"
    )
    parser.add_argument(
        "--epochs", default=5, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="learning rate"
    )
    parser.add_argument("--data_dir", default="", type=str, help="data directory")
    parser.add_argument(
        "--case_sensitive",
        action="store_true",
        help="have case sensitive tokenizer (default is no)",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="max sequence length"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient_accumulation_steps",
    )
    parser.add_argument(
        "--weight_decay_finetune",
        type=float,
        default=1e-5,
        help="weight decay finetune",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument("--save", default=None, type=str, help="path to save model")
    parser.add_argument(
        "--load", default=None, type=str, help="path to load prediction model"
    )
    parser.add_argument(
        "--evaluate_every",
        default=1000,
        type=int,
        help="evaluate every xx number of steps",
    )
    parser.add_argument(
        "--print_every",
        default=100,
        type=int,
        help="print loss every xx number of steps",
    )
    parser.add_argument(
        "--num_train_examples",
        default=100000000,
        type=int,
        help="number of training examples",
    )
    parser.add_argument(
        "--dump_explanations",
        action="store_true",
        help="whether to dump explanations or not",
    )
    parser.add_argument(
        "--explanation_type",
        default="attention",
        choices=[
            "attention",
            "grad_norm",
            "grad_times_inp",
            "integrated_gradients",
            "lime",
        ],
        help="choice of explanations to dump ...",
    )
    parser.add_argument(
        "--user", default="danish037", type=str, help="username for wandb project"
    )
    parser.add_argument(
        "--top_k", default=20, type=int, help="number of top features to analyse"
    )
    parser.add_argument("--task", default="review", type=str, help="task to run game")
    return parser
