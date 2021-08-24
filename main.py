import argparse
from datetime import datetime

import pandas as pd

from src.train import train

EPOCHS = 1000
DEVICE = None
BATCH_SIZE = 128
SUMMARY_PATH = "training_summaries/" + str(datetime.now())
N_SUMMARY = 100
N_EVAL = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0

if __name__ == "__main__":
    print("Summary path:", SUMMARY_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Number of samples per batch"
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default=SUMMARY_PATH,
        help="Path to store tensorboard logs",
    )
    parser.add_argument(
        "--n_summary",
        type=int,
        default=N_SUMMARY,
        help="Number steps between each training summary log to tensorboard",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=N_EVAL,
        help="Number steps between each evaluation",
    )
    parser.add_argument(
        "--layer_sizes",
        type=str,
        default="256,128,2",
        help="Sizes of hidden layers separated by commas (including last layer)",
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=None,
        help="Probability to drop values in dropout layers",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=WEIGHT_DECAY,
        help="L2 regularization parameter",
    )
    parser.add_argument(
        "--device", type=str, default=DEVICE, help="Device to run training on"
    )
    args = parser.parse_args()

    # Load data
    data = pd.read_csv("data_new/final_merged.csv")
    data = data.dropna(subset=["image_url", "encoded_text", "views"])
    data = data.reset_index(drop=True)

    train_df = data.iloc[: (int(len(data) * 0.9))]
    train_df = train_df.sample(frac=1)
    val_df = data.iloc[(int(len(data) * 0.9)) :]
    val_df = val_df.sample(frac=1)
    val_df.reset_index(inplace=True)
    train_df.reset_index(inplace=True)

    layer_sizes = list(map(lambda s: int(s), args.layer_sizes.split(",")))

    train(
        train_df,
        val_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        summary_path=args.summary_path,
        n_summary=args.n_summary,
        n_eval=args.n_eval,
        layer_sizes=layer_sizes,
        dropout_prob=args.dropout_prob,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
    )
