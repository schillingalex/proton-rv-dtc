import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk generation of features given JSON metadata files")
    parser.add_argument("-f", "--features-file", dest="features_file", type=str, required=True,
                        help="Features file to split into train/val/test")
    parser.add_argument("--val-size", dest="val_size", type=float, default=0.125,
                        help="Fraction of training data to use for the validation set")
    parser.add_argument("--test-size", dest="test_size", type=float, default=0.2,
                        help="Fraction of data to use for the test set")
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=0,
                        help="Seed for random number generator for reproducibility")
    args = parser.parse_args()

    X = pd.read_csv(args.features_file)
    X_train, X_test = train_test_split(X, test_size=args.test_size, random_state=args.seed)
    X_train, X_val = train_test_split(X_train, test_size=args.val_size, random_state=args.seed)

    print("Total samples:", len(X))
    print("Train/val/test:", len(X_train), len(X_val), len(X_test))

    train_file = os.path.splitext(args.features_file)[0] + "_train.csv"
    val_file = os.path.splitext(args.features_file)[0] + "_val.csv"
    test_file = os.path.splitext(args.features_file)[0] + "_test.csv"
    X_train.to_csv(train_file, index=False)
    X_val.to_csv(val_file, index=False)
    X_test.to_csv(test_file, index=False)
