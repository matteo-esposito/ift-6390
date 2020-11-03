#!/usr/bin/env python
# coding: utf-8

# # Arxiv Paper Classification
# ### Matteo Esposito

import re
import sys

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


# ## Preliminary

def split_training_set(df, ratio=0.7):
    """Splits training set into train and validation at a ratio of 70/30"""
    df.sample(frac=1)
    train = df[: int(ratio * df.shape[0])]
    validation = df[int(ratio * df.shape[0]):]
    return train, validation


# Data Cleanup
def clean(df, t):
    """
    Clean dataset.
    """
    df[t] = df[t].apply(lambda x: re.sub("[0-9]", "", x))
    df[t] = df[t].apply(lambda x: re.sub("\[[^]]*]", "", x))
    df[t] = df[t].apply(lambda x: re.sub("<$\.*?>", "", x))
    df[t] = df[t].apply(lambda x: re.sub("\n", " ", x))
    df[t] = df[t].apply(lambda x: x.lower())
    df[t] = df[t].apply(lambda x: x.strip())
    df[t] = df[t].apply(lambda x: re.sub("[^A-Za-z0-9]+", " ", x))

    # remove stop words
    stopwords = [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "ain",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "couldn",
        "couldn't",
        "d",
        "did",
        "didn",
        "didn't",
        "do",
        "does",
        "doesn",
        "doesn't",
        "doing",
        "don",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn",
        "hadn't",
        "has",
        "hasn",
        "hasn't",
        "have",
        "haven",
        "haven't",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "just",
        "ll",
        "m",
        "ma",
        "me",
        "mightn",
        "mightn't",
        "more",
        "most",
        "mustn",
        "mustn't",
        "my",
        "myself",
        "needn",
        "needn't",
        "no",
        "nor",
        "not",
        "now",
        "o",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "re",
        "s",
        "same",
        "shan",
        "shan't",
        "she",
        "she's",
        "should",
        "should've",
        "shouldn",
        "shouldn't",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "that'll",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "ve",
        "very",
        "was",
        "wasn",
        "wasn't",
        "we",
        "were",
        "weren",
        "weren't",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
        "y",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "could",
        "he'd",
        "he'll",
        "he's",
        "here's",
        "how's",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "let's",
        "ought",
        "she'd",
        "she'll",
        "that's",
        "there's",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "what's",
        "when's",
        "where's",
        "who's",
        "why's",
        "would",
    ]
    df[t] = df[t].apply(
        lambda text: ["".join(w) for w in text.split(" ") if w not in stopwords]
    )

    # Convert list back to string
    newabs = [" ".join(map(str, abstract_list)) for abstract_list in df[t]]
    df[t] = newabs
    return df


def get_prediction_error(y_pred, y_true):
    """
    Function to be used to get error in predictions.
    (Only used while developping the models and creating comparison metrics.)
    """
    total_points = y_true.shape[0]
    correctly_classified_points = sum(y_pred == y_true)
    return round(1 - correctly_classified_points / total_points, 2)


# ## Q4 - Naive Bayes Baselines

# ### Random Classifier

def random_classification(out, train, test):
    random_submission = pd.DataFrame({"Id": [], "Category": []})
    random_submission["Id"] = test["Id"]
    random_submission["Category"] = np.random.choice(
        train["Category"].unique(), test.shape[0]
    )
    random_submission.to_csv(out, index=False)
    print(f"Random classification done and output to {out}")


# ### Naive Bayes Classifier using Bag of Words

class DataFormatter:
    def __init__(self, train, test, validation):
        self.train = train
        self.test = test
        self.validation = validation
        self.vocab = []

    def get_vocab(self):
        """Create a vocabulary of the unique words in all the abstracts provided in the training set."""
        for wordstring in self.train:
            for word in wordstring.split(" "):
                if word not in self.vocab:
                    self.vocab.append(word)


    def create_bow(self):
        """Create bag of words (i.e. binarize data for nb classification)."""
        returnarrs = []
        for _, df in enumerate([self.train, self.test, self.validation]):
            bag_of_words = []
            for i in range(df.shape[0]):
                temp_bow = {}
                abstract = df.iloc[i].split()

                # Initialize
                for word in self.vocab:
                    temp_bow[word] = 0

                # Populate with 1 in case where the word is in the provided test case.
                for word in abstract:
                    if word in temp_bow.keys():
                        temp_bow[word] = 1

                bag_of_words.append(temp_bow)

            for i in range(df.shape[0]):
                bag_of_words[i] = list(bag_of_words[i].values())

            returnarrs.append(np.array(bag_of_words))

        return returnarrs


class BernoulliNB:
    def __init__(self, X, y, alpha: float = 1.0):
        """
        Bernoulli Naive Bayes Classifier Constructor
        """
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.num_features = X.shape[1]
        self.conditionals = np.zeros((self.num_classes, self.num_features))
        self.alpha = alpha

    def fit(self):
        """
        Get all conditional probabilities to be used in the posterior probability calculation in get_posteriors().
        """
        for cat_idx, cat in enumerate(self.classes):
            self.conditionals[cat_idx] = (self.X[self.y == cat].sum(axis=0) + self.alpha * np.ones(
                self.num_features)) / (self.X[self.y == cat].shape[0] + self.alpha * self.num_classes)

    def get_posteriors(self, test_case):
        """
        Apply the probability function for a Bernoulli random variable.
        """
        return np.prod((np.ones((self.num_classes, self.num_features)) - self.conditionals) * (
                    np.ones((self.num_classes, self.num_features)) - test_case) + self.conditionals * test_case, axis=1)

    def generate_matrix_prediction(self, unseen_data):
        """
        Take a maximum of the posterior probabilities of every class for every observation. Here we apply the posterior
        probability function for a single case for every row/observation
        """
        def predict_single_case(test_case):
            return np.argmax(self.get_posteriors(test_case))
        return np.apply_along_axis(predict_single_case, axis=1, arr=unseen_data)


def main():
    # Get command line args
    ALGO = sys.argv[1].lower()
    TRAIN = sys.argv[2]
    TEST = sys.argv[3]
    OUTPATH = sys.argv[4]

    print(f"Algo: {ALGO}\nTrain: {TRAIN}\nTest: {TEST}\nOutpath: {OUTPATH}\n")

    # Get data and split it
    train_full = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    train, validation = split_training_set(train_full)

    print(">> Data read and split")

    # Clean/preprocess data.
    train = clean(train, "Abstract")
    validation = clean(validation, "Abstract")
    test = clean(test, "Abstract")

    print(">> Data processed")

    # Proceed with classification
    if ALGO == "random":

        # Run random classifier.
        print(">> Running Random Classifier")
        random_classification(train=train, test=test, out=OUTPATH)

    # Run Naive Bayes Classifier
    elif ALGO == "nbbernoulli":

        print(">> Running Bernoulli Naive Bayes Classifier")

        # Format data into acceptable format (binary)
        d = DataFormatter(
            train=train["Abstract"],
            test=test["Abstract"],
            validation=validation["Abstract"],
        )
        d.get_vocab()
        table_output = d.create_bow()
        train_abstracts, test_abstracts, validation_abstracts = (
            table_output[0],
            table_output[1],
            table_output[2],
        )

        # Get predictions and map integer output to string categories.
        cat_mapping = {
            0: "astro-ph",
            1: "astro-ph.CO",
            2: "astro-ph.GA",
            3: "astro-ph.SR",
            4: "cond-mat.mes-hall",
            5: "cond-mat.mtrl-sci",
            6: "cs.LG",
            7: "gr-qc",
            8: "hep-ph",
            9: "hep-th",
            10: "math.AP",
            11: "math.CO",
            12: "physics.optics",
            13: "quant-ph",
            14: "stat.ML",
        }

        # Create classifier
        bnb = BernoulliNB(X=train_abstracts, y=train["Category"], alpha=0.3)
        bnb.fit()
        print(">> Training done. Generating submission")

        # Map predictions and output submission file.
        y_test = bnb.generate_matrix_prediction(test_abstracts)
        y_test = np.vectorize(cat_mapping.get)(y_test)
        test["Category"] = y_test
        test.drop(columns=["Abstract"], inplace=True)
        # LEADERBOARD SCORE: 0.76822 w/ 0.3 (FULL DATA: 0.77755 w/ 0.3)
        test.to_csv(OUTPATH, index=False)
        print(f"Bernoulli NB classification done and output to {OUTPATH}")
    else:
        print("INVALID algorithm. Choose from ['random', 'nbbernoulli']")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Code to produce comparison plots.
# import seaborn as sns
# import matplotlib.pyplot as plt

# alpha_errors = {}
# alpha_errors[a] = np.mean(y_validation == validation['Category'])

# # Organise data into dataframes. scores = {"Alpha": list(alpha_errors.keys()) + list(alpha_errors.keys()),
# "Accuracy": list(alpha_errors.values()) + [0.7733333333333333, 0.7813333333333333, 0.7893333333333333,
# 0.7933333333333333, 0.792, 0.7933333333333333, 0.792], "Source": ["Bernoulli", "Bernoulli", "Bernoulli",
# "Bernoulli", "Bernoulli", "Bernoulli", "Bernoulli", "Multinomial", "Multinomial", "Multinomial", "Multinomial",
# "Multinomial", "Multinomial", "Multinomial"]} df = pd.DataFrame(scores)

# # Generate graph plt.figure(figsize=(10, 7)) plt.title('Validation Set Prediction Accuracy of Naive Bayes
# Classification on Arxiv Article Abstracts\n Using 70/30 Train/Validation Split') sns.lineplot(x="Alpha",
# y="Accuracy", hue="Source", data=df) plt.savefig('compare.png') plt.show();
