import math
import sys

from loguru import logger
import tqdm
import numpy as np
import pandas as pd
import argparse

from text_helper import *


class LogOddsRatio:
    """
    Log-odds-ratio with informative Dirichlet priors
    """

    def __init__(self, corpus_i, corpus_j, background_corpus=None, lower_case=True, rm_stopwords=True,
                 rm_punctuations=True, tokenizer=None):
        """
        Create a class object and prepare word counts for log-odds-ratio computation

        Args:
            corpus_i:
                A list of documents, each contains a string
            corpus_j:
                A list of documents, each contains a string
            background_corpus (default = None):
                If None, it will be assigned to a concatenation of `corpus_i` and `corpus_j`
            lower_case:
                Whether lower case all words
            rm_stopwords:
                Whether remove stopwords in preprocessing step
            rm_punctuations:
                Whether remove punctuations in preprocessing step
            tokenizer:
                To specify a specific tokenizer for tokenization step
        """

        def preprocessing(corpus):
            # if lower_case:
            #     corpus = [text.lower() for text in corpus]
            # corpus = decontract(corpus)
            # tokenized_corpus = parallel_tokenize(corpus, tokenizer)
            tokenized_corpus = [text.lower().split(' ') for text in corpus]
            if rm_stopwords:
                tokenized_corpus = remove_stopwords(tokenized_corpus)
            if rm_punctuations:
                tokenized_corpus = remove_punctuations(tokenized_corpus)
            return tokenized_corpus

        # Convert a list of string into a list of lists of words
        logger.info("Preprocessing corpus-i")
        corpus_i = preprocessing(corpus_i)
        logger.info("Preprocessing corpus-j")
        corpus_j = preprocessing(corpus_j)
        if background_corpus != None:
            logger.info("Preprocessing corpus-background")
            background_corpus = preprocessing(background_corpus)

        # Compute word counts of every words on each corpus separately
        logger.info("Getting word counts from corpus-i")
        self.y_i = get_word_counts(corpus_i)
        logger.info("Getting word counts from corpus-j")
        self.y_j = get_word_counts(corpus_j)
        logger.info("Getting word counts from corpus-background")
        if background_corpus:
            self.alpha = get_word_counts(background_corpus)
        else:
            # Combine words and sum their counts of corpus i and j in case no specified background corpus
            self.alpha = {k: self.y_i.get(k, 0) + self.y_j.get(k, 0) for k in set(self.y_i) | set(self.y_j)}

        # Sort dicts
        logger.debug("Start sorting and backing up to files")
        self.y_i = {k: v for k, v in sorted(self.y_i.items(), key=lambda item: item[1], reverse=True)}
        self.y_j = {k: v for k, v in sorted(self.y_j.items(), key=lambda item: item[1], reverse=True)}
        self.alpha = {k: v for k, v in sorted(self.alpha.items(), key=lambda item: item[1], reverse=True)}

        # Initialize necessary variables
        self.delta = None
        self.sigma_2 = None
        self.z_scores = None

        # Compute
        logger.info("Start computing delta")
        self._compute_delta()
        logger.info("Start computing sigma^2")
        self._compute_sigma_2()
        logger.info("Start computing Z-score")
        self._compute_z_scores()

        # Sort dicts
        logger.debug("Start sorting and backing up to files")
        self.delta = {k: v for k, v in sorted(self.delta.items(), key=lambda item: item[1], reverse=True)}
        self.sigma_2 = {k: v for k, v in sorted(self.sigma_2.items(), key=lambda item: item[1], reverse=True)}
        self.z_scores = {k: v for k, v in sorted(self.z_scores.items(), key=lambda item: item[1], reverse=True)}

    def _compute_delta(self):
        """ The usage difference for word w among two corpora i and j
        """
        self.delta = dict()
        n_i = sum(self.y_i.values())
        n_j = sum(self.y_j.values())
        alpha_zero = sum(self.alpha.values())
        logger.debug(f"Size of corpus-i: {n_i}")
        logger.debug(f"Size of corpus-j: {n_j}")
        logger.debug(f"Size of background corpus: {alpha_zero}")

        try:
            for w in set(self.y_i) | set(self.y_j):  # iterate through all words among two corpora
                first_log = math.log10((self.y_i.get(w, 0) + self.alpha.get(w, 0)) / (
                            n_i + alpha_zero - self.y_i.get(w, 0) - self.alpha.get(w, 0)))
                second_log = math.log10((self.y_j.get(w, 0) + self.alpha.get(w, 0)) / (
                            n_j + alpha_zero - self.y_j.get(w, 0) - self.alpha.get(w, 0)))
                self.delta[w] = first_log - second_log
        except ValueError as e:
            logger.debug(f"Y-i of the word {w}:", self.y_i.get(w, 0))
            logger.debug(f"alpha of the word {w}:", self.alpha.get(w, 0))
            logger.debug(f"value:", (self.y_i.get(w, 0) + self.alpha.get(w, 0)) /
                         (n_i + alpha_zero - self.y_i.get(w, 0) - self.alpha.get(w, 0)))
            raise e

    def _compute_sigma_2(self):
        """ Compute estimated values of sigma squared
        """
        self.sigma_2 = dict()
        for w in self.delta:
            self.sigma_2[w] = (1 / (self.y_i.get(w, 0) + self.alpha.get(w, 0))) + (
                        1 / (self.y_j.get(w, 0) + self.alpha.get(w, 0)))

    def _compute_z_scores(self):
        self.z_scores = dict()
        for w in self.delta:
            self.z_scores[w] = self.delta.get(w, 0) / math.sqrt(self.sigma_2.get(w, 0))


def main(filepath_corpus_i, filepath_corpus_j, save_root):
    # Argument setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath_corpus_i", default=filepath_corpus_i, type=str, required=False)
    parser.add_argument("--filepath_corpus_j", default=filepath_corpus_j, type=str, required=False)
    parser.add_argument("--filepath_background_corpus", default=None, type=str, required=False)
    parser.add_argument("--save_top_words", default=None, type=int, required=False)
    parser.add_argument("--log_level", default=None, type=str, required=False)
    args = parser.parse_args()

    # Set new log level, default is "DEBUG"
    if args.log_level != None:
        logger.remove()
        logger.add(sys.stderr, level=args.log_level)

    # Read file into list of texts
    df_corpus_i = pd.read_csv(args.filepath_corpus_i)
    if len(df_corpus_i) == 0:
        f = open(save_root + "z_scores.txt", "w", encoding='utf-8')
        f.close()
        return
    corpus_i = [text.strip() for text in df_corpus_i["tokenized tweet"]]
    del df_corpus_i
    df_corpus_j = pd.read_csv(args.filepath_corpus_j)
    corpus_j = [text.strip() for text in df_corpus_j["tokenized tweet"]]
    del df_corpus_j
    if args.filepath_background_corpus != None:
        df_corpus_bg = pd.read_csv(args.filepath_background_corpus)
        corpus_bg = [text.strip() for text in df_corpus_bg["tokenized tweet"]]
        del df_corpus_bg
    else:
        corpus_bg = None

    # # Specify the tokenizer
    # tokenizer = pkuseg.pkuseg()
    # Start log-odds-ratio preprocessing
    log_odds_ratio = LogOddsRatio(
        corpus_i, corpus_j, corpus_bg)

    # Write to files as backup
    # with open(save_root+"vocabs_i.txt", "w") as f:
    #     for k, v in log_odds_ratio.y_i.items():
    #         f.write(f"{k},{v}\n")
    # with open(save_root+"vocabs_j.txt", "w") as f:
    #     for k, v in log_odds_ratio.y_j.items():
    #         f.write(f"{k},{v}\n")
    # with open(save_root+"vocabs_alpha.txt", "w") as f:
    #     for k, v in log_odds_ratio.alpha.items():
    #         f.write(f"{k},{v}\n")
    # with open(save_root+"delta.txt", "w") as f:
    #     for k, v in log_odds_ratio.delta.items():
    #         f.write(f"{k},{v}\n")
    # with open(save_root+"sigma_2.txt", "w") as f:
    #     for k, v in log_odds_ratio.sigma_2.items():
    #         f.write(f"{k},{v}\n")
    with open(save_root+"z_scores.txt", "w", encoding='utf-8') as f:
        i = 0
        for k, v in log_odds_ratio.z_scores.items():
            f.write(f"{k},{v}\n")
            i += 1
            if i >= 100:
                break

    # Save top words into a file
    if args.save_top_words != None and args.save_top_words > 0:
        if args.save_top_words > len(log_odds_ratio.z_scores):
            raise ValueError("--save_top_words must be less than or equal to vocab size")

        logger.info(f"Saving top {args.save_top_words} words ranked by Z-score")
        with open(save_root+"top_words.txt", "w") as f:
            i = 0
            for k, v in log_odds_ratio.z_scores.items():
                f.write(k + "\n")
                i += 1
                if i >= args.save_top_words:
                    break


if __name__ == "__main__":
    for i in range(1, 13):
        filepath_corpus_i = ''
        filepath_corpus_j = ''
        root = ''
        main(filepath_corpus_i, filepath_corpus_j, root)
        logger.success("Done!")
