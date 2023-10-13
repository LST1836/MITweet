import string
import tqdm
from nltk.corpus import stopwords


def remove_stopwords(corpus, language='english'):
    stop_words = set(stopwords.words(language))
    processed_corpus = []
    for words in corpus:
        words = [w for w in words if not w in stop_words]
        processed_corpus.append(words)
    return processed_corpus


def remove_punctuations(corpus):
    punctuations = string.punctuation + '！“”‘’（），。、：；″°′《》？←→↑↓【】……——'
    processed_corpus = []
    for words in corpus:
        words = [w for w in words if not w in punctuations]
        processed_corpus.append(words)
    return processed_corpus


def get_word_counts(corpus):
    # Initializing Dictionary
    d = {}

    # Counting number of times each word comes up in list of words (in dictionary)
    for words in tqdm.tqdm(corpus, desc="Word Counting"):
        for w in words:
            d[w] = d.get(w, 0) + 1
    return d