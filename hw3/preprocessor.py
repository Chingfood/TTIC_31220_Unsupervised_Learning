from __future__ import print_function

import os
import re
import string
import argparse
import tempfile
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser(description="Extract tf-idf feature for corpus")
parser.add_argument("--doc", type=str, help="unprocessed documents (dir)")
parser.add_argument("--index", type=str, help="document ids (file)")
parser.add_argument("--tfidf", type=str, help="output feature (file)")
args = parser.parse_args()

def tokenize_sent(din, dout, idx_fn):
    stemmer = SnowballStemmer("english")
    with open(idx_fn, "r") as fo:
        fns = map(lambda x: x.strip(), fo.readlines())
    print("Sent-tokenize doc")
    for fn in fns:
        doc_fin = os.path.join(din, fn)
        with open(doc_fin, "r") as fo:
            doc = "".join(fo.readlines())
        # remove non-ascii chars
        doc = ''.join([i if ord(i) < 128 else ' ' for i in doc])
        # convert to lower case
        doc = doc.lower()
        # remove numbers
        doc = re.sub(r'\d+', '', doc)
        # sentence tokenize
        doc = sent_tokenize(doc)
        # tokenize and stem each sentence
        for i in range(len(doc)):
            words = word_tokenize(doc[i])
            words = [stemmer.stem(w) for w in words]
            doc[i] = " ".join(words)

        doc = "\n".join(doc)
        fout = os.path.join(dout, fn)
        with open(fout, "w") as fo:
            fo.write(doc)
    return

def make_vocab(din, dict_fn, idx_fn):
    print("Make dictionary: %s" % (dict_fn))
    with open(idx_fn, "r") as fo:
        fns = list(map(lambda x: x.strip(), fo.readlines()))
    doc_fns = map(lambda x: os.path.join(din, x), fns)
    words = []
    for doc_fn in doc_fns:
        with open(doc_fn, "r") as fo:
            lns = list(map(lambda x: x.strip(), fo.readlines()))
            doc = " ".join(lns)
        doc = re.split(' ', doc)
        words = words + doc
    word_counter = Counter(words)
    with open(dict_fn, "w") as fo:
        for w, freq in word_counter.items():
            if freq > 0:
                fo.write(w+"\n")
    return

def make_bow(din, dout, dict_fn, idx_fn):
    print("Making bag-of-words")
    with open(idx_fn, "r") as fo:
        fns = map(lambda x: x.strip(), fo.readlines())
    stop_set = set(stopwords.words('english'))

    bow_fns = map(lambda x: os.path.join(dout, x), fns)
    doc_fns = map(lambda x: os.path.join(din, x), fns)
    vocab = {}
    with open(dict_fn, "r") as fo:
        words = map(lambda x: x.strip(), fo.readlines())
        for i in range(len(words)):
            vocab[words[i]] = i
    for i in range(len(fns)):
        doc_fn = os.path.join(din, fns[i])
        bow_fn = os.path.join(dout, fns[i])
        with open(doc_fn, "r") as fo:
            doc = map(lambda x: x.strip(), fo.readlines())
        # split doc into word list
        doc = " ".join(doc)
        doc = re.split(' ', doc)
        # vocab
        bow = np.zeros(len(vocab), dtype=np.float32)
        for w in doc:
            if w in stop_set:
                continue
            if w in vocab:
                bow[vocab[w]] += 1

        bow = bow/(bow.sum()+1)
        np.save(bow_fn, bow)
    return

def make_tfidf(din, tfidf_fn, dict_fn, idx_fn):
    print("Making Tf-idf")
    with open(idx_fn, "r") as fo:
        fns = [x.strip() for x in fo.readlines()]
    docs = []
    for i in range(len(fns)):
        doc_fn = os.path.join(din, fns[i])
        with open(doc_fn, "r") as fo:
            doc = [x.strip() for x in fo.readlines()]
        doc = " ".join(doc)
        docs.append(doc)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, max_features=int(2.0e5), ngram_range=(1,3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    np.save(tfidf_fn, tfidf_matrix)
    return

def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        token_dir = tmp_dir
        dict_file = os.path.join(tmp_dir, "dict")
        # process document
        tokenize_sent(args.doc, token_dir, args.index)
        # make vocabulary
        make_vocab(token_dir, dict_file, args.index)
        # make tf-idf
        make_tfidf(token_dir, args.tfidf, dict_file, args.index)
    return

if __name__ == "__main__":
    main()
