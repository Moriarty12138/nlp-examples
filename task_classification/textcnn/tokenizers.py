#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import os

import args
from utils import load_dataset


class TextCNNTokenizer(object):
    def __init__(self, vocab_path=None):
        self.word_index = {}
        self.index_word = {}
        self.vocab_path = vocab_path if vocab_path else "vocab.txt"
        if self.vocab_path and os.path.exists(self.vocab_path):
            self.load_vocab(self.vocab_path)
        else:
            texts, _ = load_dataset(args.train_dataset_path)
            self.build_vocab(texts)

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as reader:
            for idx, word in enumerate(reader.readlines()):
                word = word.strip()
                self.word_index[word] = idx
                self.index_word[idx] = word

    def build_vocab(self, texts, max_size=args.vocab_size):
        word_counts = collections.OrderedDict()
        for text in texts:
            words = args.tokenizer.tokenize(text=text)
            for w in words:
                if w in word_counts:
                    word_counts[w] += 1
                else:
                    word_counts[w] = 1
        wcounts = list(word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        wcounts = wcounts[:max_size-2]
        sortes_vocab = ["[PAD]", "[UNK]"]
        sortes_vocab.extend(wc[0] for wc in wcounts)
        self.word_index = dict(
            zip(sortes_vocab, list(range(2, len(sortes_vocab) + 2)))
        )
        self.index_word = {c: w for w, c in self.word_index.items()}
        with open(self.vocab_path, 'w', encoding='utf-8') as writer:
            for word, id in self.word_index.items():
                writer.write(word.strip() + '\n')

    def word2id(self, word):
        return self.word_index.get(word, 1)

    def id2word(self, idx):
        return self.index_word[idx]

    def words2id(self, words):
        return [self.word2id(word) for word in args.tokenizer.tokenize(words)]

    def id2words(self, idxes):
        return [self.id2word(id) for id in idxes]


if __name__ == '__main__':
    tokenizer = TextCNNTokenizer()
    sentence = "the movie The In-Laws not exactly a holiday movie but funny and good!"
    print(tokenizer.words2id(sentence))
    ids = [1, 1062, 395, 1, 50, 1962, 395, 1062, 395, 1, 8, 766,
           1139, 1, 34, 1322, 1559, 1062, 50, 34, 1, 50, 1962, 8,
           50, 1, 1139, 1559, 354, 1]
    print(tokenizer.id2words(ids))
