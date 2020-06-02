import collections
import math
import os
import random

import torch
import torch.nn.functional as F
import logging


class Vocabulary:
    def __init__(self):
        self.index2word = {}
        self.word2index = {}
        self.add_word('<unk>')
        self.add_word('<pad>')
        self.add_word('<eos>')
        self.add_word('<bos>')

    def __len__(self):
        return len(self.word2index)

    #　 if 'hoge' in Vocabulary　とかすると，containsが返される
    def __contains__(self, word):
        return word in self.word2index.keys()

    def add_word(self, word):
        if word not in self.word2index.keys():
            self.index2word[len(self.index2word)] = word
            self.word2index[word] = len(self.word2index)

    def get_word(self, index):
        return self.index2word[index]

    def get_index(self, word):
        return self.word2index[word]

    def sentence2index(self, wakati, length):
        ret = [self.get_index(word) if word in self.word2index.keys(
        ) else self.get_index('<unk>')for word in wakati]
        pad = [self.get_index('<pad>') for i in range(length-len(wakati))]
        eos = [self.get_index('<eos>')]
        bos = [self.get_index('<bos>')]
        return bos+ret+eos+pad

    def save(self, vocab_file):
        with open(vocab_file, 'w') as f:
            for word in self.word2index:
                print(word, file=f)

    def load(self, vocab_file):
        with open(vocab_file, 'r') as f:
            for line in f:
                line = line.rstrip().split()
                word = line[0]
                self.add_word(word)
        # print("load from {} word size:{} dim_emb:{}".format(vocabfile,len(self.word2index),self.dim_emb))
        return
