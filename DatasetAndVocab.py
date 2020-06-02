import collections
from Parse import make_wakati
import torch
import torch.nn as nn
from Vocabulary import Vocabulary


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, trainpath, vocab):

        self.data = []
        self.vocab = vocab
        max_length = -1
        with open(trainpath, 'r') as f:
            # 文の最大の長さを取得
            for line in f:
                line = line.strip()
                max_length = max(max_length, len(line))
            f.seek(0)

            for line in f:
                line = line.strip()
                if 0 < len(line):
                    words = self.vocab.sentence2index(
                        make_wakati(line), length=max_length)
                    self.data.append(torch.LongTensor(words))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def make_data_set_and_vocab(trainpath=None, vectorpath=None, threshhold=0):
    vocab = Vocabulary()
    if vectorpath is not None:
        vocab.load(vectorpath)

    counter = collections.Counter()
    with open(trainpath, 'r') as f:
        for line in f:
            words = make_wakati(line.strip())
            for word in words:
                counter[word] += 1

    # for word, _ in counter.most_common(self.n_max_word - 2):
    for word, cnt in counter.most_common():
        if cnt <= threshhold:
            break
        if word not in vocab:
            vocab.add_word(word)
    vocab.save('vocab')

    # ここからデータセット作成
    data_set = MyDataset(trainpath=trainpath, vocab=vocab)

    return data_set, vocab
