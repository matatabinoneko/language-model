import random
import torch
import torch.nn.functional as F


class LM(torch.nn.Module):
    def __init__(self, vocab, dim_emb, dim_hid=256):
        super().__init__()

        self.vocab = vocab
        self.embed = torch.nn.Embedding(len(vocab), dim_emb)
        self.rnn = torch.nn.LSTM(dim_emb, dim_hid, batch_first=True)
        self.out = torch.nn.Linear(dim_hid, len(vocab))

    def forward(self, x, state=None):
        x = self.embed(x)
        x, (h, c) = self.rnn(x, state)
        x = self.out(x)
        return x, (h, c)

    # def to_int(self, a):
    #     if a == -float('inf'):
    #         return 0
    #     else:
    #         return int(1e9*a)

    def generate(self, start=None, max_len=100):

        if start is None:
            start = random.choice(self.vocab.index2word)

        idx = self.embed.weight.new_full(
            (1, 1),
            self.vocab.get_index(start),
            dtype=torch.long)
        decoded = [start]
        state = None
        unk = self.vocab.get_index('<unk>')
        while decoded[-1] != '<eos>' and len(decoded) < max_len:
            x, state = self.forward(idx, state)
            x[:, :, unk] = -float('inf')

            # prob = list(map(self.to_int, x.squeeze().tolist()))

            # idx = torch.tensor(random.choices(
            #     list(range(len(prob))), weights=prob, k=1)).view(1, -1)

            idx = torch.argmax(x, dim=-1)

            word = self.vocab.get_word(idx.item())
            decoded.append(word)
        return ' '.join(decoded)
