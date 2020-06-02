from Parse import make_wakati
from DatasetAndVocab import *
from LanguageModel import LM
from trainer import run_trainer


trainpath = './scraping.txt'
vectorpath = './twitter100M-vocab100K-dim300.fasttext.vec'
batch_size = 32
threshhold = 10

data_set, vocab = make_data_set_and_vocab(
    trainpath=trainpath, threshhold=threshhold)

data_loader = dataloader = torch.utils.data.DataLoader(
    data_set, batch_size=batch_size, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_trainer(data_loader=data_loader, vocab=vocab,
            device=device, max_epochs=10000,)
