import MeCab
import re
import unicodedata

tagger = MeCab.Tagger("-Owakati")


def make_wakati(sentence):
    sentence = tagger.parse(sentence)
    # sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence)
    # sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", sentence)
    sentence = unicodedata.normalize("NFKC", sentence)
    sentence = sentence.rstrip()
    wakati = sentence.split(" ")
    wakati = list(filter(("").__ne__, wakati))
    return wakati
