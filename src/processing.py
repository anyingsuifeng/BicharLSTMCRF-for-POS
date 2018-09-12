import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset

class Processing(object):
    SOS = '<SOS>'
    EOS = '<EOS>'
    UNK = '<UNK>'

    def __init__(self,data_file,embed_file):
        self.sentences = self.data_handle(data_file,True)
        self.words,self.tags,self.chars = self.parse(self.sentences)
        self.embedding_matrix = self.read_embed_file(embed_file)
        # print(len(self.words),len(self.tags),len(self.chars),self.embedding_matrix.size())

    def read_embed_file(self,embed_file):
        #读取预处理词向量文件
        with open(embed_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_list = line.split()
                self.embed_dim = len(line_list) - 1
                word = line_list[0]
                self.words.add(word)
                self.chars |= set(word)
        # 得到所有的词和字符，包括训练集和预训练词向量中的词
        self.words.add(self.SOS)
        self.words.add(self.EOS)
        self.words.add(self.UNK)
        self.chars.add(self.UNK)
        self.words = sorted(self.words)
        self.chars = sorted(self.chars)
        self.word2id = {word: id for id, word in enumerate(self.words)}
        self.char2id = {char: id for id, char in enumerate(self.chars)}
        self.tag2id = {tag: id for id, tag in enumerate(self.tags)}
        self.uwi = self.word2id[self.UNK]           #未知词id
        self.uci = self.char2id[self.UNK]           #未知字符id
        embedding_matrix = torch.randn(len(self.words),self.embed_dim)/self.embed_dim**0.5
        with open(embed_file, 'r', encoding='utf-8') as f:
            for line in f:
                word_embedding = []
                line_list = line.split()
                word = line_list[0]
                for i in range(self.embed_dim):
                    word_embedding.append(float(line_list[i + 1]))
                embedding_matrix[self.word2id[word]] = torch.tensor(word_embedding)
        return embedding_matrix

    def load(self, fdata, max_len=10):
        x, lens, char_x, char_lens, y = [], [], [], [], []
        # 句子按照长度从大到小有序
        sentences = sorted(self.data_handle(fdata),
                           key=lambda x: len(x),
                           reverse=True)
        for sentence in sentences:
            length = len(sentence)
            wiseq = [self.word2id.get(sentence[i][0],self.uwi) for i in range(length)]
            tiseq = [self.tag2id[sentence[i][1]] for i in range(length)]
            x.append(torch.tensor([[wi] for wi in wiseq]))
            lens.append(length)
            char_x.append(torch.tensor([
                [self.char2id.get(char, self.uci)
                 for char in word[:max_len]] + [0] * (max_len - len(word))
                for [word,_] in sentence ]))
            char_lens.append(torch.tensor([min(len(word), max_len)
                                           for [word,_] in sentence]))
            y.append(torch.tensor([ti for ti in tiseq]))
        x = pad_sequence(x, True)
        lens = torch.tensor(lens)
        char_x = pad_sequence(char_x, True)
        char_lens = pad_sequence(char_lens, True)
        y = pad_sequence(y, True, padding_value=-1)
        dataset = TensorDataset(x, lens, char_x, char_lens, y)
        return dataset


    @staticmethod
    def data_handle(data_file,Print=False):
        #数据处理函数，将文本转化为句子
        sentences = []
        sentence =[]
        sentence_num = 0
        word_num = 0
        with open(data_file,"r",encoding='utf-8') as data:
            for line in data:
                if len(line) == 1:
                    sentences.append(sentence)
                    sentence = []
                    sentence_num += 1
                else:
                    word = line.split()[1]
                    tag = line.split()[3]
                    #处理ctb5test集中出现的NP问题
                    if data_file == "../data/ctb5/test.conll" and tag == "NP":
                        tag = "NN"
                    sentence.append([word,tag])
                    word_num += 1
        if Print:
            print("文件%s:共%d个句子，%d个词" % (data_file, sentence_num, word_num ))
        return sentences

    @staticmethod
    def parse(sentences):
        #得到训练集中的words，tags和chars
        words = set()
        tags = set()
        chars = set()
        for sentence in sentences:
            for word,tag in sentence:
                words.add(word)
                tags.add(tag)
                chars |= set(word)
        return words,tags,chars
