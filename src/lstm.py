import torch.nn as nn
import torch.utils.data as Data
from torch.nn.utils.rnn import *
from clstm import CharLSTM
from crf import CRF

class LSTMTagger(nn.Module):
    def __init__(self, word2id, char2id, tag2id, pretrain_embedding, embed_dim, char_embed_dim, n_hidden):
        super(LSTMTagger,self).__init__()
        self.word2id = word2id              #通过预训练emdedding得到的word字典
        self.char2id = char2id
        self.tag2id = tag2id
        self.word_num = len(word2id)
        self.char_num = len(char2id)
        self.tag_num = len(tag2id)
        self.embed_dim = embed_dim
        self.embedding = torch.nn.Embedding.from_pretrained(torch.FloatTensor(pretrain_embedding),
                                                            freeze=False)       #加载预训练embedding矩阵并设置为可变
#        self.pre_embedding = nn.Embedding(self.word_num,self.embedding_dim)
        self.clstm = CharLSTM(chrdim=self.char_num,
                              embdim=embed_dim,
                              char_embed=char_embed_dim,
                              )
        self.wlstm = nn.LSTM(input_size = embed_dim + char_embed_dim,
                            hidden_size = n_hidden // 2,
                            num_layers = 1,
                            batch_first = True,
                            bidirectional = True
                            )
        self.out = nn.Linear(n_hidden, self.tag_num)
        self.crf = CRF(self.tag_num)
        self.drop = nn.Dropout()



    def forward(self, x, lens, char_x, char_lens):
        B, T, N= x.shape
        # 获取掩码[B,T]，每行前多少个1就代表多少个词
        mask = torch.arange(T) < lens.unsqueeze(-1)
        # 获取词嵌入向量
        x = self.embedding(x).view(B, T, -1)

        # 获取字嵌入向量
        char_x = self.clstm(char_x[mask], char_lens[mask])
        char_x = pad_sequence(torch.split(char_x, lens.tolist()), True)

        # 拼接词表示和字表示
        x = torch.cat((x, char_x), dim=-1)
        x = self.drop(x)

        x = pack_padded_sequence(x, lens, True)
        x, _ = self.wlstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.drop(x)
        return self.out(x)

    @torch.no_grad()
    def evaluate(self, loader, loss_fnc):
        # 设置为评价模式
        self.eval()

        loss, tp, total = 0, 0, 0
        # 从加载器中加载数据进行评价
        for x, lens, char_x, char_lens, y in loader:
            # 获取掩码
            mask = y.ge(0)
            y = y[mask]
            out = self.forward(x, lens, char_x, char_lens)
            emit = out.transpose(0, 1)  # [T, B, N]
            target = pad_sequence(torch.split(y, lens.tolist()))  # [T, B]
            mask = mask.t()  # [T, B]
            predict = self.crf.viterbi(emit, mask)
            loss += self.crf(emit, target, mask)
            tp += torch.sum(predict == y).item()
            total += lens.sum().item()
        loss /= len(loader)

        return tp, total, loss

    def get_loader(self, dataset, batch_size, thread_num, shuffle):
        data_loader = Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=thread_num,
            collate_fn=self.collate_fn,
        )
        return data_loader


    def save(self,lstm_file):
        with open(lstm_file,'wb') as f:
            torch.save(self,lstm_file)

    def collate_fn(self, data):
        # 按照长度调整顺序
        data.sort(key=lambda x: x[1], reverse=True)
        x, lens, char_x, char_lens, y = zip(*data)
        # 获取句子的最大长度
        max_len = lens[0]
        # 去除无用的填充数据
        x = torch.stack(x)[:, :max_len]
        lens = torch.tensor(lens)
        char_x = torch.stack(char_x)[:, :max_len]
        char_lens = torch.stack(char_lens)[:, :max_len]
        y = torch.stack(y)[:, :max_len]
        return x, lens, char_x, char_lens, y

    @staticmethod
    def load(lstm_file):
        with open(lstm_file, 'rb') as f:
            lstm = torch.load(f)
        return lstm


