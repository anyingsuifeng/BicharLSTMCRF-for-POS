class Config(object):
    def __init__(self):
        #数据文件
        self.train_file = '../data/ctb5/train.conll'                     #训练集文件
        self.dev_file = '../data/ctb5/dev.conll'                         #开发集文件
        self.test_file = '../data/ctb5/test.conll'                       #测试集文件
        self.pre_trained_embed_file = '../embedding/embed_100.txt'       #预训练词嵌入文件
#        self.word_file = '../result/word.pkl'                           #保存词的文件，词从预训练词嵌入文件读取
#        self.tag_file = '../result/tag.pkl'                             #保存词性文件
#        self.embedding_file = '../result/embedding.pkl'                 #保存词嵌入矩阵文件

        #模型参数：
        self.embed_dim = 100                                             #表示每个词的向量维度
        self.n_hidden = 300                                              #隐藏层神经元数目
        self.char_embed_dim = 200                                        #字嵌入得到的隐藏神经元数目
        self.lstm_file = '../result/bicharlstmcrf.pkl'                   #保存模型的pkl文件


        #每个词用向量表示的维度
        self.epochs = 100                                                #最大迭代次数
        self.batch_size = 25                                             #每个批次的数目
        self.interval = 10                                               #多少次正确率没有提升就退出
        self.learn_rate = 0.001                                          #学习速率
        self.thread_num = 4                                              #最大线程数
        self.shuffle = True                                              #是否打乱