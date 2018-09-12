# BicharLSTMCRF-for-POS
### 一、目录文件

```
./data:                   
	ctb5:					# ctb5数据文件
	ctb7:                     # ctb7数据文件
./embedding:
	base_embeddings_50.txt    # 50维预训练词向量
	embed_100.txt			 # 100维预训练词向量
./result:                 
	bicharlstmcrf.pkl         # 保存模型文件
	BicharLSTMCRF.txt         # 运行结果文件
./src：         
	config.py                 # 配置代码 
	processing.py             # 预处理代码
	clstm.py                  # char_embedding模块代码
	crf.py                    # crf模块代码
	lstm.py                   # lstm代码
	run.py                    # 运行代码
./README.md                    # 使用说明
```



### 二、运行

##### 1.运行环境

```
python 3.6
pytorch 0.4
```

##### 2.运行方法

```bash
$ cd ./src
$ vim config.py                #修改配置文件
$ python run.py                #运行代码
```



### 三、参考结果

```
文件../data/ctb5/train.conll:共16091个句子，437991个词
文件../data/ctb5/dev.conll:共803个句子，20454个词
文件../data/ctb5/test.conll:共1910个句子，50319个词

训练预料：
句子数：16091
词数：54303
字符数：7477
词性数：32

LSTMTagger(
  (embedding): Embedding(54303, 100)
  (clstm): CharLSTM(
    (embed): Embedding(7477, 100)
    (lstm): LSTM(100, 100, batch_first=True, bidirectional=True)
  )
  (wlstm): LSTM(300, 150, batch_first=True, bidirectional=True)
  (out): Linear(in_features=300, out_features=32, bias=True)
  (crf): CRF()
  (drop): Dropout(p=0.5)
)
```

| batch-size | learn_rate | iter | dev准确率 | test准确率 | 时间/迭代 |
| :--: | :------: | :---------: | :----: | :--------: | :-------: |
|     25     | 0.001 |     13/23     | 95.67% |   95.62%   | ~7min |

