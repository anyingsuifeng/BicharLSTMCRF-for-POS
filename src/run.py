from config import Config
from processing import Processing
from lstm import LSTMTagger
from torch.nn.utils.rnn import *
import torch
from datetime import datetime

if __name__ == '__main__':

    all_time_start = datetime.now()
    config = Config()
    corpus = Processing(config.train_file,config.pre_trained_embed_file)
    # train_sentences = Processing.data_handle(config.train_file)
    dev_sentences = Processing.data_handle(config.dev_file,True)
    test_sentences = Processing.data_handle(config.test_file,True)
    print("\n训练预料：")
    print("句子数：%d" % len(corpus.sentences))
    print("词数：%d" % len(corpus.words))
    print("字符数：%d" % len(corpus.chars))
    print("词性数：%d" % len(corpus.tags))


    lstm = LSTMTagger(corpus.word2id, corpus.char2id, corpus.tag2id,
                      corpus.embedding_matrix,config.embed_dim,config.char_embed_dim, config.n_hidden)
    train_data_loader = lstm.get_loader(dataset=corpus.load(config.train_file),
                                        batch_size=config.batch_size,
                                        thread_num=config.thread_num,
                                        shuffle=config.shuffle)
    dev_data_loader = lstm.get_loader(dataset=corpus.load(config.dev_file),
                                      batch_size=config.batch_size,
                                      thread_num=config.thread_num,
                                      shuffle=config.shuffle)
    test_data_loader = lstm.get_loader(dataset=corpus.load(config.test_file),
                                       batch_size=config.batch_size,
                                       thread_num=config.thread_num,
                                       shuffle=config.shuffle)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=config.learn_rate)
    loss_func = torch.nn.CrossEntropyLoss()

    torch.set_num_threads(config.thread_num)

    print("\n")
    print(lstm)
    print("\n开始训练\nbatch_size = %d\nlearn_rate = %f"% (config.batch_size, config.learn_rate))
    max_dev_precision, max_dev_precision_index, max_dev_precision_loss= 0.0, 0, 0.0
    for i in range(config.epochs):
        print("================================Epoch%d====================================" % i)
        start = datetime.now()
        lstm.train()
        # 从加载器中加载数据进行训练
        for batch in train_data_loader:
            lstm.train()
            # 清除梯度
            optimizer.zero_grad()

            x, lens, char_x, char_lens, y = batch
            # 获取掩码
            mask = y.ge(0)
            y = y[mask]
            out = lstm(x, lens, char_x, char_lens)

            emit = out.transpose(0, 1)  # [T, B, N]
            target = pad_sequence(torch.split(y, lens.tolist()))  # [T, B]
            mask = mask.t()  # [T, B]
            loss = lstm.crf(emit, target, mask)

            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()

        train_correct, train_total ,train_loss = lstm.evaluate(train_data_loader)
        print('train : accuracy = %d / %d = %.4f ,train_loss = %.8f' % (train_correct, train_total, train_correct / train_total, train_loss))
        dev_correct, dev_total , dev_loss = lstm.evaluate(dev_data_loader)
        dev_precision = dev_correct/dev_total
        print('dev : accuracy = %d / %d = %.4f ,dev_loss = %.8f' % (dev_correct, dev_total, dev_correct / dev_total, dev_loss))
        test_correct, test_total,test_loss = lstm.evaluate(test_data_loader)
        print('test : accuracy = %d / %d = %.4f ,test_loss = %.8f'  % (test_correct, test_total, test_correct / test_total, test_loss))
        if dev_precision > max_dev_precision:
            max_dev_precision = dev_precision
            max_dev_precision_index = i
            max_dev_precision_loss = dev_loss
            lstm.save(config.lstm_file)
        stop = datetime.now()
        time = stop - start
        print("此轮耗时：%s" % time)
        if i - max_dev_precision_index >= 10:
            break

    print("\ndev准确率最高轮次%d，准确率为%.4f，loss为%.8f" % (max_dev_precision_index, max_dev_precision, max_dev_precision_loss))

    # 恢复模型测试test集
    print("恢复模型并测试此时test集:")
    lstm = LSTMTagger.load(config.lstm_file)
    test_correct, test_total, test_loss = lstm.evaluate(test_data_loader)
    print('accuracy = %d / %d = %.4f  loss = %.8f' % (test_correct, test_total, test_correct / test_total, test_loss))

    all_time_stop = datetime.now()
    all_time = all_time_stop - all_time_start
    print("\n总耗时：%s" % all_time)



