from collections import Counter
import jieba
import json
import torch


# 读取数据并分词
def load_data(train=True, category='label_desc', participle=True, key_words=True):
    dataset = []
    length = 0
    if train:
        file_path = 'tnews_public/tnews_public/train.json'
    else:
        file_path = 'tnews_public/tnews_public/test.json'
    for line in open(file_path, "r", encoding='utf-8'):
        instance = json.loads(line)
        sent = []
        if key_words and len(instance['keywords']):
            sent += instance['keywords'].split(',')
        if participle:
            sent += [x for x in jieba.cut(instance['sentence'])] # 分词
        else:
            sent += [x for x in instance['sentence']] # 不分词
        length = max(length, len(sent))
        label = instance[category]
        dataset.append([sent, label])
    return dataset


# 统计词汇序号
def make_data(dataset, drop_num=1):
    counter = Counter()
    for (sent, label) in dataset:
        counter.update(sent)
    vocab = [(word, count) for (word, count) in counter.items() if count > drop_num]
    sorted_vocab = sorted(vocab, key=lambda x: x[1], reverse=False)
    word2idx = {'unk': 0}
    word2idx.update({word: i + 1 for i, (word, count) in enumerate(sorted_vocab)})
    # idx2word = {i: word for (word, i) in word2idx.items()}
    return word2idx


# 输入数据编码（标签的编码为其他函数，因为标签的编码不必须，且需要可逆）
def serialization(dataset, word2idx):
    inputs = []
    labels = []
    for sent, label in dataset:
        idx_seq = []
        for word in sent:
            if word in word2idx:
                idx_seq.append(word2idx[word])
            else:
                idx_seq.append(word2idx["unk"])
        inputs.append(idx_seq)
        labels.append(label)
    return inputs, labels


# 自变量等长编辑
def padding_and_cut(inputs, max_length):
    # padding 补零
    for i in range(len(inputs)):
        if len(inputs[i]) < max_length:
            inputs[i] += [0]*(max_length-len(inputs[i]))
        else:
            # 截取
            inputs[i] =inputs[i][:max_length]
    return inputs


# 供外部调用
def read_data(train=True, sequence_length=20, category='label_desc', drop_num=1, participle=True, key_words=True):
    data_train = load_data(category=category, participle=participle, key_words=key_words)
    data_test = load_data(train=False, category=category, participle=participle, key_words=key_words)
    data = data_train + data_test
    data_w2i = make_data(data, drop_num=drop_num)
    num_vocab = len(data_w2i)
    if train:
        inputs, labels = serialization(data_train, data_w2i)
    else:
        inputs, labels = serialization(data_test, data_w2i)
    inputs = padding_and_cut(inputs, max_length=sequence_length)
    return inputs, labels, num_vocab


# 标签编码
def label_digitize(labels):
    counter = Counter()
    counter.update(labels)
    label2idx = {label:i for i, (label, _) in enumerate(counter.items())}
    for i, label in enumerate(labels):
        labels[i] = label2idx[label]
    return labels, label2idx


if __name__ == '__main__':
    dtype = torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    inputs, labels, _ = read_data(train=False, sequence_length=50)
    labels = label_digitize(labels)
    # print(inputs)
    # print(labels)