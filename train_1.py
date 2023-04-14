from torch import optim
from torch.utils.tensorboard import SummaryWriter
from model_1 import *
from tqdm import tqdm


if __name__ == '__main__':
    batch_size = 512
    epoch = 300
    learning_rate = 1e-3
    sequence_length = 64 # 截取的序列长度
    cov_size = 256 # 卷积核数
    embedding_size = 50 # 词向量化维度
    region_size = 3 # 卷积核高度(如果muti_conv==true, 则卷积核高度为(region_size,region_size+1,region_size+2))
    min_word_frequency = 1 # 最小词频
    participle = False # 是否分词
    muti_conv = False # 是否采用多卷积
    keywords = True # 是否使用keywords
    record = False # 是否保存
    '''
    b=4_lr=1e-3_sl=20_cs=10_es=100_rs=5_m=1
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 读取并预处理数据
    train_inputs, train_labels, vocab_size = read_data(sequence_length=sequence_length, drop_num=min_word_frequency, participle=participle, key_words=keywords)
    test_inputs, test_labels, _ = read_data(train=False,sequence_length=sequence_length, drop_num=min_word_frequency, participle=participle, key_words=keywords)
    train_len = len(train_labels)
    test_len = len(test_labels)
    labels = train_labels + test_labels
    labels, label2idx = label_digitize(labels)
    num_classes = len(label2idx)
    train_labels = labels[:len(train_labels)]
    test_labels = labels[len(train_labels):]
    train_inputs = torch.LongTensor(train_inputs).to(device)
    train_labels = torch.LongTensor(train_labels).to(device)
    test_inputs = torch.LongTensor(test_inputs).to(device)
    test_labels = torch.LongTensor(test_labels).to(device)
    train_dataset = Data.TensorDataset(train_inputs, train_labels)
    test_dataset = Data.TensorDataset(test_inputs, test_labels)
    train_data_loader = Data.DataLoader(train_dataset, batch_size, True)
    test_data_loader = Data.DataLoader(test_dataset, batch_size, True)

    # 训练
    textcnn = TextCNN(num_classes, vocab_size, cov_size=cov_size, embedding_size=embedding_size,
                      region_size=region_size, sequence_length=sequence_length, multi_conv=muti_conv).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(textcnn.parameters(), lr=learning_rate)
    if record:
        writer = SummaryWriter()
    all_step = 0
    for i in tqdm(range(epoch)):

        # 测试
        right_num = 0
        test_total_loss = 0
        for j, (x_test, y_test) in enumerate(test_data_loader):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_test_pre = textcnn(x_test)
            test_total_loss += criterion(y_test_pre, y_test)
            right_num += sum(torch.argmax(y_test_pre, dim=1) == y_test)
        acc_test = float(right_num / test_len)
        print()
        print('第{}个epoch训练开始，测试集准确率{}，总损失{}'.format(i + 1, acc_test, test_total_loss))
        if record:
            writer.add_scalar('test_accuracy', acc_test, i+1)
            writer.add_scalar('test_total_loss', test_total_loss, i + 1)

        # 训练
        for j, (x_train, y_train) in enumerate(train_data_loader):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_train_pre = textcnn(x_train)
            loss = criterion(y_train_pre, y_train)
            if record:
                if all_step % 1000 == 0:
                    writer.add_scalar('train_loss', loss, all_step)
            all_step += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if record:
            torch.save(textcnn, 'model_1\\epoch_{}.pth'.format(i + 1))
