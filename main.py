import csv
import os
import datetime as dt
import pickle
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pytorchtools import EarlyStopping
from ge2e import GE2ELoss
from torch.optim import SGD
import torch
import torch.utils.data as Data
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix

from dataprocess.read import to_transpose, plot_pic

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [2, 3]))
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 调制信号类别
classes = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
           b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}
classes1 = {'QAM16', 'QAM64', '8PSK', 'WBFM', 'BPSK', 'CPFSK', 'AM-DSB', 'GFSK', 'PAM4', 'QPSK', 'AM-SSB'}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

snrs = ""
mods = ""


# 画混淆矩阵
def plot_Matrix(testloader):
    plt.figure(4)
    correct = 0
    total = 0
    truesz = []
    predsz = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            numpy1 = labels.cpu().numpy()
            truesz = np.hstack((truesz, numpy1))
            numpy = predicted.cpu().numpy()
            predsz = np.hstack((predsz, numpy))
    print('测试集精度: %.3f %%' % (100 * correct / total))
    y_true = truesz
    y_pred = predsz
    C = confusion_matrix(y_true, y_pred)
    confusion = np.array(C)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(rotation=45)
    classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
               'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9, 'AM-SSB': 10}
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.title('Confusion matrix(all)')
    # 显示数据
    for first_index in range(len(confusion)):  # 第几行
        for second_index in range(len(confusion[first_index])):  # 第几列
            # plt.text(first_index, second_index, round(confusion[second_index][first_index]/np.sum(confusion[second_index]), 2),
            plt.text(first_index, second_index, confusion[second_index][first_index],
                     fontsize=8, verticalalignment='center', horizontalalignment='center')
    # 在matlab里面可以对矩阵直接imagesc(confusion)
    # 显示
    plt.show()


# 数据读取
def data_process(fp):
    global snrs, mods, lbl
    Xd = pickle.load(open(fp, 'rb'), encoding='bytes')  # 读取数据
    '''
    从数据集中遍历返回snrs,mods两个参数. 1，0是j的参数
    map会根据提供的函数对指定序列做映射。lambda 表达式  冒号左边是函数的参数，右边是函数的返回值（逻辑表达式）
    map(lambda x: x[j], Xd.keys())#Xd.key()取出key,x[key]再返回

    Xd.keys()取出Xd中的键keys,形为('8PSK',-10),
    故snrs值为:sorted(list(set(map(lambda x: x[1], Xd.keys())))),
    mods值为:sorted(list(set(map(lambda x: x[0], Xd.keys()))))
    '''
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    y = []
    lbl = []

    for mod in mods:
        for snr in snrs:
            '''
            X.append(Xd[(mod, snr)])  # X将Xd开头的(mod,snrs)去掉后的矩阵
            for i in range(Xd[(mod, snr)].shape[0]):
                y.append(mod)  # lpl:(b'8PSK',-20)... len:220000
                lbl.append(snr)
            '''
            if snr >= 0:
                X.append(Xd[(mod, snr)])  # X将Xd开头的(mod,snrs)去掉后的矩阵
                for i in range(Xd[(mod, snr)].shape[0]):
                    y.append(mod)  # lpl:(b'8PSK',-20)... len:220000
                    lbl.append(snr)

    X = np.vstack(X)  # 垂直（按照行顺序）的把数组给堆叠起来。（220000,2,128）

    Y = [classes[yy] for yy in y]
    Y = np.array(Y, dtype=np.int64)  # list转换成int64
    Y = torch.from_numpy(Y)  # 数据的类别标签要转化为int64
    return (X, Y)


# 数据分割
def train_test_split(X, Y, test_size, random_state):
    global test_idx
    np.random.seed(random_state)
    n_examples = X.shape[0]

    n_train = int(n_examples * (1 - test_size))
    train_idx = np.random.choice(
        range(0, n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    return (X_train, X_test, Y_train, Y_test)


# 定义网络的训练过程函数
def train_model(model, traindataloader, testdataloader, train_rate, criterion, optimizer, num_epochs=25):
    # train_rate:训练集batchsize百分比;criterion：损失函数；optimizer：优化方法；
    # 计算训练使用的batch数量
    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * train_rate)
    # 复制模型的参数
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    lr_list = []
    val_corrects = 0
    val_num = 0

    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    since = time.time()

    with torch.no_grad():
        model.eval()
        for data in testdataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            val_num += labels.size(0)
            val_corrects += torch.sum(predicted == labels)

    best_acc = val_corrects.double().item() / val_num
    print('起始精度:%.3f %%' % (best_acc * 100))

    for epoch in range(num_epochs):
        print('-' * 50)
        every_since = time.time()
        # 每个epoch有两个训练阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        correct = 0
        total = 0

        for i, data in enumerate(traindataloader, 0):
            model.train()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            model.train()
            optimizer.zero_grad()
            output = model(inputs)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(pre_lab == labels.data)
            train_num += labels.size(0)

        with torch.no_grad():
            model.eval()
            for data in testdataloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                val_num += labels.size(0)
                val_corrects += torch.sum(predicted == labels)

        lr_list.append(optimizer.param_groups[0]['lr'])
        # lr_scheduler.step()
        # 计算一个epoch在训练集和验证集上的的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)

        val_acc_all.append(val_corrects.double().item() / val_num)

        # 将损失和精度保存到csv表格中
        list1 = [epoch + 1, train_loss_all[-1], train_acc_all[-1] * 100, val_acc_all[-1] * 100]
        with open("/home/data/jjg1/rnn/2016ares.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list1)
            csvfile.close()

            '''
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)
            if step < train_batch_num:
                model.train()  # 设置模型为训练模式
                optimizer.zero_grad()
                output = model(inputs)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                train_corrects += torch.sum(pre_lab == labels.data)
                train_num += labels.size(0)
                # 保存模型
                # torch.save(model.state_dict(), './model/'+str(epoch)+"_"+str(step)+'.pkl')
            else:
                model.eval()  # 设置模型为训练模式评估模式
                output = model(inputs)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(pre_lab == labels.data)
                val_num += inputs.size(0)

        lr_list.append(optimizer.param_groups[0]['lr'])
        # lr_scheduler.step()
        # 计算一个epoch在训练集和验证集上的的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('[{}] 训练损失: {:.4f}  训练精度: {:.2f}%  验证损失: {:.4f}  验证精度: {:.2f}%'
              .format(epoch + 1, train_loss_all[-1], train_acc_all[-1] * 100, val_loss_all[-1], val_acc_all[-1] * 100))
        # 将损失和精度保存到csv表格中
        list1 = [epoch + 1, train_loss_all[-1], train_acc_all[-1] * 100, val_loss_all[-1], val_acc_all[-1] * 100]
        with open("/home/data/jjg1/rnn/2016alenet.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list1)
            csvfile.close()
        '''
        running_loss = 0.0
        # 拷贝模型最高精度下的参数
        now_time = dt.datetime.now().strftime('%F %T')
        every_time_use = time.time() - every_since
        total_time_use = time.time() - since

        if val_acc_all[-1] >= best_acc:
            best_acc = val_acc_all[-1]
            print('[{}] 训练损失: {:.4f}  训练精度: {:.2f}%  测试精度: {:.2f}%  保存模型'
                  .format(epoch + 1, train_loss_all[-1], train_acc_all[-1] * 100, val_acc_all[-1] * 100))
            torch.save(model, '/home/data/jjg1/rnn/2016apkl/cs.pkl', _use_new_zipfile_serialization=False)
        else:
            print('[{}] 训练损失: {:.4f}  训练精度: {:.2f}%  测试精度: {:.2f}%'
                  .format(epoch + 1, train_loss_all[-1], train_acc_all[-1] * 100, val_acc_all[-1] * 100))

        print("本轮用时 {:.0f}m {:.5f}s  总共用时 {:.0f}m {:.5f}s 当前时间"
              .format(every_time_use // 60, every_time_use % 60, total_time_use // 60,
                      total_time_use % 60) + '   ' + now_time)

    # early_stopping(correct / total, model)

    # 使用最好模型的参数
    model = torch.load('/home/data/jjg1/rnn/2016apkl/cs.pkl')  # 加载模型

    train_process = pd.DataFrame(
        data={"epoch": range(epoch + 1),  # 必须加1，才能和下面的train_loss_all等长度一致
              "lr_list": lr_list,
              "train_loss_all": train_loss_all,
              # "val_loss_all": val_loss_all,
              "train_acc_all": train_acc_all,
              "val_acc_all": val_acc_all})
    return model, train_process


# 导入网络
from NET.zfnet import ZFNet
from NET.DRSN import DRSN
from NET.resnet import ResNet
from vit_pytorch.efficient import ViT
from nystrom_attention import Nystromformer

if __name__ == "__main__":
    num_epochs = 30  # epoch总数
    test_size = 0.3
    lr = 0.005
    X, Y = data_process("/home/data/jjg1/data/RML2016.10a_dict.pkl")  # 数据读取预处理
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size, random_state=1)  # 划分训练集、测试集
    X_train, X_test = to_transpose(X_train, X_test, 128)  # 将数据转换成I/Q
    train_data = Data.TensorDataset(X_train, Y_train)  # 将训练集转化为张量后,使用TensorDataset将X和Y整理到一起
    train_loader = Data.DataLoader(
        dataset=train_data,  # 使用的数据集
        batch_size=256,  # 批处理样本大小
        shuffle=True,  # 每次迭代前打乱数据
        num_workers=2,  # 使用1个进程  只能开一个
    )  # 定义一个数据加载器，将训练数据集进行批量处理
    test_data = Data.TensorDataset(X_test, Y_test)  # 将训练集转化为张量后,使用TensorDataset将X和Y整理到一起
    test_loader = Data.DataLoader(
        dataset=test_data,  # 使用的数据集
        batch_size=256,  # 批处理样本大小
        shuffle=True,  # 每次迭代前打乱数据
        num_workers=2,  # 使用1个进程  只能开一个
    )  # 定义一个数据加载器，将训练数据集进行批量处理
    efficient_transformer = Nystromformer(
        dim=512,
        depth=6,
        heads=8,
        num_landmarks=256
    )

    v = ViT(
        dim=512,
        image_size=128,
        patch_size=2,
        num_classes=11,
        transformer=efficient_transformer
    )

    net = DRSN(34)
    # net = torch.load('/home/data/jjg1/rnn/2016apkl/cs.pkl')
    net = torch.nn.DataParallel(net)
    net = net.to(device)  # 将模型放入GPU

    # 模型训练
    optimizer = torch.optim.Adam(net.parameters(), lr)
    # criterion = GE2ELoss().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.RMSprop(net.parameters(), lr, alpha=0.9)
    patience = 50

    net, train_process = train_model(net, train_loader, test_loader, 0.75, criterion, optimizer, num_epochs)
    print("模型训练完毕")

    # 绘图
    # 图片一：可视化学习率
    # plot_pic(train_process)

    acc = {}
    net.eval()

    '''
    for snr in snrs:
        test_SNRs = map(lambda x: lbl[x], test_idx)
        test_SNRs = list(test_SNRs)
        test_X_i = X_test1[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test1[np.where(np.array(test_SNRs) == snr)]
        test_data1 = Data.TensorDataset(test_X_i, test_Y_i)
        test_loader1 = Data.DataLoader(
            dataset=test_data1,  # 使用的数据集
            batch_size=128,  # 批处理样本大小
            shuffle=True,  # 每次迭代前打乱数据
            num_workers=2,  # 使用1个进程  只能开一个
        )
        val_num = 0
        val_corrects = 0
        with torch.no_grad():
            for data in test_loader1:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                val_num += labels.size(0)
                val_corrects += (predicted == labels).sum().item()
        acc[snr] = val_corrects / val_num
        print(acc[snr])
    # 图片三：测试集准确度随信噪比的变化曲线
    plt.figure(3)
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Picture 3 Classification accuracy of model")
    plt.yticks(np.linspace(0, 1, 11))
    '''
    correct = 0
    total = 0
    truesz = []
    predsz = []
    for snr in snrs:
        if snr >= 0:
            test_SNRs = map(lambda x: lbl[x], test_idx)
            test_SNRs = list(test_SNRs)
            test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
            test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
            test_data1 = Data.TensorDataset(test_X_i, test_Y_i)
            test_loader1 = Data.DataLoader(
                dataset=test_data1,  # 使用的数据集
                batch_size=128,  # 批处理样本大小
                shuffle=True,  # 每次迭代前打乱数据
                num_workers=2,  # 使用1个进程  只能开一个
            )
            val_num = 0
            val_corrects = 0
            with torch.no_grad():
                for data in test_loader1:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_num += labels.size(0)
                    total += labels.size(0)
                    val_corrects += (predicted == labels).sum().item()
                    correct += (predicted == labels).sum().item()
                    numpy1 = labels.cpu().numpy()
                    truesz = np.hstack((truesz, numpy1))
                    numpy = predicted.cpu().numpy()
                    predsz = np.hstack((predsz, numpy))
            acc[snr] = val_corrects / val_num
            print(acc[snr])

    print('0-18db精度: %.3f %%' % (100 * correct / total))
    y_true = truesz
    y_pred = predsz
    C = confusion_matrix(y_true, y_pred)
    confusion = np.array(C)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(rotation=45)
    classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4,
               'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7, 'PAM4': 8, 'QPSK': 9, 'AM-SSB': 10}
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.title('Confusion matrix(0-18dB)')
    # 显示数据
    for first_index in range(len(confusion)):  # 第几行
        for second_index in range(len(confusion[first_index])):  # 第几列
            # plt.text(first_index, second_index, round(confusion[second_index][first_index]/np.sum(confusion[second_index]), 2),
            plt.text(first_index, second_index, confusion[second_index][first_index],
                     fontsize=8, verticalalignment='center', horizontalalignment='center')
    # 在matlab里面可以对矩阵直接imagesc(confusion)
    # 显示
    plt.show()


    # 图片三：测试集准确度随信噪比的变化曲线
    snr = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    plt.figure(3)
    plt.plot(snr, list(map(lambda x: acc[x], snr)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Picture 3 Classification accuracy of model")
    plt.yticks(np.linspace(0, 1, 11))
    # 图片四：计算混淆矩阵并可视化

    plot_Matrix(test_loader)
    print("程序运行完毕")
