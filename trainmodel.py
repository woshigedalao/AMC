import csv
import datetime as dt
import time
import pandas as pd
import torch


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

    #early_stopping = EarlyStopping(patience=patience, verbose=True)
    since = time.time()

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
        with open("/home/data/jjg1/rnn/2016adr34.csv", "a", newline='') as csvfile:
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
            torch.save(model, '/home/data/jjg1/rnn/2016apkl/dr34.pkl', _use_new_zipfile_serialization=False)
        else:
            print('[{}] 训练损失: {:.4f}  训练精度: {:.2f}%  测试精度: {:.2f}%'
                  .format(epoch + 1, train_loss_all[-1], train_acc_all[-1] * 100, val_acc_all[-1] * 100))

        print("本轮用时 {:.0f}m {:.5f}s  总共用时 {:.0f}m {:.5f}s 当前时间"
              .format(every_time_use // 60, every_time_use % 60, total_time_use // 60,
                      total_time_use % 60) + '   ' + now_time)

       # early_stopping(correct / total, model)

    # 使用最好模型的参数
    model = torch.load('/home/data/jjg1/rnn/2016apkl/dr34.pkl')  # 加载模型

    train_process = pd.DataFrame(
        data={"epoch": range(epoch + 1),  # 必须加1，才能和下面的train_loss_all等长度一致
              "lr_list": lr_list,
              "train_loss_all": train_loss_all,
              #"val_loss_all": val_loss_all,
              "train_acc_all": train_acc_all,
              "val_acc_all": val_acc_all})
    return model, train_process