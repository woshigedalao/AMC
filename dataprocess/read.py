import numpy as np
import torch
from numpy import mat, multiply
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt

# 数据转换 6
def to_transpose(X_train, X_test, nsamples):

    X_train_cmplx = X_train[:, 0, :] + 1j * X_train[:, 1, :]
    X_test_cmplx = X_test[:, 0, :] + 1j * X_test[:, 1, :]
    # 时域
    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:, 1, :], X_train[:, 0, :]) / np.pi

    # 频域
    fft_point = 128
    window = mat(np.hanning(fft_point))
    temp_data = multiply(window, X_train_cmplx)

    dat_out_amp = fftshift(abs(fft(temp_data)))
    dat_out_ang = fftshift(np.angle(fft(temp_data)))

    max_dat = dat_out_amp.max(axis=1)
    max1 = max_dat.reshape((77000, 1))
    dat_out_amp = 20 * np.log10(dat_out_amp)
    dat_out_amp = dat_out_amp - max1 + 60
    a = dat_out_amp.min(axis=1)
    a = a.reshape((77000, 1))
    dat_out_amp = dat_out_amp - a

    dat_out_amp = np.reshape(dat_out_amp, (-1, 1, nsamples))
    dat_out_ang = np.reshape(dat_out_ang, (-1, 1, nsamples))

    X_train_amp = np.reshape(X_train_amp, (-1, 1, nsamples))
    X_train_ang = np.reshape(X_train_ang, (-1, 1, nsamples))

    # 合并
    # X_train = np.concatenate((dat_out_amp, dat_out_ang, X_train_amp, X_train_ang, X_train), axis=1)
    a = np.concatenate((dat_out_amp, dat_out_ang), axis=1)
    b = np.concatenate((X_train_amp, X_train_ang), axis=1)

    X_train = np.array([a, b, X_train])
    X_train = X_train.transpose((1, 0, 2, 3))

    X_train = torch.from_numpy(X_train.astype(np.float32))  # 将数据转换成pytorch能够读取的类型

    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:, 1, :], X_test[:, 0, :]) / np.pi

    temp_data1 = multiply(window, X_test_cmplx)

    dat_out_amp1 = fftshift(abs(fft(temp_data1)))
    dat_out_ang1 = fftshift(np.angle(fft(temp_data1)))

    max_dat1 = dat_out_amp1.max(axis=1)
    max11 = max_dat1.reshape((33000, 1))
    dat_out_amp1 = 20 * np.log10(dat_out_amp1)
    dat_out_amp1 = dat_out_amp1 - max11 + 60
    a = dat_out_amp1.min(axis=1)
    a = a.reshape((33000, 1))
    dat_out_amp1 = dat_out_amp1 - a

    dat_out_amp1 = np.reshape(dat_out_amp1, (-1, 1, nsamples))
    dat_out_ang1 = np.reshape(dat_out_ang1, (-1, 1, nsamples))

    X_test_amp = np.reshape(X_test_amp, (-1, 1, nsamples))
    X_test_ang = np.reshape(X_test_ang, (-1, 1, nsamples))

    #X_test = np.concatenate((dat_out_amp1, dat_out_ang1, X_test_amp, X_test_ang, X_test), axis=1)
    c = np.concatenate((dat_out_amp1, dat_out_ang1), axis=1)
    d = np.concatenate((X_test_amp, X_test_ang), axis=1)

    X_test = np.array([c, d, X_test])
    X_test = X_test.transpose((1, 0, 2, 3))
    X_test = torch.from_numpy(X_test.astype(np.float32))  # 将数据转换成pytorch能够读取的类型


    print(X_train.shape)
    return (X_train, X_test)


'''
# 数据转换 4
def to_transpose(X_train, X_test, nsamples):
    X_train_cmplx = X_train[:, 0, :] + 1j * X_train[:, 1, :]
    X_test_cmplx = X_test[:, 0, :] + 1j * X_test[:, 1, :]

    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:, 1, :], X_train[:, 0, :]) / np.pi

    X_train_amp = np.reshape(X_train_amp, (-1, 1, nsamples))
    X_train_ang = np.reshape(X_train_ang, (-1, 1, nsamples))

    X_train = np.concatenate((X_train_amp, X_train_ang, X_train), axis=1)
    X_train = torch.from_numpy(X_train.astype(np.float32))  # 将数据转换成pytorch能够读取的类型

    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:, 1, :], X_test[:, 0, :]) / np.pi

    X_test_amp = np.reshape(X_test_amp, (-1, 1, nsamples))
    X_test_ang = np.reshape(X_test_ang, (-1, 1, nsamples))

    X_test = np.concatenate((X_test_amp, X_test_ang, X_test), axis=1)
    X_test = torch.from_numpy(X_test.astype(
        np.float32))  # 将数据转换成pytorch能够读取的类型
    print(X_train.shape)
    return (X_train, X_test)
'''

# 画学习率和损失图
def plot_pic(train_process):
    plt.figure(1)
    plt.plot(train_process.epoch, train_process.lr_list)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("Picture 1 learning rate")

    # 图片二：可视化模型训练过程中的损失函数
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all, "ro-", label="Train loss")
    #plt.plot(train_process.epoch, train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss", rotation=90)
    plt.subplot(1, 2, 2)
    # 可视化模型训练过程中的精度
    plt.plot(train_process.epoch, train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process.epoch, train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc", rotation=90)
    plt.legend()
    plt.suptitle("Picture 2 Loss and accuracy during training")
    plt.show()