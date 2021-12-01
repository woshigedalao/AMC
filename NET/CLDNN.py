import torch
import torch.nn as nn

class CLDNN(nn.Module):
    def __init__(self):
        super(CLDNN, self).__init__()
        # 定义第一个隐藏层
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((2, 0, 0, 0)),
            # kernel_size[0]--H   kernel_size[1]--W
            nn.Conv2d(3, 128, kernel_size=(1, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # 定义第二个隐藏层
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((2, 0, 0, 0)),
            nn.Conv2d(128, 64, kernel_size=(1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 定义第三个隐藏层
        self.conv3 = nn.Sequential(
            nn.ZeroPad2d((2, 0, 0, 0)),
            nn.Conv2d(64, 32, kernel_size=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # 定义第四个隐藏层
        self.conv4 = nn.Sequential(
            nn.ZeroPad2d((2, 0, 0, 0)),
            nn.Conv2d(32, 16, kernel_size=(1, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # LSTM为长短时神经网络
        self.lstm = nn.LSTM(
            input_size=20,
            hidden_size=100,
            num_layers=1,
            batch_first=True,  # 若为True，则第一个维度为batch
            bidirectional=True  # 如果为True，则为双向RNN；默认为False
        )

        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(220, 50),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(50, 11)
        )

    def forward(self, input):
        # input: [batch, time, features]
        # [batchsize, channel, H, W]----[batchsize, 1, 4, 128] .cuda()不能删除


        # CNN计算结果
        conv1 = self.conv1(input)               # [batchsize, 128, 4, 128]
        conv2 = self.conv2(conv1)               # [batchsize, 64, 4, 128]
        conv3 = self.conv3(conv2)               # [batchsize, 32, 1, 128]
        cnn_out = self.conv4(conv3)             # [batchsize, 16, 1, 128]

        # 融合input和CNN的结果，一起送至RNN
        cnn_out = cnn_out.squeeze(2)            # [batchsize, 16, 128]
        print(cnn_out.shape)
        cnn_out = cnn_out.permute(0, 2, 1)       # [batchsize, 127, 16]

        #input = input.squeeze(1)                # [batchsize, 4, 128]
        input = input.permute(0, 2, 1)          # [batchsize, 128, 4]

        # batch_first: If ``True``, then the input and output tensors are provided as (batch, seq, feature).
        combine_input_cnn = torch.cat(
            [input, cnn_out], dim=2)   # [batchsize, 128, 20]

        rnn_in = combine_input_cnn             # [batchsize, 128, 20]

        # RNN计算结果
        # [batchsize, 128, 200] 如果是双向就要*2  (batch, seq, feature)
        rnn_out, _ = self.lstm(rnn_in)

        # 融合CNN和RNN的结果，一起送至DNN
        # [batchsize, 128, 200+20]
        conbime_cnn_rnn = torch.cat([combine_input_cnn, rnn_out], dim=2)

        z = self.classifier(conbime_cnn_rnn[:, -1, :])  # [batchsize, 11]
        return z
