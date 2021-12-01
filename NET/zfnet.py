import torch
import torch.nn as nn

class ZFNet(nn.Module):
    def __init__(self):
        super(ZFNet, self).__init__()
        self.conv = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 32, 2, 2),     #nn.Conv2d(3, 32, 7, 2),
            nn.ReLU(),
            nn.MaxPool2d(1),     #nn.MaxPool2d(3, 2),
            # 第二次
            nn.Conv2d(32, 64, 2, 2),    #n.Conv2d(32, 64, 5, 2),
            nn.ReLU(),
            nn.MaxPool2d(1),      #nn.MaxPool2d(3, 2),
            # 第三层
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            # 第四层
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            # 第五层
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(1),
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(4096, 4096),     # nn.Linear(65280, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 11),    # nn.Linear(4096, 12),
        )

    def forward(self, img):
        img = img.unsqueeze(1)
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


