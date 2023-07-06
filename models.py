import torch.nn as nn

class NoiseNet(nn.Module):
    def __init__(self,in_channels:int = 1) -> None:
        super().__init__()
        self.conv_layer = nn.Sequential(
            # [,1,128,128]
             nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=3),
             nn.ReLU(),
            # [,64,126,126]
             nn.AvgPool2d(2),
            # [,64,63,63]
             nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2),
             nn.ReLU(),
            # [,128,30,30]
             nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3),
            # [,128,28,28]
        )
        self.relu = nn.ReLU()
        self.line_layer = nn.Sequential(
            nn.Linear(in_features=29*29*128,out_features=512),
            nn.Sigmoid(),
            nn.Linear(in_features=512,out_features=100),
            nn.Sigmoid(),
            nn.Linear(in_features=100,out_features=2)
        )
    def forward(self,x):
        batch_size = x.size(0)
        y = self.conv_layer(x)
        y = self.relu(y)
        y = y.view(batch_size,-1)
        y = self.line_layer(y)
        return y