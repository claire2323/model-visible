import torch
import torch.nn as nn
from torchsummary import summary

#以Alexnet为例
class Alexnet(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(3,96,kernel_size=11,stride=4,padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3,stride=2),
      nn.Conv2d(96,256,kernel_size=5,padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3,stride=2),
      nn.Conv2d(256,384,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.Conv2d(384,384,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.Conv2d(384,256,kernel_size=3,padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3,stride=2),
      nn.Flatten(),
      nn.Linear(256*5*5,4096),nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(4096,2000),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(2000,10),
    )
  def forward(self,x):
    return self.net(x)
