from tensorboardX import SummaryWriter
from model import *
import torch

net = Alexnet()
net = net.cuda()
img = torch.rand((1,3,224,224))
img = img.cuda()
with SummaryWriter(log_dir = 'logs') as w:
#dim=0 为行 dim=1为列
  w.add_graph(net,img)
