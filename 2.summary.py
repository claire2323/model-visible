from torchsummary import summary
from model import *

net = Alexnet()
net = net.cuda()
#出问题时会报错
print(summary(net,(3,224,224))
