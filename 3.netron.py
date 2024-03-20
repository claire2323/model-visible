import onnx
import netron
from model import *

net = Alexnet()
net = net.cuda()
img = torrch.rand((1,3,224,224))
img = img.cuda()
torch.onnx.export(model = net,args = img,f = 'model.onnx',input_name = ['image'],output_name = ['feature map'])
onnx.save(onnx.shape_inference.infer_shapes(onnx.load('model.onnx')),'model.onnx')
netron.start("model.onnx")
