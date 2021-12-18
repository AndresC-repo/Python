"""
Easy straight forward implemntation of classic CNN network for YOLO
Some changes from the original paper were done to make it easier to run:
- lin1 outputs 4096 but for simple testing here its 496
- LeakyReLU is implemented instead of simple ReLU
- Dropout prob is 0

Code is a bit big and could be done smaller by using nn.Sequential
and/or a config architecture file to run in sequence as most blocks are often repeated
For now I just wanted to make it clear. Will probably be changed in the future.
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
	def __init__(self,in_channels, out_channels, kernel_size, stride, padding):
		super(ConvBlock, self).__init__()
		self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
		self.batchnorm = nn.BatchNorm2d(out_channels)
		self.aFunc = nn.LeakyReLU(0.1)

	def forward(self, x):
		return(self.aFunc(self.batchnorm(self.cnn(x))))


class YoloNet(nn.Module):
	def __init__(self, in_c, S, C, B):
		super(YoloNet, self).__init__()
		self.Block1 = ConvBlock(in_channels=in_c, out_channels=64, kernel_size=7, stride=2, padding=3)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		# ---- # 
		self.Block2 = ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
		#self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
		# ---- # 
		self.Block3 = ConvBlock(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0)
		self.Block4 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
		self.Block5 = ConvBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)
		self.Block6 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
		#self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
		# ---# 
		self.Block7 = ConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
		self.Block8 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
		# - 1
		self.Block9 = ConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
		self.Block10 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
		# - 2
		self.Block11 = ConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
		self.Block12 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
		# - 3
		self.Block13 = ConvBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
		self.Block14 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
		# - 4
		self.Block15 = ConvBlock(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
		self.Block16 = ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
		#self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
		# ---- # 
		self.Block17 = ConvBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
		self.Block18 = ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
		self.Block19 = ConvBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
		self.Block20 = ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
		self.Block21 = ConvBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0)
		self.Block22 = ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
		self.Block23 = ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
		self.Block24 = ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)
		# ---- # 
		self.Block25 = ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
		self.Block26 = ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
		# ---- #
		self.flat = nn.Flatten()
		self.lin1 = nn.Linear(1024 * S * S, 496)
		self.drop = nn.Dropout(0.0)
		self.actf = nn.LeakyReLU(0.1)
		self.lin2 = nn.Linear(496, S * S * (C + B * 5))

	def forward(self, x):
		x = self.pool(self.Block1(x))
		x = self.pool(self.Block2(x))
		x = self.pool(self.Block6(self.Block5(self.Block4(self.Block3(x)))))
		x = self.Block8(self.Block7(x))
		x = self.Block10(self.Block9(x))
		x = self.Block12(self.Block11(x))
		x = self.Block14(self.Block13(x))
		x = self.pool(self.Block16(self.Block15(x)))
		x = self.Block20(self.Block19(x))
		x = self.Block22(self.Block21(x))
		x = self.Block24(self.Block23(x))
		x = self.Block26(self.Block25(x))
		x = self.lin2(self.actf(self.drop(self.lin1(self.flat(x)))))
		return x

if __name__ == "__main__":
	S=7
	C=20
	B=2
	network = YoloNet(in_c=3, S=7, C=20, B=2)
	image = torch.rand([1, 3, 448, 448])
	"""
	torch.Size([1, 64, 112, 112])
	torch.Size([1, 192, 56, 56])
	torch.Size([1, 512, 28, 28])
	torch.Size([1, 512, 28, 28])
	torch.Size([1, 512, 28, 28])
	torch.Size([1, 512, 28, 28])
	torch.Size([1, 512, 28, 28])
	torch.Size([1, 1024, 14, 14])
	torch.Size([1, 1024, 14, 14])
	torch.Size([1, 1024, 14, 14])
	torch.Size([1, 1024, 7, 7])
	torch.Size([1, 1024, 7, 7])
	"""
	testShape = torch.rand([1, S*S*(C+B*5)])
	print(network(image).shape)
	assert(network(image).shape == testShape.shape)







