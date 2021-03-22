


import torch.nn.functional as F

from .unet import *



class UNET(nn.Module):
	def __init__(self, n_channels, n_classes, bilinear = True):
		super(UNET, slef).__init__()

		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		self.inc = MultilayerConv(n_channles, 4);

		#find the correct matrix to downsample in 13 layer unet
		self.down1 = Down(3,4)
		self.down2 = Down(4,8)


		




def forward(self,x):
	
	logits = self.outc(x)
	return logits
