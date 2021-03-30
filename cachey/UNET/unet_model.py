


import torch.nn.functional as F

from .unet import *



class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, bilinear = True, encode = True):
		super(UNET, slef).__init__()

		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		self.inc = MultilayerConv(n_channels, 84);

		#find the correct matrix to downsample in 13 layer unet

    if encode:
    	self.down1 = Down(84,128)
    	self.down2 = Down(128,256)
    	self.down3 = Down(256, 512)
    	self.down4 = Down(512,256)
    	self.down5 = Down(256, 128)
    	self.out = OutConv(128,128)
	else:
		self.down1 = Down(64, 128)
		self.down1 = down(64, 128)
		self.down2 = down(128, 256)
		self.down3 = down(256, 512)
		self.down4 = down(512, 512)
		self.up1 = up(1024, 256)
		self.up2 = up(512, 128)
		self.up3 = up(256, 64)
		self.up4 = up(128, 64)
		self.outc = outconv(64, n_classes)



		


	def forward(self,x):
		x1 = self.inc(x)
		x2 = self.Down1(x1)
		x3 = self.Down(x2)
		x4 = self.Down(x3)
		x5 = self.Down(x4)
		x = self.out(x5)
		logits = self.out(x)
		return logits

#encoder-decoder
'''	def forward_endc(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		logits = self.out(x)
		return logits'''
