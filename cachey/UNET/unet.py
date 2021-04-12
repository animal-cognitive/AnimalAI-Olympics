

""" This file contains main unet sampling. We can use UNET encoder by using Down(inchannel, outchannel) in channel is imput size and outchannel is desired size """


import torch
import torch.nn as nn
import torch.nn.functional as F



class MultilayerConv(nn.Module):
	

	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if mid_channels==None:
			mid_channels = out_channels
		self.Multilayer_Conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			
			)
	def forward(self, x):
		return self.Multilayer_Conv(x)



class Down(nn.Module):
	

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool3d(2),
			MultilayerConv(in_channels,out_channels),
		)


	def forward(self, x):
		return self.maxpool_conv(x)




class up(nn.Module):
	

	def __init__(self, in_channels, bilinear=True):
		super().__init__()

		if bilinear:
			self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
			self.conv = MultilayerConv(in_channels, out_channels, in_channels//2)
		else:
			self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
			self.conv = MultilayerConv(in_channels, out_channels)


	def forward(self, x1, x2):
	
		x1 = self.up(x1)

		diffy = x2.size()[2] - x1.size()[2]
		diffx = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffx //2, diffx-diffx//2, diffy//2, diffy-diffy//2])
		x = torch.cat([x2,x1], dim=1)
		return self.conv(x)



class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)


	def forward(self,x):
		return self.conv(x)
