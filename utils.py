import torch.nn as nn

class DNN(nn.Module):
  def __init__(self, layer_list):
    super(DNN, self).__init__()
    self.layer_list = layer_list
    self.layers = nn.ModuleList(self.layer_list)

  def forward(self, inputs):
    for i in range(len(self.layer_list)):
      inputs = self.layers[i](inputs)
    return inputs


class Branch(nn.Module):
	def __init__(self, layer):
		super(Branch, self).__init__()
		if(layer is not None):
			self.layer = nn.Sequential(*layer)
		else:
			self.layer = layer
	def forward(self, x):
		if(self.layer is None):
			return x
		else:
			return self.layer(x)

def norm():
  norm_layer = [nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LocalResponseNorm(size=3, alpha=5e-05, beta=0.75)]
  return norm_layer

conv = lambda n: [nn.Conv2d(n, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)]
cap =  lambda n: [nn.MaxPool2d(kernel_size=3), Flatten(), nn.Linear(n, 10)]

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        x = x.view(x.size(0), -1)
        return x


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x
