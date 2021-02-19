import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from networks.base import Function

class QNetworkDiscrete(nn.Module):
	def __init__(self, ob_dim, ac_dim, hidden):
		super().__init__()
		# hidden layers
		self.fc = nn.ModuleList()
		nodes = [ob_dim] + hidden			
		for i in range(len(hidden)):
			self.fc.append(nn.Linear(nodes[i], nodes[i+1]))
		# output layer
		self.out = nn.Linear(nodes[-1], ac_dim)

	def forward(self, x):
		for fc in self.fc:
			x = F.relu(fc(x))
		x = self.out(x)
		return x 

class QNetworkContinuous(nn.Module):
	def __init__(self, ob_dim, ac_dim, hidden):
		super().__init__()
		# hidden layers
		self.fc = nn.ModuleList()
		nodes = [ob_dim + ac_dim] + hidden			
		for i in range(len(hidden)):
			self.fc.append(nn.Linear(nodes[i], nodes[i+1]))
		# output layer
		self.out = nn.Linear(nodes[-1], 1)

	def forward(self, x, u):
		x = torch.cat([x, u], 1)
		for fc in self.fc:
			x = F.relu(fc(x))
		x = self.out(x)
		return x 


class QFunction(Function):
    def __init__(self, ob_dim, ac_dim, discrete=True, hidden=[64,64], lr=0.0005, target=False):

        if discrete:
            net = QNetworkDiscrete(ob_dim, ac_dim, hidden)
        else:
            net = QNetworkContinuous(ob_dim, ac_dim, hidden)
            
        opt = optim.Adam(net.parameters(), lr=lr)

        super().__init__(net, opt, target)

