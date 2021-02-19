import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal
from networks.base import Function

POLICY = ['gaussian', 'softmax', 'deterministic']

class PolicyNetwork(nn.Module):
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

class SoftmaxPolicyNetwork(PolicyNetwork):
    def __init__(self, ob_dim, ac_dim, hidden):
        super().__init__(ob_dim, ac_dim, hidden)

    def forward(self, x):
        out = super().forward(x)
        probs = F.softmax(out, dim=-1)
        if not self.training:
            return torch.argmax(probs, dim=-1)
        else:
            return Categorical(probs)

class GaussianPolicyNetwork(PolicyNetwork):
    def __init__(self, ob_dim, ac_dim, max_ac, min_ac, hidden):
        super().__init__(ob_dim, 2 * ac_dim, hidden)
        self._center = torch.tensor(max_ac + min_ac) / 2
        self._scale = torch.tensor(max_ac - min_ac) / 2

    def forward(self, x):
        out = super().forward(x)
        ac_dim = out.shape[-1] // 2
        means = self._squash(out[:, :ac_dim])
        if not self.training:
            return means
        else:
            logvars = out[:, ac_dim:] * self._scale
            std = logvars.exp_()
            return Independent(Normal(means, std), 1)
    
    def _squash(self, x):
        return torch.tanh(x) * self._scale + self._center

    def to(self, device):
        self._center = self._center.to(device)
        self._scale = self._scale.to(device)
        return super().to(device)

class DeterministicPolicyNetwork(PolicyNetwork):
    def __init__(self, ob_dim, ac_dim, max_ac, min_ac, hidden):
        super().__init__(ob_dim, ac_dim, hidden)
        self._tanh_mean = torch.tensor(max_ac + min_ac) / 2
        self._tanh_scale = torch.tensor(max_ac - min_ac) / 2

    def forward(self, x):
        out = super().forward(x)
        return self._squash(out)

    def _squash(self, x):
        return torch.tanh(x) * self._tanh_scale + self._tanh_mean

    def to(self, device):
        self._tanh_mean = self._tanh_mean.to(device)
        self._tanh_scale = self._tanh_scale.to(device)
        return super().to(device)


class Policy(Function):
    def __init__(self, ob_dim, ac_dim, policy, max_ac=None, min_ac=None, hidden=[64,64], lr=0.0005, target=False):

        if policy == 'deterministic':
            net = DeterministicPolicyNetwork(ob_dim, ac_dim, max_ac, min_ac, hidden)
        elif policy == 'gaussian':
            net = GaussianPolicyNetwork(ob_dim, ac_dim, max_ac, min_ac, hidden)
        elif policy == 'softmax':
            net = SoftmaxPolicyNetwork(ob_dim, ac_dim, hidden)
        else: 
            raise ValueError

        opt = optim.Adam(net.parameters(), lr=lr)

        super().__init__(net, opt, target)
