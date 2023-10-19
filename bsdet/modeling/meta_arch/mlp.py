import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input, output, hiddens=None, active='none'):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        if hiddens is not None:
            self.layers.append(nn.Linear(input, hiddens[0]))
            if len(hiddens) > 1 and active != 'none':
                if active == 'relu':
                    self.layers.append(nn.ReLU())

            for i in range(1, len(hiddens)):
                self.layers.append(nn.Linear(hiddens[i-1], hiddens[i]))
                if active == 'relu':
                    self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hiddens[-1], output))
        else:
            self.layers.append(nn.Linear(input, output))
        for l in self.layers:
            if isinstance(l, nn.Linear):
                nn.init.normal_(l.weight, std=0.01)
                nn.init.constant_(l.bias, 0)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x