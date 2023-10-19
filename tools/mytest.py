import torch
import torch.nn as nn
import torch.nn.functional as F

class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(7, 5, bias=False)

    def forward(self, x):
        w = torch.cat([self.p.weight[:3, :].detach(), self.p.weight[3:, :]], dim=0)
        x_ex = x[:, None, :].expand(-1, w.size(0), -1)
        w_ex = w[None, :, :].expand_as(x_ex)
        return F.cosine_similarity(x_ex, w_ex, dim=-1)


a = A()
# 设置优化器
optim = torch.optim.SGD(a.parameters(), lr=0.01)

old_w = a.p.weight.clone().detach()
print(a.p.weight)
running_loss = 0.0
in_data = torch.rand(2, 7)
outputs = a(in_data)
result_loss = (torch.rand(2, 5) - outputs).relu().sum()
optim.zero_grad()# 梯度清零要把网络模型当中每一个 调节 参数梯度 调为0，参数清零
result_loss.backward()# 反向传播求解梯度调用存损失函数的反向传播，求出每个节点的梯度，
optim.step()#   更新权重参数调用优化器，对每个参数进行调优，循环
print(a.p.weight)
print(a.p.weight - old_w)
