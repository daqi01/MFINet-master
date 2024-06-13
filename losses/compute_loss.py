import torch
import torch.nn as nn
import copy

class LossL1(nn.Module):
    def __init__(self):
        super(LossL1, self).__init__()
        self.loss = nn.L1Loss()

    def __call__(self, input, target):
        return self.loss(input, target)


class LossL2(nn.Module):
    def __init__(self):
        super(LossL2, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, input, target):
        return self.loss(input, target)

def compute_loss(pose_7d: torch.Tensor, igt: torch.Tensor, tgt: torch.Tensor, src: torch.Tensor):

    l1_criterion = LossL1()
    l2_criterion = LossL2()

    B, N = igt.size()
    lr = torch.add(l1_criterion(pose_7d[:, :4], igt[:, :4]), 4 * l2_criterion(pose_7d[:, 4:], igt[:, 4:])) # (B,1)
    lr = torch.sum(lr) / B

    ld = torch.sum(l2_criterion(tgt[0],tgt[2]) + l2_criterion(tgt[1],tgt[3]) + l2_criterion(src[0], src[2]) + l2_criterion(src[1], src[3]))/B
    return lr + ld*0.001


class computeLoss(nn.Module):
    def __init__(self):
        super(computeLoss, self).__init__()

    def forward(self, pose_7d, igt, tgt, src):
        return compute_loss(pose_7d, igt, tgt, src)

if __name__ == '__main__':
    loss = computeLoss()
    a = torch.randn(10, 7).cuda()
    b = torch.randn(10, 7).cuda()
    c = torch.rand(10, 512, 1024)
    d = torch.rand(10, 512, 1024)

    v = loss(a, b, c, d)