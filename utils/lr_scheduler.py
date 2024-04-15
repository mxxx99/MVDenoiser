import types
import math
from torch._six import inf
from functools import wraps
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

import math
import torch.optim as optim
 
class WarmupLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
 
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return [base_lr * math.exp(-(self.last_epoch - self.warmup_steps + 1) * self.gamma) for base_lr in self.base_lrs]


class WarmupMultiStepLR(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, milestones, warmup_steps, gamma=0.1, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
            
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            if self.last_epoch not in self.milestones:
                return [group['lr'] for group in self.optimizer.param_groups]
            return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                    for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        milestones = list(sorted(self.milestones.elements()))
        return [base_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
                for base_lr in self.base_lrs]
    

if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    optimizer_G = torch.optim.Adam([{'params':torch.ones(1),'initial_lr':0.0002}], lr=0.0002)
    scheduler_G = WarmupMultiStepLR(optimizer_G, warmup_steps=20,milestones=[200,300,400],gamma=0.3162277,last_epoch=300)
    lrs=[]
    epochs=[]
    for epoch in range(500):
        scheduler_G.step()
        epochs.append(epoch)
        lrs.append(scheduler_G.get_lr())
    plt.plot(epochs,lrs)
    plt.savefig('test.png')