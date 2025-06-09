import math
import torch.optim as optim
 
class WarmupLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, gamma = 5e-3, last_epoch=-1):
        """
        https://blog.csdn.net/Freeandeasy_roni/article/details/129047249
        optimizer: 优化器对象
        warmup_steps: 学习率线性增加的步数
        gamma: 学习率下降系数
        last_epoch: 当前训练轮数
        """
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
 
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 学习率线性增加
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # 学习率按指数衰减
            #return [base_lr * math.exp(-(self.last_epoch - self.warmup_steps + 1) * self.gamma) for base_lr in self.base_lrs]
            return self.base_lrs
 