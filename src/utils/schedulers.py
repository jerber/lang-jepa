import math


class WarmupCosineSchedule:
    """
    I implement a warmup + cosine decay learning rate scheduler.
    During the warmup period, I increase LR from start_lr to ref_lr.
    Then I apply a cosine decay from ref_lr down to final_lr.
    """
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, T_max, final_lr=0.0):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0

    def step(self):
        """
        I advance the scheduler by one step.
        Returns the new learning rate.
        """
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr
        return new_lr


class CosineWDSchedule:
    """
    I implement a cosine decay for weight decay.
    """
    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0

    def step(self):
        """
        I advance the weight decay schedule by one step.
        Returns the new weight decay.
        """
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        # handle direction (in case final_wd > ref_wd or final_wd < ref_wd)
        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd
