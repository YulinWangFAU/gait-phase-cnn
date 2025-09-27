import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, verbose=True, min_delta=0.0, mode='min', path='checkpoint.pt'):
        self.patience = patience #如果连续 5 个 epoch 没有提升，就停止训练
        self.verbose = verbose  # 是否打印日志
        self.min_delta = min_delta  #最小改善幅度，只有超过这个值才算“有提升”
        self.mode = mode  #  监控的指标是最小化还是最大化（比如 Loss 要 min，Accuracy 要 max）
        self.counter = 0  # 记录连续多少次没有提升
        self.early_stop = False  #提前停止标志
        self.path = path   #最佳模型保存路径

        if self.mode == 'min':
            self.best_score = np.inf
            self.monitor_op = lambda current, best: current < best - self.min_delta
        elif self.mode == 'max':
            self.best_score = -np.inf
            self.monitor_op = lambda current, best: current > best + self.min_delta
        else:
            raise ValueError("mode should be 'min' or 'max'")

    def __call__(self, current_score, model):
        if self.monitor_op(current_score, self.best_score):
            if self.verbose:
                print(f"✅ Metric improved ({self.best_score:.4f} → {current_score:.4f}). Saving model.")
            self.best_score = current_score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
