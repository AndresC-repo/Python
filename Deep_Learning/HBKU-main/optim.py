# Optimizer
import torch.optim as optim
from torch.optim import lr_scheduler


def get_optim(model_ft, lr):
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr)
    # Scheduler for linear learning rate reduction
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    return optimizer_ft, scheduler
