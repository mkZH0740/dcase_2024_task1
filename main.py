from lightning.pytorch.cli import LightningCLI
import torch

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cli = LightningCLI()
