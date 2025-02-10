import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.regression import mean_absolute_error
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from config import LoadConfig
from data import CustomDataLoader
from models import model_registry

warnings.filterwarnings('ignore')


class TrainingContext:
    def __init__(self, option):
        self.option = option
        self.accelerator = Accelerator(log_with='wandb') if option.OPTIM.WANDB else Accelerator()

    def __enter__(self):
        return self.accelerator

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.accelerator.end_training()


class MetricsCalculator:
    def __init__(self, device):
        self.criterion_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
        self.device = device

    def calculate_metrics(self, res, tar):
        psnr = peak_signal_noise_ratio(res, tar, data_range=1)
        ssim = structural_similarity_index_measure(res, tar, data_range=1)
        mae = mean_absolute_error((res * 255).flatten(), (tar * 255).flatten())
        lpips = self.criterion_lpips(res, tar)
        return psnr, ssim, mae, lpips


def seed_everything(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model, weight_path, optimizer=None, device=None):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weight file not found: {weight_path}")

    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    model.to(device)
    epoch = checkpoint.get('epoch', 0)
    return model, epoch, optimizer


def create_data_loader(option, data_dir, mode, batch_size, num_workers, shuffle=True):
    dataset = CustomDataLoader(
        data_dir,
        img_options={'w': option.TRAINING.PS_W, 'h': option.TRAINING.PS_H, 'mode': mode}
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )


def validate(model, val_loader, accelerator, calculator):
    metric_psnr = metric_ssim = metric_mae = metric_lpips = 0
    size = len(val_loader)
    for test_data in tqdm(val_loader, disable=not accelerator.is_local_main_process):
        inp, tar = test_data[0].contiguous(), test_data[1]
        with torch.no_grad():
            res = model(inp).clamp(0, 1)
        res, tar = accelerator.gather((res, tar))
        psnr, ssim, mae, lpips = calculator.calculate_metrics(res, tar)
        metric_psnr += psnr
        metric_ssim += ssim
        metric_mae += mae
        metric_lpips += lpips
    return metric_psnr / size, metric_ssim / size, metric_mae / size, metric_lpips / size


def train(option):
    with TrainingContext(option) as accelerator:
        device = accelerator.device
        if accelerator.is_local_main_process:
            os.makedirs(option.TRAINING.SAVE_DIR, exist_ok=True)

        accelerator.init_trackers(option.MODEL.SESSION + "-" + option.MODEL.MODEL, config=option)

        # Data Loaders
        train_loader = create_data_loader(option, data_dir=option.TRAINING.TRAIN_DIR, mode='train',
                                          batch_size=option.OPTIM.BATCH_SIZE, num_workers=16, shuffle=True)
        val_loader = create_data_loader(option, data_dir=option.TRAINING.VAL_DIR, mode='test',
                                        batch_size=1, num_workers=8, shuffle=False)

        # Model & Loss
        model = model_registry.get(option.MODEL.MODEL)()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=option.OPTIM.LR_INITIAL,
            betas=(option.OPTIM.BETA1, option.OPTIM.BETA2),
            eps=1e-8
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, option.OPTIM.NUM_EPOCHS,
                                                         eta_min=option.OPTIM.LR_MIN)

        if option.TRAINING.RESUME:
            model, epoch, optimizer = load_model(model, option.TRAINING.WEIGHT, optimizer, device)
            for i in range(1, epoch):
                scheduler.step()

        # Prepare with Accelerator
        train_loader, val_loader, model, optimizer, scheduler = accelerator.prepare(
            train_loader, val_loader, model, optimizer, scheduler
        )

        best_epoch, best_psnr = 1, 0
        criterion_psnr = torch.nn.MSELoss()
        calculator = MetricsCalculator(device)

        # Training Loop
        for epoch in range(1, option.OPTIM.NUM_EPOCHS + 1):
            model.train()
            for data in tqdm(train_loader, disable=not accelerator.is_local_main_process):
                inp, tar = data[0].contiguous(), data[1]
                optimizer.zero_grad()
                res = model(inp).clamp(0, 1)
                loss_psnr = criterion_psnr(res, tar)
                loss_ssim = 1 - structural_similarity_index_measure(res, tar, data_range=1)
                train_loss = loss_psnr + 0.2 * loss_ssim
                accelerator.backward(train_loss)
                optimizer.step()
            scheduler.step()

            # Validation
            if epoch % option.TRAINING.VAL_AFTER_EVERY == 0:
                model.eval()
                metric_psnr, metric_ssim, metric_mae, metric_lpips = validate(model, val_loader, accelerator,
                                                                              calculator)

                if metric_psnr > best_psnr and accelerator.is_local_main_process:
                    best_epoch, best_psnr = epoch, metric_psnr
                    checkpoint_file = os.path.join(option.TRAINING.SAVE_DIR, f"{option.MODEL.MODEL}_best.pth")
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, checkpoint_file)

                if accelerator.is_local_main_process:
                    accelerator.log({
                        "PSNR": metric_psnr,
                        "SSIM": metric_ssim,
                        "MAE": metric_mae,
                        "LPIPS": metric_lpips
                    }, step=epoch)
                    print(f"Epoch: {epoch}, PSNR: {metric_psnr}, SSIM: {metric_ssim}, "
                          f"Best PSNR: {best_psnr}, Best Epoch: {best_epoch}")
                    checkpoint_file = os.path.join(option.TRAINING.SAVE_DIR, f"{option.MODEL.MODEL}_latest.pth")
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, checkpoint_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model using a specified YAML configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    opt = LoadConfig(args.config)
    seed_everything(opt.OPTIM.SEED)
    train(opt)
