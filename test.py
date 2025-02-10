import argparse
import os

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.regression import mean_absolute_error
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import save_image
from tqdm import tqdm

from config import LoadConfig
from data import CustomDataLoader
from models import model_registry


class TestingContext:
    def __init__(self, option):
        self.option = option
        self.accelerator = Accelerator()

    def __enter__(self):
        return self.accelerator

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


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


def create_test_loader(option, test_dir, num_workers):
    test_dataset = CustomDataLoader(
        test_dir,
        img_options={'w': option.TRAINING.PS_W,
                     'h': option.TRAINING.PS_H, 'mode': 'test'}
    )
    return DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )


def test(option):
    with TestingContext(option) as accelerator:
        device = accelerator.device
        # Load model
        model = model_registry.get(option.MODEL.MODEL)
        model, _, _ = load_model(model, option.TESTING.WEIGHT, device=device)

        # Data Loader
        test_loader = create_test_loader(option, test_dir=option.TESTING.TEST_DIR, num_workers=16)

        # Output directory
        result_dir = option.TESTING.RESULT_DIR
        if accelerator.is_local_main_process:
            os.makedirs(result_dir, exist_ok=True)

        model, test_loader = accelerator.prepare(model, test_loader)

        model.eval()
        calculator = MetricsCalculator(device)
        metric_psnr = metric_ssim = metric_mae = metric_lpips = 0
        size = len(test_loader)

        # Testing loop
        for idx, data in enumerate(tqdm(test_loader, disable=not accelerator.is_local_main_process)):
            inp, tar, f_name = data[0], data[1], data[2][0]
            with torch.no_grad():
                res = model(inp).clamp(0, 1)  # Generate prediction

            res, tar = accelerator.gather((res, tar))
            psnr, ssim, mae, lpips = calculator.calculate_metrics(res, tar)
            metric_psnr += psnr
            metric_ssim += ssim
            metric_mae += mae
            metric_lpips += lpips
            # Save output image
            output_path = os.path.join(result_dir, f_name)
            save_image(res, output_path)

        metric_psnr /= size
        metric_ssim /= size
        metric_mae /= size
        metric_lpips /= size

        print('PSNR: ', metric_psnr)
        print('SSIM: ', metric_ssim)
        print('MAE: ', metric_mae)
        print('LPIPS: ', metric_lpips)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a model using a specified YAML configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    opt = LoadConfig(args.config)
    test(opt)
