import os

import albumentations as A
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class CustomDataLoader(Dataset):
    def __init__(self, root_folder, inp_folder='input', tar_folder='target', img_options=None):
        super(CustomDataLoader, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(root_folder, inp_folder)))
        tar_files = sorted(os.listdir(os.path.join(root_folder, tar_folder)))

        self.inp_filenames = [os.path.join(
            root_folder, inp_folder, x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(
            root_folder, tar_folder, x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        if self.img_options['mode'] == 'train':
            self.transform = A.Compose([
                A.Flip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Rotate(p=0.3),
                A.Transpose(p=0.3),
                A.RandomResizedCrop(
                    height=img_options['h'], width=img_options['w']),
            ],
                additional_targets={
                    'target': 'image',
                }
            )
        else:
            self.transform = A.Compose([
                A.NoOp(),
            ],
                additional_targets={
                    'target': 'image',
                }
            )

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        inp_img = np.array(inp_img)
        tar_img = np.array(tar_img)

        transformed = self.transform(image=inp_img, target=tar_img)

        inp_img = TF.to_tensor(transformed['image'])
        tar_img = TF.to_tensor(transformed['target'])

        filename = os.path.basename(tar_path)

        return inp_img, tar_img, filename
