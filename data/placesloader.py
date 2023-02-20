import torch
import numpy as np
from random import randint
from skimage import color, io
import torchvision
import csv

class InputTransCombine():
    def __init__(self, crop_size = 224):
        self.crop_size = crop_size

    def __call__(self, image):
        assert len(image.shape) == 3

        # print(f'Image Shape : {image.shape}')
        h, w, _ = image.shape
        
        assert min(h, w) >= self.crop_size

        offset_h = randint(0, h-self.crop_size)
        offset_w = randint(0, w-self.crop_size)

        image = image[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size]

        img_lab = color.rgb2lab(image) # H, W, 3

        # print(f' L Channel {img_lab[:, :, :1].min()} {img_lab[:, :, :1].max()}') # [0, 100]
        # print(f' A Channel {img_lab[:, :, 1:2].min()} {img_lab[:, :, 1:2].max()}') # [-128, 127]
        # print(f' B Channel {img_lab[:, :, 2:].min()} {img_lab[:, :, 2:].max()}') #[-128, 127]

        # Norm to [0, 1]
        img_lab[:, :, :1] = img_lab[:, :, :1] / 100.0
        img_lab[:, :, 1:] = (img_lab[:, :, 1:] + 128.0) / 256.0

        img_lab_t = np.transpose(img_lab, (2, 0, 1)).astype(np.float32)
        img_tensor = torch.from_numpy(img_lab_t)

        (L_chan, ab_chan) = img_tensor[:1, :, :], img_tensor[1:, :, :]

        return (L_chan, ab_chan)

class ImageFolderDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, crop_size = 224) -> None:
        super().__init__(root = root, loader = io.imread)
        self.trans = InputTransCombine(crop_size)

    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        L_chan, ab_chan = self.trans(image)
        return L_chan, ab_chan
    

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, crop_size = 224) -> None:
        super().__init__()
        self.pic_list = []

        with open(root+'imgList.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile, dialect='excel')
            for name in csvreader:
                self.pic_list.append(root + name[0])
                # print(f'Add Pic {root + name[0]}')
                # exit()
            
        self.trans = InputTransCombine(crop_size)

    def __getitem__(self, idx):
        image = io.imread(self.pic_list[idx])
        L_chan, ab_chan = self.trans(image)
        return L_chan, ab_chan
    


if __name__ == "__main__":
    print(__file__)
    testset = ImageDataset('./data/Places205/testset/')
    for i in range(32):
        a = torch.cat(testset[i], dim=0)
        print(f'A.shape = {a.shape}')
