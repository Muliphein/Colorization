import cv2 as cv
from model import CO, LCI
import numpy as np
import torch
import torch.nn as nn
import csv
import os
import argparse
from data import ImageDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage import io, color

INPUT_CASE = './data/CUO_Example/'

def deal_args():
    parser = argparse.ArgumentParser(description='Colorization Setting')
    parser.add_argument('--mode', type=str, default='evaluate')
    parser.add_argument('--model', type=str, default='CO')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--logsdir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="Places")
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    return args


class Training():
    def __init__(self, args) -> None:
        if args.model == "LCI":
            self.model = LCI()
        else:
            print('Not Implement')
            assert(False)

        if args.dataset == "Places":
            self.testset = ImageDataset('./data/Places205/testset/')
            self.trainset = ImageDataset('./data/Places205/trainset/')
        else:
            print('Not Implement')
            assert(False)

        if args.logsdir == None:
            args.logsdir = f'./logs/{args.model}_{args.dataset}/'

        if not os.path.exists(args.logsdir+'ckpt'):
            os.makedirs(args.logsdir+'ckpt')
        if not os.path.exists(args.logsdir+'images-train'):
            os.makedirs(args.logsdir+'images-train')
        if not os.path.exists(args.logsdir+'images-valid'):
            os.makedirs(args.logsdir+'images-valid')

        self.LOG_DIR = args.logsdir
        self.DEVICES = 'cuda:0'
        print(f'Using Devices : {self.DEVICES}')
        self.EPOCH_MAX = args.epoch
        self.model.cuda()
        
        # Load Check Point
        
        self.testloader = DataLoader(self.testset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
        self.trainloader = DataLoader(self.trainset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)

        self.mse_loss = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.now_epoch = 0

        if args.ckpt is not None:
            self.load_ckpt(args.ckpt)
    

    def load_ckpt(self, ckpt):
        checkpoint = torch.load(ckpt, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])
        self.now_epoch = checkpoint['epoch']+1
        print(f'Load CheckPoint {ckpt}')

    def save_ckpt(self):
        ckpt_save_path = self.LOG_DIR + f'ckpt/{self.now_epoch}.pt'
        torch.save(
            {
                'epoch' : self.now_epoch,
                'model' : self.model.state_dict(),
                'optim' : self.optimizer.state_dict()
            }, ckpt_save_path
        )
        print(f'Save CheckPoint to {ckpt_save_path}')

    def write_to_csv(self, filename, items):
        with open(filename, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(items)
        pass

    def save_pic(self, L_chan, ab_chan, ab_out, name):
        # DeNorm
        if L_chan.shape[0] > 16:
            L_chan = L_chan[:16, :, :, :]
            ab_out = ab_out[:16, :, :, :]
            ab_chan = ab_chan[:16, :, :, :]
        L_chan = L_chan.detach().cpu().numpy() * 100.0
        ab_out = ab_out.detach().cpu().numpy() * 256.0 - 128.0
        ab_chan = ab_chan.detach().cpu().numpy() * 256.0 - 128.0
        pic_list = []
        for i in range(L_chan.shape[0]):
            L = L_chan[i]       #[1, h, w]
            ab_pred = ab_out[i] #[2, h, w]
            ab_gt = ab_chan[i]  #[2, h, w]
            # print(f'L shape {L.shape}')
            # print(f'ab_pred shape {ab_pred.shape}')
            # print(f'ab_gt shape {ab_gt.shape}')
            img_pred = np.concatenate([L, ab_pred], axis=0).transpose((1, 2, 0))
            
            # print(f' L Channel {img_pred[:, :, :1].min()} {img_pred[:, :, :1].max()}') # [0, 100]
            # print(f' A Channel {img_pred[:, :, 1:2].min()} {img_pred[:, :, 1:2].max()}') # [-128, 127]
            # print(f' B Channel {img_pred[:, :, 2:].min()} {img_pred[:, :, 2:].max()}') #[-128, 127]

            img_gt = np.concatenate([L, ab_gt], axis=0).transpose((1, 2, 0))
            img_gray_rgb = color.gray2rgb(L[0]) / 100.0
            img_pred_rgb = color.lab2rgb(img_pred)
            # print(f'img_pred_rgb {img_pred_rgb.shape} {img_pred_rgb.min()},{img_pred_rgb.max()}')
            img_gt_rgb = color.lab2rgb(img_gt)
            # print(f'img_gt_rgb {img_gt_rgb.shape}')
            img_gray_pred_gt = np.concatenate([img_gray_rgb, img_pred_rgb, img_gt_rgb], axis=1)
            # print(f'img_gray_pred_gt {img_gray_pred_gt.shape}')
            pic_list.append(img_gray_pred_gt)
        

        if len(pic_list) % 2 == 1:
            pic_list.append(np.zeros_like(pic_list[0]))
        
        double_pic_list = []
        for i in range(len(pic_list)//2):
            double_pic_list.append(np.concatenate([pic_list[i*2], pic_list[i*2+1]], axis=1))

        img_all = (np.concatenate(double_pic_list, axis=0)*255).astype(np.uint8)
        # print(f'img_all {img_all.shape} {img_all.min()},{img_all.max()}')
        
        io.imsave(name, img_all)

    def train_epoch(self):
        epoch_loss = 0.0
        self.model.train()
        with tqdm(total=len(self.trainloader), ncols=80, desc=f'Train:Epoch-{self.now_epoch}') as tbar:
            for batch_id, data in enumerate(self.trainloader):
                L_chan, ab_chan = data
                L_chan = L_chan.to(self.DEVICES)
                ab_chan = ab_chan.to(self.DEVICES)
                self.optimizer.zero_grad()
                ab_out = self.model(L_chan)
                loss = self.mse_loss(ab_chan, ab_out)
                loss.backward()
                self.optimizer.step()
                batch_loss = loss.item()
                self.write_to_csv(self.LOG_DIR+'train_batch.csv', [batch_id, batch_loss])
                epoch_loss += batch_loss * data[0].shape[0]
                tbar.update(1)
                if batch_id % 1000 == 0:
                    self.save_pic(L_chan, ab_chan, ab_out, self.LOG_DIR + f'images-train/{self.now_epoch}_{batch_id}.jpg')

        epoch_loss /= len(self.trainset)
        self.write_to_csv(self.LOG_DIR+'train_epoch_loss.csv', [epoch_loss, ])


    def test_epoch(self):
        epoch_loss = 0.0
        self.model.eval()
        with tqdm(total=len(self.testloader), ncols=80, desc=f'Valid:Epoch-{self.now_epoch}') as tbar:
            for batch_id, data in enumerate(self.testloader):
                L_chan, ab_chan = data
                L_chan = L_chan.to(self.DEVICES)
                ab_chan = ab_chan.to(self.DEVICES)
                ab_out = self.model(L_chan)
                loss = self.mse_loss(ab_chan, ab_out)
                batch_loss = loss.item()
                epoch_loss += batch_loss * data[0].shape[0]
                tbar.update(1)
                if batch_id % 100 == 0:
                    self.save_pic(L_chan, ab_chan, ab_out, self.LOG_DIR + f'images-valid/{self.now_epoch}_{batch_id}.jpg')

        epoch_loss /= len(self.testset)
        self.write_to_csv(self.LOG_DIR+'test_epoch_loss.csv', [epoch_loss, ])

    def run(self):

        while self.now_epoch < self.EPOCH_MAX:
            self.train_epoch()
            self.save_ckpt()
            self.test_epoch()
            self.now_epoch += 1

    pass


if __name__ == "__main__":

    args = deal_args()

    if args.mode == 'evaluate':
        print(f'Evaluate Start on Model {args.model}')
        if args.model == 'CO':
            gray_pic_name = INPUT_CASE + '2.bmp'
            appendix_pic_name = INPUT_CASE + '3.bmp'
            gray_pic = cv.imread(gray_pic_name)[:, :, ::-1].copy()
            appendix_pic = cv.imread(appendix_pic_name)[:, :, ::-1].copy()
            model = CO()
            out_pic = model(gray_pic, appendix_pic)
            out = np.concatenate([gray_pic[:, :, ::-1],appendix_pic[:, :, ::-1],out_pic[:, :, ::-1]], axis = 1)
            cv.imwrite(f'./result/{args.model}.bmp', out)
        else:
            print('Not Implement')
            assert(False)
        print('Result Save to '+f'./result/{args.model}.bmp')
    elif args.mode == 'train':
        if args.model == "LCI":
            runner = Training(args)
            runner.run()
        else:
            print('Not Implement')
            assert(False)
    pass