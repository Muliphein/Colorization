import cv2 as cv
from model import CO, LCI, P2P, DDPM
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
    parser.add_argument('--p2p_lambda', type=float, default=100.0)
    parser.add_argument('--train_report_batch_freq', type=int, default=2500)
    parser.add_argument('--valid_report_batch_freq', type=int, default=100)
    args = parser.parse_args()
    return args


class Training():
    def __init__(self, args) -> None:
        self.args = args

        # Get Model
        if args.model == "LCI":
            self.model = LCI()
        elif args.model == "P2P":
            self.model = P2P()
        elif args.model == "DDPM":
            self.model = DDPM()
        else:
            raise(NotImplementedError)

        if args.dataset == "Places":
            self.testset = ImageDataset('./data/Places205/testset/', crop_size=args.crop_size)
            self.trainset = ImageDataset('./data/Places205/trainset/', crop_size=args.crop_size)
        else:
            raise(NotImplementedError)

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
        
        # Get Dataset
        
        self.testloader = DataLoader(self.testset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
        self.trainloader = DataLoader(self.trainset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)

        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

        # Set Optim (Model-Based)
        if args.model == "LCI" or args.model == "DDPM":
            self.optimizer = torch.optim.Adam(self.model.parameters())
        elif args.model == "P2P":
            self.optimizerG = torch.optim.Adam(self.model.netG.parameters())
            self.optimizerD = torch.optim.Adam(self.model.netD.parameters())
        else:
            raise(NotImplementedError)

        self.now_epoch = 0

        if args.ckpt is not None:
            self.load_ckpt(args.ckpt)
    
    def load_ckpt(self, ckpt):
        # Load CKPT (Model-Based)
        if self.args.model == "LCI" or self.args.model == "DDPM":
            checkpoint = torch.load(ckpt, map_location=torch.device("cpu"))
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optim'])
            self.now_epoch = checkpoint['epoch']+1
        elif self.args.model == "P2P":
            checkpoint = torch.load(ckpt, map_location=torch.device("cpu"))
            self.model.load_state_dict(checkpoint['model'])
            self.optimizerG.load_state_dict(checkpoint['optimG'])
            self.optimizerD.load_state_dict(checkpoint['optimD'])
            self.now_epoch = checkpoint['epoch']+1
        else:
            raise(NotImplementedError)
        print(f'Load CheckPoint {ckpt}')
        self.now_epoch -= 1
        self.test_epoch()
        self.now_epoch += 1

    def save_ckpt(self):
        ckpt_save_path = self.LOG_DIR + f'ckpt/{self.now_epoch}.pt'
        
        # Save CKPT (Model-Based)
        
        if self.args.model == "LCI" or self.args.model == "DDPM":
            torch.save(
                {
                    'epoch' : self.now_epoch,
                    'model' : self.model.state_dict(),
                    'optim' : self.optimizer.state_dict()
                }, ckpt_save_path
            )
        
        elif self.args.model == "P2P":
            torch.save(
                {
                    'epoch' : self.now_epoch,
                    'model' : self.model.state_dict(),
                    'optimG' : self.optimizerG.state_dict(),
                    'optimD' : self.optimizerD.state_dict(),
                }, ckpt_save_path
            )

        else:
            raise(NotImplementedError)
        print(f'Save CheckPoint to {ckpt_save_path}')

    def write_to_csv(self, filename, items):
        with open(filename, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(items)
        pass

    def save_pic(self, L_chan, ab_chan, ab_out, name):
        # print(f' L_chan {L_chan.min()} {L_chan.max()}')
        # DeNorm
        if L_chan.shape[0] > 16:
            L_chan = L_chan[:16, :, :, :]
            ab_out = ab_out[:16, :, :, :]
            ab_chan = ab_chan[:16, :, :, :]
        L_chan = (L_chan.detach().cpu().numpy() * 100.0).clip(0, 100)
        ab_out = (ab_out.detach().cpu().numpy() * 256.0 - 128.0).clip(-128, 127)
        ab_chan = (ab_chan.detach().cpu().numpy() * 256.0 - 128.0).clip(-128, 127)
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

        img_all = (np.concatenate(double_pic_list, axis=0)*255).clip(0, 255).astype(np.uint8)
        # print(f'img_all {img_all.shape} {img_all.min()},{img_all.max()}')
        
        io.imsave(name, img_all)
        # exit()

    def train_epoch(self):
        if self.args.model == "LCI":
            epoch_loss = 0.0
            self.model.train()
            with tqdm(total=len(self.trainloader), ncols=80, desc=f'Train:Epoch-{self.now_epoch}') as tbar:
                for batch_id, data in enumerate(self.trainloader):
                    L_chan, ab_chan = data
                    # print(L_chan.min(), L_chan.max())
                    L_chan = L_chan.to(self.DEVICES)
                    ab_chan = ab_chan.to(self.DEVICES)
                    self.optimizer.zero_grad()
                    ab_out = self.model(L_chan)
                    loss = self.mse_loss(ab_chan, ab_out)
                    loss.backward()
                    self.optimizer.step()
                    batch_loss = loss.item()
                    self.write_to_csv(self.LOG_DIR+'train_batch.csv', [self.now_epoch, batch_id, batch_loss])
                    epoch_loss += batch_loss * data[0].shape[0]
                    tbar.update(1)
                    if batch_id % self.args.train_report_batch_freq == 0:
                        self.save_pic(L_chan, ab_chan, ab_out, self.LOG_DIR + f'images-train/{self.now_epoch}_{batch_id}.jpg')

            epoch_loss /= len(self.trainset)
            self.write_to_csv(self.LOG_DIR+'train_epoch_loss.csv', [epoch_loss, ])
        elif self.args.model == "P2P":
            epoch_loss_G = 0.0
            self.model.train()
            with tqdm(total=len(self.trainloader), ncols=80, desc=f'Train:Epoch-{self.now_epoch}') as tbar:
                for batch_id, data in enumerate(self.trainloader):
                    L_chan, ab_chan = data
                    L_chan = L_chan.to(self.DEVICES)
                    ab_chan = ab_chan.to(self.DEVICES)

                    # Train discrimnator
                    ab_fake = self.model.netG(L_chan)
                    d_real = self.model.netD(L_chan, ab_chan)
                    d_real_loss = self.bce_loss(d_real, torch.ones_like(d_real))
                    d_fake = self.model.netD(L_chan, ab_fake)
                    d_fake_loss = self.bce_loss(d_fake, torch.zeros_like(d_fake))
                    d_loss = (d_real_loss + d_fake_loss) / 2
                    self.optimizerD.zero_grad()
                    d_loss.backward()
                    self.optimizerD.step()

                    # Train Generator
                    ab_fake = self.model.netG(L_chan)
                    g_fake = self.model.netD(L_chan, ab_fake)
                    g_fake_loss = self.bce_loss(g_fake, torch.ones_like(g_fake))
                    l_loss = self.l1_loss(ab_fake, ab_chan) * self.args.p2p_lambda
                    g_loss = l_loss + g_fake_loss
                    self.optimizerG.zero_grad()
                    g_loss.backward()
                    self.optimizerG.step()

                    batch_loss = l_loss.item()/self.args.p2p_lambda
                    batch_d_loss = d_loss.item()
                    epoch_loss_G += batch_loss * data[0].shape[0]
                    self.write_to_csv(self.LOG_DIR+'train_batch.csv', [self.now_epoch, batch_id, batch_loss, batch_d_loss, g_fake_loss.item()])

                    tbar.update(1)
                    if batch_id % self.args.train_report_batch_freq == 0:
                        self.save_pic(L_chan, ab_chan, ab_fake, self.LOG_DIR + f'images-train/{self.now_epoch}_{batch_id}.jpg')

            epoch_loss_G /= len(self.trainset)
            self.write_to_csv(self.LOG_DIR+'train_epoch_loss.csv', [epoch_loss_G, ])
        elif self.args.model == "DDPM":
            epoch_loss = 0.0
            self.model.train()
            with tqdm(total=len(self.trainloader), ncols=80, desc=f'Train:Epoch-{self.now_epoch}') as tbar:
                for batch_id, data in enumerate(self.trainloader):
                    L_chan, ab_chan = data
                    # print(L_chan.min(), L_chan.max())
                    L_chan = L_chan.to(self.DEVICES)
                    ab_chan = ab_chan.to(self.DEVICES)
                    self.optimizer.zero_grad()
                    loss = self.model(ab_chan, L_chan)
                    loss.backward()
                    self.optimizer.step()

                    batch_loss = loss.item()
                    self.write_to_csv(self.LOG_DIR+'train_batch.csv', [self.now_epoch, batch_id, batch_loss])
                    epoch_loss += batch_loss * data[0].shape[0]
                    tbar.update(1)
                    if batch_id % self.args.train_report_batch_freq == 0:
                        ab_out, _ = self.model.restoration(L_chan, sample_num=8)
                        self.save_pic(L_chan, ab_chan, ab_out, self.LOG_DIR + f'images-train/{self.now_epoch}_{batch_id}.jpg')

            epoch_loss /= len(self.trainset)
            self.write_to_csv(self.LOG_DIR+'train_epoch_loss.csv', [epoch_loss, ])
        else :
            print('Not Implement')
            raise(NotImplementedError)

    def test_epoch(self):
        if self.args.model == "LCI":
            epoch_loss = 0.0
            self.model.eval()
            with torch.no_grad():
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
                        if batch_id % self.args.valid_report_batch_freq == 0:
                            self.save_pic(L_chan, ab_chan, ab_out, self.LOG_DIR + f'images-valid/{self.now_epoch}_{batch_id}.jpg')

                epoch_loss /= len(self.testset)
                self.write_to_csv(self.LOG_DIR+'test_epoch_loss.csv', [self.now_epoch, epoch_loss, ])
        elif self.args.model == "P2P":
            epoch_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                with tqdm(total=len(self.testloader), ncols=80, desc=f'Valid:Epoch-{self.now_epoch}') as tbar:
                    for batch_id, data in enumerate(self.testloader):
                        L_chan, ab_chan = data
                        L_chan = L_chan.to(self.DEVICES)
                        ab_chan = ab_chan.to(self.DEVICES)
                        ab_out = self.model.netG(L_chan)
                        loss = self.mse_loss(ab_chan, ab_out)
                        batch_loss = loss.item()
                        epoch_loss += batch_loss * data[0].shape[0]
                        tbar.update(1)
                        if batch_id % self.args.valid_report_batch_freq == 0:
                            self.save_pic(L_chan, ab_chan, ab_out, self.LOG_DIR + f'images-valid/{self.now_epoch}_{batch_id}.jpg')

                epoch_loss /= len(self.testset)
                self.write_to_csv(self.LOG_DIR+'test_epoch_loss.csv', [self.now_epoch, epoch_loss, ])
        elif self.args.model == "DDPM":
            epoch_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                with tqdm(total=len(self.testloader), ncols=80, desc=f'Valid:Epoch-{self.now_epoch}') as tbar:
                    for batch_id, data in enumerate(self.testloader):
                        L_chan, ab_chan = data
                        L_chan = L_chan.to(self.DEVICES)
                        ab_chan = ab_chan.to(self.DEVICES)
                        ab_out, _ = self.model.restoration(L_chan, y_t = L_chan, sample_num=8)
                        loss = self.mse_loss(ab_chan, ab_out)
                        batch_loss = loss.item()
                        epoch_loss += batch_loss * data[0].shape[0]
                        tbar.update(1)
                        if batch_id % self.args.valid_report_batch_freq == 0:
                            self.save_pic(L_chan, ab_chan, ab_out, self.LOG_DIR + f'images-valid/{self.now_epoch}_{batch_id}.jpg')

                epoch_loss /= len(self.testset)
                self.write_to_csv(self.LOG_DIR+'test_epoch_loss.csv', [self.now_epoch, epoch_loss, ])
        else :
            print('Not Implement')
            raise(NotImplementedError)

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
        if args.model == "LCI" or args.model == "P2P" or args.model == "DDPM":
            runner = Training(args)
            runner.run()
        else:
            print('Not Implement')
            assert(False)
    pass