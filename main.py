import cv2 as cv
from model import CO
import numpy as np
import torch
import argparse

INPUT_CASE = './data/CUO_Example/'

def deal_args():
    parser = argparse.ArgumentParser(description='Colorization Setting')
    parser.add_argument('--mode', type=str, default='evaluate')
    parser.add_argument('--model', type=str, default='CO')
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()
    return args


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
            out = np.concatenate([gray_pic,appendix_pic,out_pic[:, :, ::-1]], axis = 1)
            cv.imwrite(f'./result/{args.model}.bmp', out)
        else:
            print('Not Implement')
            assert(False)
        print('Result Save to '+f'./result/{args.model}.bmp')
    elif args.mode == 'train':
        print('Not Implement')
        assert(False)
    pass