import cv2 as cv
from model import CO
import numpy as np
import torch

INPUT_CASE = './data/CUO_Example/'

if __name__ == "__main__":
    print('Evaluate Start')
    gray_pic_name = INPUT_CASE + '2.bmp'
    appendix_pic_name = INPUT_CASE + '3.bmp'

    # Get Picture BGR2RGB
    gray_pic = cv.imread(gray_pic_name)[:, :, ::-1].copy()
    appendix_pic = cv.imread(appendix_pic_name)[:, :, ::-1].copy()

    model = CO()
    out_pic = model(gray_pic, appendix_pic) 

    out = np.concatenate([gray_pic,appendix_pic,out_pic[:, :, ::-1]], axis = 1)

    cv.imwrite('result.bmp', out)

    pass