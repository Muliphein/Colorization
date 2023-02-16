# Colorization Using Optimization
# A. Levin D. Lischinski and Y. Weiss Colorization using Optimization.
# SIGGRAPH, ACM Transactions on Graphics, Aug 2004.

import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from tqdm import tqdm
from sklearn.preprocessing import normalize

class CO(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def rgb2yiq(rgb):
        rgb = rgb / 255.0
        r = rgb[:, :, 0]
        b = rgb[:, :, 2]
        y = np.dot(rgb, np.array([0.30, 0.59, 0.11]))
        i = 0.74*(r-y) - 0.27*(b-y)
        q = 0.48*(r-y) + 0.41*(b-y)
        yiq = np.stack([y, i, q], axis=2)
        return yiq

    @staticmethod
    def yiq2rgb(yiq):
        r = np.dot(yiq, np.array([1.0,  0.9468822170900693,  0.6235565819861433]))
        g = np.dot(yiq, np.array([1.0, -0.27478764629897834, -0.6356910791873801]))
        b = np.dot(yiq, np.array([1.0, -1.1085450346420322,  1.7090069284064666]))
        rgb = np.stack([r, g, b], axis=2).clip(0, 1)*255.0
        return rgb

    
    def forward(self, gray_rgb, appendix_rgb):
        appendix_yiq = self.rgb2yiq(appendix_rgb)
        isColored = np.abs(gray_rgb / 255.0 - appendix_rgb / 255.0).sum(axis=2) > 0.01

        gray_yiq = self.rgb2yiq(gray_rgb)
        YIQ = np.stack([gray_yiq[:, :, 0], appendix_yiq[:, :, 1], appendix_yiq[:, :, 2]], axis=2)
        n, m = YIQ.shape[:-1]
        image_size = n*m
        ind_matrix = np.arange(image_size).reshape(n, m)
        pixel_in_window = 3 ** 2

        pixel_upper = image_size * pixel_in_window
        row_ind = np.zeros(pixel_upper, dtype=np.int64)
        col_ind = np.zeros(pixel_upper, dtype=np.int64)
        vals = np.zeros(pixel_upper)
        top = 0


        for i in range(n):
            for j in range(m):
                if not isColored[i, j]:
                    temp_value = np.zeros(pixel_in_window)
                    temp_top = 0
                    for x in range(max(0, i-1), min(n, i+2)):
                        for y in range(max(0, j-1), min(m, j+2)):
                            if (x != i or y != j):
                                row_ind[top] = ind_matrix[i, j]
                                col_ind[top] = ind_matrix[x, y]
                                temp_value[temp_top] = YIQ[x, y, 0]
                                top += 1
                                temp_top += 1
                    temp_value[temp_top] = YIQ[i, j, 0]
                    temp_value = temp_value.copy()
                    center_value = YIQ[i, j, 0]
                    variance = np.mean((temp_value[0:temp_top+1] - np.mean(temp_value[0:temp_top+1]))**2)
                    variance_scaled = 0.6 * variance
                    variance_scaled = 2e-6 if variance_scaled < 2e-6 else variance_scaled
                    
                    vals[top-temp_top:top] = np.exp(-((temp_value[0:temp_top] - center_value)**2) / variance_scaled)
                    vals[top-temp_top:top] = - vals[top-temp_top:top] / np.sum(vals[top-temp_top:top])


                row_ind[top] = ind_matrix[i, j]
                col_ind[top] = ind_matrix[i, j]
                vals[top] = 1
                top += 1
        vals = vals[:top]
        row_ind = row_ind[:top]
        col_ind = col_ind[:top]

        A = sparse.csr_matrix((vals, (row_ind, col_ind)), (image_size, image_size))

        b = np.zeros(A.shape[0])

        colored_pos = np.nonzero(isColored.reshape(image_size))
        colorized = np.zeros_like(YIQ)
        colorized[:, :, 0] = YIQ[:, :, 0]
        
        b[colored_pos] = YIQ[:, :, 1].reshape(image_size)[colored_pos]
        colorized[:, :, 1] = scipy.sparse.linalg.spsolve(A, b).reshape(n, m)
        b[colored_pos] = YIQ[:, :, 2].reshape(image_size)[colored_pos]
        colorized[:, :, 2] = scipy.sparse.linalg.spsolve(A, b).reshape(n, m)

        out_pic = self.yiq2rgb(colorized).astype(np.uint8)
        return out_pic


