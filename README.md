# Colorization
CV课程作业，共实现了四个算法。

可以通过运行`./script/`下的脚本进行程序的运行

运行环境配置即可完成环境配置，运行相关的脚本和代码即可

```
conda env create -f environment.yml
```

#### 1 Colorization using Optimization

Levin A ,  Lischinski D ,  Weiss Y . Colorization using optimization[J]. ACM Transactions on Graphics (TOG), 2004.

![co](result/CO.bmp)

↑HaHaHa

#### 2 Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification

Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa ACM Transaction on Graphics (Proc. of SIGGRAPH 2016), 2016

![lcir](structure/LCI.bmp)

> Gray Input, Colorization, Ground Truth (10 epoches)

![lci](result/LCI_1.jpg)

#### 3 Image-to-Image Translation with Conditional Adversarial Networks

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros, CVPR, 2017.

![p2pr](structure/P2P.bmp)

> Gray Input, Colorization, Ground Truth (4 epoches)

![p2p](result/P2P_4_0.jpg)

#### 4 Denoising Diffusion Probabilistic Models

Ho, Jonathan and Jain, Ajay and Abbeel, Pieter, Advances in Neural Information Processing Systems, 2020

![ddpmr](structure/DDPM.bmp)

> Gray Input, Colorization, Ground Truth

![DDPM1](result/DDPM_0_0.jpg)

![DDPM2](result/DDPM_0_1.jpg)