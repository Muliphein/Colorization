# Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification
# Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa
# ACM Transaction on Graphics (Proc. of SIGGRAPH 2016), 2016

import torch
import torch.nn as nn
import torch.nn.functional as F

class LCI(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lowlevelfeat_channel_size = [1, 64, 128, 128, 256, 256, 512]
        self.lowlevelfeat_kernel_size = [3, 3, 3, 3, 3, 3]
        self.lowlevelfeat_stride_size = [2, 1, 2, 1, 2, 1]
        self.lowlevelfeat_conv= [
            nn.Conv2d(
                in_channels = self.lowlevelfeat_channel_size[i],
                out_channels = self.lowlevelfeat_channel_size[i+1],
                stride = self.lowlevelfeat_stride_size[i],
                kernel_size = self.lowlevelfeat_kernel_size[i],
                padding=1
            ) for i in range(6)
        ]
        # print(self.lowlevelfeat_conv)

        self.midlevelfeat_conv= [
            nn.Conv2d(
                in_channels = 512, out_channels = 512, stride = 1, kernel_size = 3, padding=1
            ),
            nn.Conv2d(
                in_channels = 512, out_channels = 256, stride = 1, kernel_size = 3, padding=1
            ),
        ]
        
        self.globalfeat_channels_size = [512, 512, 512, 512, 512]
        self.globalfeat_kernel_size = [3, 3, 3, 3]
        self.globalfeat_stride_size = [2, 1, 2, 1]
        self.globalfeat_conv = [
            nn.Conv2d(
                in_channels = self.globalfeat_channels_size[i],
                out_channels = self.globalfeat_channels_size[i+1],
                stride = self.globalfeat_stride_size[i],
                kernel_size = self.globalfeat_kernel_size[i],
                padding=1
            ) for i in range(4)
        ]

        self.global_linear_size = [7*7*512, 1024, 512, 256]
        self.global_linear = [
            nn.Linear(self.global_linear_size[i], self.global_linear_size[i+1])
            for i in range(3)
        ]

        self.fusion_layer = nn.Conv2d(512, 256, 1, 1, 0)

        class InterPolate(nn.Module):
            def __init__(self) -> None:
                super().__init__()
            
            def forward(self, x):
                return F.interpolate(input=x, scale_factor=2)



        self.colorization_seq = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.ReLU(),
            InterPolate(),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(),
            InterPolate(),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, 1, padding=1),
            nn.Sigmoid(),
            InterPolate(),
        )

    
    def forward(self, x):

        out = x

        # Low Level Features
        for i in range(len(self.lowlevelfeat_conv)):
            # print(self.lowlevelfeat_conv[i])
            # print(f'Out Shape : {out.shape}')
            out = F.relu(self.lowlevelfeat_conv[i](out))
            # print(f'Out Shape : {out.shape}')
        
        global_out = out
        mid_out = out

        # Mid Level Features
        for i in range(len(self.midlevelfeat_conv)):
            mid_out = F.relu(self.midlevelfeat_conv[i](mid_out))
        
        # Global Level Features
        for i in range(len(self.globalfeat_conv)):
            global_out = F.relu(self.globalfeat_conv[i](global_out))
        global_out = global_out.view(-1, 7 * 7 * 512)
        for i in range(len(self.global_linear)):
            global_out = F.relu(self.global_linear[i](global_out)) # [bs, 256]

        # Fusion
        h, w = mid_out.shape[2:4]
        global_out_stack = torch.stack([global_out]*w, 2)
        global_out_stack = torch.stack([global_out_stack]*h, 2)

        volume = torch.cat([mid_out, global_out_stack], 1)
        volume = F.relu(self.fusion_layer(volume))

        # Colorization
        volume = self.colorization_seq(volume)
        
        return volume

if __name__ == "__main__":
    model = LCI()
    input = torch.randn(16, 1, 224, 224)
    print(f'Model LCI Input Shape : {input.shape} ', end='')
    output = model(input)
    print(f'Output Shape : {output.shape}')

    pass