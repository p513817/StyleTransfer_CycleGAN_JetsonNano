import torch
from torch import nn
from torchsummary import summary


def conv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0):
    
    layer = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
                          nn.InstanceNorm2d(out_dim), 
                          nn.ReLU(True))
    return layer

def dconv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0, output_padding=0):
    
    layer = nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, output_padding),
                          nn.InstanceNorm2d(out_dim), 
                          nn.ReLU(True))
    return layer

class ResidualBlock(nn.Module):
    
    def __init__(self, dim, use_dropout):
        super(ResidualBlock, self).__init__()
        res_block = [nn.ReflectionPad2d(1),
                     conv_norm_relu(dim, dim, kernel_size=3)]
        
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                      nn.InstanceNorm2d(dim)]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)

class Generator(nn.Module):
    
    def __init__(self, input_nc=3, output_nc=3, filters=64, use_dropout=True, n_blocks=6):
        super(Generator, self).__init__()
        
        # 向下採樣
        model = [nn.ReflectionPad2d(3),
                 conv_norm_relu(input_nc   , filters * 1, 7),
                 conv_norm_relu(filters * 1, filters * 2, 3, 2, 1),
                 conv_norm_relu(filters * 2, filters * 4, 3, 2, 1)]

        # 頸脖層
        for i in range(n_blocks):
            model += [ResidualBlock(filters * 4, use_dropout)]

        # 向上採樣
        model += [dconv_norm_relu(filters * 4, filters * 2, 3, 2, 1, 1),
                  dconv_norm_relu(filters * 2, filters * 1, 3, 2, 1, 1),
                  nn.ReflectionPad2d(3),
                  nn.Conv2d(filters, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)    # model 是 list 但是 sequential 需要將其透過 , 分割出來

    def forward(self, x):
        return self.model(x)

import time
import os
import cv2
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import time

"""
torch                         1.6.0+cu101    
torchsummary                  1.5.1          
torchtext                     0.3.1          
torchvision                   0.7.0+cu101
"""

def init_model():
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    G_B2A = Generator().to(device)
    G_B2A.load_state_dict(torch.load(os.path.join("weights", "netG_B2A.pth"), map_location=device ))
    G_B2A.eval()
    
    return G_B2A

def test(G, img):
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    transform = transforms.Compose([transforms.Resize((256,256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    data = transform(img).to(device)
    
    data = data.unsqueeze(0)
    
    start_time = time.time()

    out = (0.5 * (G(data).data + 1.0)).squeeze(0)

    print(f'Tranfer Time : {time.time()-start_time}s')

    return out
    
if __name__=='__main__':
    
    G = init_model()
    
    cap = cv2.VideoCapture(0)
    
    change_style = False

    save_img_name = 'test.jpg'

    while(True):

        ret, frame = cap.read()
        
        # Do Something Cool 
        ############################
        
        if change_style:
            style_img = test(G, Image.fromarray(frame))
            out = np.array(style_img.cpu()).transpose([1,2,0])
        else:
            out = frame
        out = cv2.resize(out, (512, 512))
            
        ###########################
        
        cv2.imshow('webcam', out)

        key = cv2.waitKey(1)

        if key==ord('q'):
            break

        elif key==ord('s'):
            cv2.imwrite(save_img_name)
        
        elif key==ord('t'):
            change_style = False if change_style else True

    cap.release()
    cv2.destroyAllWindows()