import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import sys
sys.path.append('../')
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.modified_diffusion import StableDiffusionPipeline
from diffusers import UNet2DConditionModel
from model.arcface import ResNet18_arcface_Feature_Map

IMG_PATH_ROOT = './FASdata'
SAVE_PATH_ROOT = './rec_FASdata'

DFG_CKPT = '../model/sd-models-for-AG-FAS/checkpoint-150000'
SD_PATH = '../model/stable-diffusion-v1-5'

LABEL_PATH = './label/'

class MyDataset(Dataset):
    def __init__(self, read_json):
        
        self.img_path_list = []
        self.save_path_list = []

        for i, dict in enumerate(read_json):
            ori_img_path = dict['photo_path']
            img_path = os.path.join(IMG_PATH_ROOT, ori_img_path)
            save_path = os.path.join(SAVE_PATH_ROOT, ori_img_path)
            if os.path.isfile(save_path):
                continue
            if not os.path.isfile(img_path):
                continue
            
            self.img_path_list.append(img_path)
            self.save_path_list.append(save_path)

        self.transforms = A.Compose([
            A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.img_path_list)

    def open(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, item): 
        img_path = self.img_path_list[item]
        save_path = self.save_path_list[item]

        raw_img = self.open(img_path)
        img = self.transforms(image=raw_img)['image']

        return img, save_path


def generate_rec_img(save_label_path):
    f_sample = open(save_label_path, 'r')
    read_json = json.load(f_sample)
    f_sample.close()

    print('start')
    # DFG model
    unet = UNet2DConditionModel.from_pretrained(DFG_CKPT, subfolder="unet_ema")
    pipe = StableDiffusionPipeline.from_pretrained(SD_PATH, unet=unet)
    pipe = pipe.to("cuda")

    # face identity feature extractor
    text_encoder = ResNet18_arcface_Feature_Map().to("cuda")
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    generator = torch.Generator().manual_seed(0)
   
    # dataloader
    print("len of json: {}".format(len(read_json)))
    input_dataset = MyDataset(read_json)
    print("len of input: {}".format(len(input_dataset)))
    input_dataloader = DataLoader(input_dataset, batch_size=48, shuffle=False)

    steps = 801

    for iter, (input_img, save_path_list) in enumerate(input_dataloader):

        # generate rec img
        with torch.no_grad():
            output_img, rec_latent = pipe.id_condition_steps(input_img.cuda(), text_encoder, steps=steps, generator=generator)
            output_img = output_img.images

        # save rec img
        for i in range(len(save_path_list)):
            img = output_img[i]
            save_path = save_path_list[i]
            os.makedirs(save_path[:save_path.rfind('/')], exist_ok=True)
            img.save(save_path)

        print('{}/{}'.format(iter+1,len(input_dataloader)))
    print('\nend')


if __name__ == '__main__':
    for path in ['CASIA', 'MSU', 'REPLAY', 'OULU']:
        for file in os.listdir(LABEL_PATH + path):
            if file == 'choose_all_label.json':
                generate_rec_img(os.path.join(LABEL_PATH + path, file))