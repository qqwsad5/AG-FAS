from torch.utils.data import Dataset, DataLoader
import os
import torch
import json
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_PATH_ROOT = './data/FASdata'
REC_IMG_PATH_ROOT = './data/rec_FASdata'

def get_img(img_path, img_rec_path, open_f):
    img = open_f(img_path)
    # return img
    img_res = (img - open_f(img_rec_path))
    # norm
    mean = img_res.mean()
    std = img_res.std()
    img_res = (img_res-mean)/(std+1e-8)
    return torch.concat([img, img_res], dim=0)

class DiffusionDataset(Dataset):
    def __init__(self, data_pd_list, transforms=None, train=True, resize=None, cache=False):
        self.train = train
        self.cache = cache
        self.photo_path = []
        self.photo_rec_path = []
        self.photo_label = []
        self.photo_belong_to_video_ID = []
        self.resize = resize
        self.img = []

        if transforms is None:
            if not train:
                self.transforms = A.Compose([
                    A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
                    ToTensorV2(),
                ])
            else:
                self.transforms = A.Compose([
                    A.HorizontalFlip(),
                    A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
                    ToTensorV2(),
                ])
        else:
            self.transforms = transforms

        for data_pd in data_pd_list:
            for data in data_pd:
                img_path = os.path.join(IMG_PATH_ROOT, data['photo_path'])
                img_rec_path = os.path.join(REC_IMG_PATH_ROOT, data['photo_path'])                
                # if not exist
                if not os.path.isfile(img_path):
                    continue
                if not os.path.isfile(img_rec_path):
                    continue

                if self.cache:
                    # cache img
                    img = get_img(img_path, img_rec_path, self.open)
                    self.img.append(img)
                else:
                    self.photo_path.append(img_path)
                    self.photo_rec_path.append(img_rec_path)

                self.photo_label.append(data['photo_label'])
                self.photo_belong_to_video_ID.append(data['photo_belong_to_video_ID'])

    def open(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)['image']
        return img

    def __len__(self):
        return len(self.photo_label)

    def __getitem__(self, item):
        label = self.photo_label[item]

        if self.cache:
            # cache img
            img = self.img[item]
        else:
            img_path = self.photo_path[item]
            img_rec_path = self.photo_rec_path[item]

            img = get_img(img_path, img_rec_path, self.open)

        if not self.train:
            return img, label, self.photo_belong_to_video_ID[item]

        return img, label

def base_domain_split(src_data_list):
    '''
    input:  [src1_data, src2_data, src3_data], [src1_train_num_frames, src2_train_num_frames, src3_train_num_frames]\n
    return: real_data_pd_list, fake_data_pd_list
    '''
    print('Base Domain Split')
    src_train_data_real_list = []
    src_train_data_fake_list = []
    for index, src_data in enumerate(src_data_list):
        print('Dataset {}: '.format(index+1), src_data)
        src_train_data_fake = sample_frames(flag=0, dataset_name=src_data)
        src_train_data_real = sample_frames(flag=1, dataset_name=src_data)
        src_train_data_real_list.append(src_train_data_real)
        src_train_data_fake_list.append(src_train_data_fake)

    return src_train_data_real_list, src_train_data_fake_list

def get_domain_dataset(real_data_pd_list, fake_data_pd_list, domain_batch_size=1, resize=None):
    '''
    return: real_dataloader_list, fake_dataloader_list
    '''
    print('Load Split Domain Source Data')
    assert len(real_data_pd_list) == len(fake_data_pd_list), "domain num not match"
    return get_split_dataset(real_data_pd_list, domain_batch_size, resize=resize), \
            get_split_dataset(fake_data_pd_list, domain_batch_size, resize=resize)


def get_split_dataset(data_pd_list, batch_size, resize=None, train=True):
    dataloader_list = []
    for i in range(len(data_pd_list)):
        dataloader_list.append(DataLoader(DiffusionDataset([data_pd_list[i]], train=train, resize=resize),
                                            batch_size=batch_size, shuffle=True))
    return dataloader_list

def get_train_dataset(src_data_list, batch_size):
    src_train_data_real_list, src_train_data_fake_list = base_domain_split(src_data_list)
    return get_domain_dataset(src_train_data_real_list, src_train_data_fake_list, domain_batch_size=batch_size)

def get_valid_dataset(tgt_data, batch_size):
    print('Load Target Data')
    print('Target Data: ', tgt_data)
    tgt_test_data = sample_frames(flag=2, dataset_name=tgt_data)
    valid_dataloader = DataLoader(DiffusionDataset([tgt_test_data], train=False, cache=True), batch_size=batch_size, shuffle=False)
    return valid_dataloader

def sample_frames(flag, dataset_name):
    '''
        from every video (frames) to sample num_frames to test
        return: the choosen frames' path and label
    '''
    
    # The process is a litter cumbersome, you can change to your way for convenience
    root_path = '/SSD1/longxingming/Diffusion/data/FASdata/label/' + dataset_name
    if(flag == 0): # select the fake images
        save_label_path = root_path + '/choose_fake_label.json'
    elif(flag == 1): # select the real images
        save_label_path = root_path + '/choose_real_label.json'
    else: # select all the real and fake images
        save_label_path = root_path + '/choose_all_label.json'

    f_sample = open(save_label_path, 'r')
    final_json = json.load(f_sample)

    print('{}: {}'.format(dataset_name, len(final_json)))
    f_sample.close()

    return final_json
