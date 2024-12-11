import os
import json
import glob

# The following code for generating label files is adapted from the SSDG repository and is provided for reference only
# change your data path
DATA_DIR = './FASdata/'
SAVE_DIR = './label/'

def msu_process():
    test_list = []
    # data_label for msu
    for line in open('./data/MSU/test_sub_list.txt', 'r'):
        test_list.append(line[0:2])
    train_list = []
    for line in open('./data/MSU/train_sub_list.txt', 'r'):
        train_list.append(line[0:2])
    print(test_list)
    print(train_list)
    train_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = os.path.join(SAVE_DIR, 'MSU')
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(os.path.join(label_save_dir, 'train_label.json'), 'w')
    f_test = open(os.path.join(label_save_dir, 'test_label.json'), 'w')
    f_all = open(os.path.join(label_save_dir, 'all_label.json'), 'w')
    f_real = open(os.path.join(label_save_dir, 'real_label.json'), 'w')
    f_fake = open(os.path.join(label_save_dir, 'fake_label.json'), 'w')
    dataset_path = os.path.join(DATA_DIR, 'MSU')
    path_list = glob.glob(os.path.join(dataset_path, '**/*.png'), recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].find('/real/')
        if(flag != -1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        video_num = path_list[i].split('/')[-2].split('_')[0]
        if (video_num in train_list):
            train_final_json.append(dict)
        else:
            test_final_json.append(dict)
        all_final_json.append(dict)
        if(label == 1):
            real_final_json.append(dict)
        else:
            fake_final_json.append(dict)
    print('\nMSU: ', len(path_list))
    print('MSU(train): ', len(train_final_json))
    print('MSU(test): ', len(test_final_json))
    print('MSU(all): ', len(all_final_json))
    print('MSU(real): ', len(real_final_json))
    print('MSU(fake): ', len(fake_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()


def casia_process():
    train_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = os.path.join(SAVE_DIR, 'CASIA')
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(os.path.join(label_save_dir, 'train_label.json'), 'w')
    f_test = open(os.path.join(label_save_dir, 'test_label.json'), 'w')
    f_all = open(os.path.join(label_save_dir, 'all_label.json'), 'w')
    f_real = open(os.path.join(label_save_dir, 'real_label.json'), 'w')
    f_fake = open(os.path.join(label_save_dir, 'fake_label.json'), 'w')
    dataset_path = os.path.join(DATA_DIR, 'CASIA')
    path_list = glob.glob(os.path.join(dataset_path, '**/*.png'), recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].split('/')[-2]
        if (flag == '1' or flag == '2' or flag == 'HR_1'):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        flag = path_list[i].find('/train_release/')
        if (flag != -1):
            train_final_json.append(dict)
        else:
            test_final_json.append(dict)
        all_final_json.append(dict)
        if (label == 1):
            real_final_json.append(dict)
        else:
            fake_final_json.append(dict)
    print('\nCasia: ', len(path_list))
    print('Casia(train): ', len(train_final_json))
    print('Casia(test): ', len(test_final_json))
    print('Casia(all): ', len(all_final_json))
    print('Casia(real): ', len(real_final_json))
    print('Casia(fake): ', len(fake_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()

def replay_process():
    train_final_json = []
    valid_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = os.path.join(SAVE_DIR, 'REPLAY')
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(os.path.join(label_save_dir, 'train_label.json'), 'w')
    f_valid = open(os.path.join(label_save_dir, 'valid_label.json'), 'w')
    f_test = open(os.path.join(label_save_dir, 'test_label.json'), 'w')
    f_all = open(os.path.join(label_save_dir, 'all_label.json'), 'w')
    f_real = open(os.path.join(label_save_dir, 'real_label.json'), 'w')
    f_fake = open(os.path.join(label_save_dir, 'fake_label.json'), 'w')
    dataset_path = os.path.join(DATA_DIR, 'REPLAY')
    path_list = glob.glob(os.path.join(dataset_path, '**/*.png'), recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].find('/real/')
        if (flag != -1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        if (path_list[i].find('/replayattack-train/') != -1):
            train_final_json.append(dict)
        elif(path_list[i].find('/replayattack-devel/') != -1):
            valid_final_json.append(dict)
        else:
            test_final_json.append(dict)
        if(path_list[i].find('/replayattack-devel/') != -1):
            continue
        else:
            all_final_json.append(dict)
            if (label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
    print('\nReplay: ', len(path_list))
    print('Replay(train): ', len(train_final_json))
    print('Replay(valid): ', len(valid_final_json))
    print('Replay(test): ', len(test_final_json))
    print('Replay(all): ', len(all_final_json))
    print('Replay(real): ', len(real_final_json))
    print('Replay(fake): ', len(fake_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()

def oulu_process():
    train_final_json = []
    valid_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = os.path.join(SAVE_DIR, 'OULU')
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(os.path.join(label_save_dir, 'train_label.json'), 'w')
    f_valid = open(os.path.join(label_save_dir, 'valid_label.json'), 'w')
    f_test = open(os.path.join(label_save_dir, 'test_label.json'), 'w')
    f_all = open(os.path.join(label_save_dir, 'all_label.json'), 'w')
    f_real = open(os.path.join(label_save_dir, 'real_label.json'), 'w')
    f_fake = open(os.path.join(label_save_dir, 'fake_label.json'), 'w')
    dataset_path = os.path.join(DATA_DIR, 'OULU')
    path_list = glob.glob(os.path.join(dataset_path, '**/*.png'), recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = int(path_list[i].split('/')[-2].split('_')[-1])
        if (flag == 1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        if (path_list[i].find('/Train_files/') != -1):
            train_final_json.append(dict)
        elif(path_list[i].find('/Dev_files/') != -1):
            valid_final_json.append(dict)
        else:
            test_final_json.append(dict)
        if(path_list[i].find('/Dev_files/') != -1):
            continue
        else:
            all_final_json.append(dict)
            if (label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
    print('\nOulu: ', len(path_list))
    print('Oulu(train): ', len(train_final_json))
    print('Oulu(valid): ', len(valid_final_json))
    print('Oulu(test): ', len(test_final_json))
    print('Oulu(all): ', len(all_final_json))
    print('Oulu(real): ', len(real_final_json))
    print('Oulu(fake): ', len(fake_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()


import json
import math

def sample_frames(dataset_name):
    '''
        from every video (frames) to sample num_frames to test
        return: the choosen frames' path and label
    '''

    # The process is a litter cumbersome, you can change to your way for convenience
    root_path = SAVE_DIR + dataset_name

    for label in ['real','fake','all']:
        if label=='all':
            num_frames = 2
        else:
            num_frames = 1

        label_path = root_path + '/{}_label.json'.format(label)
        save_label_path = root_path + '/choose_{}_label.json'.format(label)

        all_label_json = json.load(open(label_path, 'r'))
        f_sample = open(save_label_path, 'w')

        length = len(all_label_json)
        # three componets: frame_prefix, frame_num, png
        saved_frame_prefix = '/'.join(all_label_json[0]['photo_path'].split('/')[:-1])
        final_json = []
        video_number = 0
        single_video_frame_list = []
        single_video_frame_num = 0
        single_video_label = 0
        c=0
        for i in range(length):
            photo_path = all_label_json[i]['photo_path']
            photo_label = all_label_json[i]['photo_label']
            frame_prefix = '/'.join(photo_path.split('/')[:-1])
            # the last frame
            if (i == length - 1):
                photo_frame = int(photo_path.split('/')[-1].split('.')[0])
                single_video_frame_list.append(photo_frame)
                single_video_frame_num += 1
                single_video_label = photo_label
            # a new video, so process the saved one
            if (frame_prefix != saved_frame_prefix or i == length - 1):
                # if single_video_label==1:
                #     c += 1
                # [1, 2, 3, 4,.....]
                single_video_frame_list.sort()
                frame_interval = math.floor(single_video_frame_num / num_frames)
                for j in range(num_frames):
                    dict = {}
                    dict['photo_path'] = saved_frame_prefix + '/' + str(
                        single_video_frame_list[(6 + j * frame_interval) % single_video_frame_num]) + '.png'
                    dict['photo_label'] = single_video_label
                    dict['photo_belong_to_video_ID'] = video_number
                    final_json.append(dict)
                video_number += 1
                saved_frame_prefix = frame_prefix
                single_video_frame_list.clear()
                single_video_frame_num = 0
            # get every frame information
            photo_frame = int(photo_path.split('/')[-1].split('.')[0])
            single_video_frame_list.append(photo_frame)
            single_video_frame_num += 1
            single_video_label = photo_label
        
            print("\r{}/{}".format(i+1, length), end='')

        print("Total video number(target): ", video_number, dataset_name)
        json.dump(final_json, f_sample, indent=4)
        f_sample.close()

if __name__=="__main__":
    msu_process()
    casia_process()
    replay_process()
    oulu_process()

    sample_frames("CASIA")
    sample_frames("MSU")
    sample_frames("REPLAY")
    sample_frames("OULU")