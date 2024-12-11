from utils.log import Logger, save_checkpoint, save_config
from utils.evaluate import eval, AverageMeter
from dataset import get_train_dataset, get_valid_dataset
from method.SSDG import SSDG_iter
from model.FAS_model import OA_Net, Discriminator

import random
import numpy as np

from config import config
 
import os
import torch
import torch.optim as optim

# fix seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = 'cuda'


def train():

    train_name_list = ['MSU', 'CASIA', 'REPLAY', 'OULU']
    dataset_name_list = ['MSU', 'CASIA', 'REPLAY', 'OULU']

    log_dir = config.log_dir

    result_collect_list = []

    for tgt_data in train_name_list:
        config.log_dir = log_dir.replace('tgt_data', tgt_data)

        src_dataset_name_list = dataset_name_list.copy()
        src_dataset_name_list.remove(tgt_data)

        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)

        # log
        log = Logger(log_valid_entry=['loss', 'top-1', 'HTER', 'AUC'],\
                    log_train_entry=['cls_l', 'adreal', 'triplet'],\
                    log_best_smaller_is_better=[True, False, True, False],\
                    log_file=config.log_dir + config.log_file,\
                    TB_path =config.log_dir)

        time_start = log.time_start.strftime('%Y-%m-%d_%H-%M-%S')
        save_path = os.path.join(config.log_dir, "CONFIG", time_start)
        save_config(config, save_path)


        # dataloader
        valid_dataloader = get_valid_dataset(tgt_data, config.valid_batch_size)
        real_dataloader_list, fake_dataloader_list = get_train_dataset(src_dataset_name_list, batch_size=config.domain_batch_size)

        # model
        net = OA_Net().to(device)
        ad_net_real = Discriminator(in_channel=net.feature_size, n_domain=len(real_dataloader_list)).to(device)

        # optimizer
        optimizer_dict = [
            {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": config.init_lr}
        ]
        optimizer_dict.append({"params": filter(lambda p: p.requires_grad, ad_net_real.parameters()), "lr": config.init_lr})
        optimizer = optim.SGD(optimizer_dict, lr=config.init_lr, momentum=config.momentum, weight_decay=config.weight_decay)

        init_param_lr = []
        for param_group in optimizer.param_groups:
            init_param_lr.append(param_group["lr"])
        
        # dataloader iter list
        real_iter_list = [iter(i) for i in real_dataloader_list]
        fake_iter_list = [iter(i) for i in fake_dataloader_list]

        max_iter = config.max_iter

        log.train_start()
        cls_loss_avg = AverageMeter()
        adreal_avg = AverageMeter()
        triplet_avg = AverageMeter()

        # iter
        for iter_num in range(max_iter+1):
            # warmup
            if config.warmup:
                if iter_num < config.lr_warmup:
                    lr = config.init_lr * ((iter_num + 1) / config.lr_warmup)
                else:
                    lr = config.init_lr * 0.5 * (1 + np.cos(np.pi * (iter_num - config.lr_warmup) / (config.max_iter - config.lr_warmup)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # data prepare
            input_data = []
            source_label = []
            real_shape_list = []
            fake_shape_list = []
            for index in range(len(real_iter_list)):
                try:
                    real_img, real_label = real_iter_list[index].__next__()
                except:
                    real_iter_list[index] = iter(real_dataloader_list[index])
                    real_img, real_label = real_iter_list[index].__next__()
                real_shape = real_img.shape[0]
                input_data.append(real_img.cuda())
                source_label.append(real_label.cuda())
                real_shape_list.append(real_shape)
            for index in range(len(fake_iter_list)):
                try:
                    fake_img, fake_label = fake_iter_list[index].__next__()
                except:
                    fake_iter_list[index] = iter(fake_dataloader_list[index])
                    fake_img, fake_label = fake_iter_list[index].__next__()
                fake_shape = fake_img.shape[0]
                input_data.append(fake_img.cuda())
                source_label.append(fake_label.cuda())
                fake_shape_list.append(fake_shape)

            input_data = torch.cat(input_data, dim=0)
            source_label = torch.cat(source_label, dim=0)

            # train
            cls_loss, adreal, triplet = SSDG_iter(net, ad_net_real, input_data, source_label, optimizer, real_shape_list, fake_shape_list, config)

            cls_loss_avg.update(cls_loss)
            adreal_avg.update(adreal)
            triplet_avg.update(triplet)

            log.writer.add_scalar("lr", optimizer.state_dict()['param_groups'][0]['lr'], iter_num)
            log.writer.add_scalar("cls_loss", cls_loss, iter_num)
            log.writer.add_scalar("adreal", adreal, iter_num)
            log.writer.add_scalar("triplet", triplet, iter_num)

            log.train_terminal_out(steps=iter_num, max_steps=max_iter)

            if (iter_num != 0 and iter_num % config.valid_per_iter == 0):
                # valid
                # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:AUC, 5:threshold, 6:ACC_threshold
                valid_args = eval(valid_dataloader, net)
                threshold = valid_args[5]

                # log
                log_best = log.valid_log(log_valid=[valid_args[0], valid_args[1], valid_args[3] * 100, valid_args[4] * 100],\
                                        log_train=[cls_loss_avg.get_avg(), adreal_avg.get_avg(), triplet_avg.get_avg()],\
                                        steps=iter_num, max_steps=max_iter)          

                # save ckpt
                save_list = {"valid_args":valid_args, "log_best":log_best, "threshold":threshold}
                time_start = log.time_start.strftime('%Y-%m-%d_%H-%M-%S')
                save_path = os.path.join(config.log_dir, "CKPT", time_start)
                best_val = None
                if log_best[2] == valid_args[3] * 100:
                    best_val = valid_args[3] * 100
                if config.save_checkpoint:
                    save_checkpoint(iter_num, net, optimizer, config.gpus, save_path, info=save_list, best_val=best_val)
                    
        # save HTER and AUC
        result_collect_list += [log_best[2], log_best[3]]
    
    file = open(log_dir.replace('tgt(tgt_data)/', 'collect.txt'), 'w')
    for i in result_collect_list:
        file.write('{:^8.3f}\t'.format(i))
    file.close()

if __name__ == '__main__':
    train()
