import sys
import os
from tensorboardX import SummaryWriter
import torch
import shutil
from datetime import datetime, timedelta

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%dh %02dm'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%dm %02ds'%(min,sec)
    else:
        raise NotImplementedError

class Eval_Logger(object):
    def __init__(self):
        self.start_time = datetime.now()
        sys.stdout.write("\n")
        sys.stdout.flush()
    def out(self, steps, max_steps):
        progress = int(50*steps/max_steps)
        time_use = (datetime.now() - self.start_time).seconds
        message = "\rVALID |"+"█"*progress+" "*(50-progress)+"| {}/{}, TIME {}".format(steps, max_steps, time_to_str(time_use, 'sec'))
        sys.stdout.write(message)
        sys.stdout.flush()

class Logger(object):
    def __init__(self, log_train_entry, log_valid_entry, log_best_smaller_is_better, log_file=None, TB_path=None):
        assert len(log_valid_entry) == len(log_best_smaller_is_better), "ERROR: log_valid size not match the best"
        self.terminal = sys.stdout
        self.log_train_entry = log_train_entry
        self.log_valid_entry = log_valid_entry
        self.log_best_smaller_is_better = log_best_smaller_is_better
        self.log_file = log_file
        self.TB_path = TB_path
        self.time_train_start = None
        self.time_start = datetime.now()
        # initialize the best results
        self.log_best = []
        for ind in log_best_smaller_is_better:
            if ind:
                self.log_best.append(float('inf'))
            else:
                self.log_best.append(-float('inf'))
        # begin time
        start_message = "\n\nPROGRAM BEGIN {} \n".format(self.time_start.strftime('%Y-%m-%d %H:%M:%S'))
        self.terminal.write(start_message)
        self.terminal.flush()
        # Tensorboard
        self.writer = None
        if self.TB_path != None:
            log_message = "WRITE LOG IN: {} \n".format(TB_path)
            self.terminal.write(log_message)
            self.terminal.flush()
            if not os.path.exists(TB_path):
                os.makedirs(TB_path)
            time_log = self.time_start.strftime('%Y-%m-%d_%H-%M-%S')
            self.writer = SummaryWriter(os.path.join(self.TB_path, time_log))
        # log file
        self.file = None
        if self.log_file != None:
            self.file = open(self.log_file, 'a')
            self.file.write(start_message)
            self.file.flush()

    # train start
    def train_start(self):
        self.time_train_start = datetime.now()
        loading_time = (self.time_train_start - self.time_start).seconds
        message = "\nLOADING TIME {}s \nTRAIN START\n".format(loading_time)
        self.terminal.write(message)
        self.terminal.flush()
        if self.log_file != None:
            self.file.write(message)
            self.file.flush()
            # write the head
            self.file.write(self.get_head())
            self.file.flush()

    # progress info
    def train_terminal_out(self, steps, max_steps):
        # assert steps > 0 and steps <= max_steps, "ERROR: iter value out of range"
        progress = int(50*steps/max_steps)
        time_use = (datetime.now() - self.time_train_start).seconds
        message = "\rTRAINING |"+"█"*progress+" "*(50-progress)+"| {}/{}, TIME {}".format(steps, max_steps, time_to_str(time_use, 'min'))
        self.terminal.write(message)
        self.terminal.flush()

    # record the validation result
    def valid_log(self, log_train, log_valid, steps, max_steps):
        assert len(log_train) == len(self.log_train_entry), "ERROR: log_train size not match the entry"
        assert len(log_valid) == len(self.log_valid_entry), "ERROR: log_valid size not match the entry"
        assert steps > 0 and steps <= max_steps, "ERROR: iter value out of range"
        
        time_use = (datetime.now() - self.time_train_start).seconds
        time_remain = int(time_use*(max_steps-steps)/steps)
        time_end = datetime.now() + timedelta(seconds=time_remain)

        # update the best
        for i in range(len(log_valid)):
            if self.log_best_smaller_is_better[i]:
                if log_valid[i] < self.log_best[i]:
                    self.log_best[i] = log_valid[i]
            else:
                if log_valid[i] > self.log_best[i]:
                    self.log_best[i] = log_valid[i]

        value_message = self.get_value(steps, log_train, log_valid, self.log_best, time_to_str(time_use, 'min'))

        message = "\nVALID LOG\n"
        message += self.get_head() + value_message
        message += "TIME LEFT {}, PROGRAM END AT {}\n\n".format(time_to_str(time_remain, 'min'), time_end.strftime('%Y-%m-%d %H:%M:%S''%Y-%m-%d %H:%M:%S'))
        self.terminal.write(message)
        self.terminal.flush()

        if self.log_file != None:
            self.file.write(value_message)
            self.file.flush()
        
        if self.TB_path != None:
            for i in range(len(log_train)):
                self.writer.add_scalar("train_"+self.log_train_entry[i], log_train[i], steps)
            for i in range(len(log_valid)):
                self.writer.add_scalar("valid_"+self.log_valid_entry[i], log_valid[i], steps)
        
        return self.log_best

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    # hean info
    def get_head(self):
        entry = "|{:^7}|".format("steps")
        head = "|{:-^7}|".format("-")

        train_entry = ""
        for i in range(len(self.log_train_entry)):
            train_entry += "{:^8}".format(self.log_train_entry[i])
        entry += train_entry + "|"
        head += "{:-^{}}|".format("TRAIN", len(train_entry))

        valid_entry = ""
        for i in range(len(self.log_valid_entry)):
            valid_entry += "{:^8}".format(self.log_valid_entry[i])
        entry += valid_entry + "|"
        head += "{:-^{}}|".format("VALID", len(valid_entry))

        entry += valid_entry + "|"
        head += "{:-^{}}|".format("BEST", len(valid_entry))

        entry += "{:^10}|\n".format("time")
        head += "{:-^10}|\n".format("-")

        return head+entry
    
    # output head
    def get_value(self, steps, log_train, log_valid, log_best, time):
        value = "|{:^7}|".format(steps)
        for i in range(len(log_train)):
            value += "{:^8.3f}".format(log_train[i])
        value += "|"
        for i in range(len(log_valid)):
            value += "{:^8.3f}".format(log_valid[i])
        value += "|"
        for i in range(len(log_best)):
            value += "{:^8.3f}".format(log_best[i])
        value += "|{:^10}|\n".format(time)
        return value

def save_config(config, save_path):
    folder_path = os.path.join(save_path, 'config')
    os.makedirs(folder_path, exist_ok=True)
    for filepath in config.save_code:
        shutil.copy(filepath, os.path.join(folder_path, filepath.split('/')[-1]))

def save_checkpoint(steps, model, optimizer, gpus, save_path, filename='_checkpoint.pth.tar', info=None, best_val=None):
    # make dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if(len(gpus) > 1):
        old_state_dict = model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            flag = k.find('module.')
            if (flag != -1):
                k = k.replace('module.', '')
            new_state_dict[k] = v
        state = {
            "steps": steps,
            "state_dict": new_state_dict,
            'optimizer': optimizer.state_dict(),
            "info": info
        }
    else:
        state = {
            "steps": steps,
            "state_dict": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "info": info
        }

    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    # just save best model
    if best_val != None:
        # delete old file
        for file in os.listdir(save_path):
            if 'best' in file:
                os.remove(os.path.join(save_path, file))
        # save the newest
        shutil.copy(filepath, os.path.join(save_path, 'steps_' + str(steps) + '_best_' + str(best_val) + '.pth.tar'))
