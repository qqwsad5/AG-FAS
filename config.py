class DefaultConfigs(object):
    seed = 666
    # SGD
    weight_decay = 5e-4
    momentum = 0.9

    # learning rate
    init_lr = 1e-3

    # SSDG config
    lambda_triplet = 1
    lambda_adreal = 0.5

    # iter
    max_iter = 4000
    valid_per_iter = 10

    # warm up
    warmup = True
    lr_warmup = int(0.1*max_iter)

    # model
    pretrained = True

    valid_batch_size = 512
    domain_batch_size = 10

    model = 'cross_{}_tri{}_ad{}'.format(max_iter, lambda_triplet, lambda_adreal)

    # training parameters
    gpus = "0"
    
    # ckpt
    save_checkpoint = True
    
    # log path
    log_dir = './logs/{}_lr{}_batch{}/tgt(tgt_data)/'.format(model, init_lr, domain_batch_size)
    log_file = 'log.txt'

    cache_path = './cache/'

    # Code Saver
    save_code = []

config = DefaultConfigs()
