import os
import random
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from config import Config
from dataset.data import load_files_sketchy_zeroshot, load_files_tuberlin_zeroshot, \
    DataGeneratorImage, DataGeneratorPaired, DataGeneratorSketch
from model.tip_model import Baseline
from util import misc
from util.logger import Logger, AverageMeter
from test_tip import validate


def main():
    cfg = Config().get_config()
    if cfg.seed == -1:
        cfg.seed = random.randint(1, 10000)
    random.seed(cfg.seed)
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu_id)
        torch.cuda.manual_seed(cfg.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)
    print(f'Experiment configurations are: {str(cfg)}')

    # check configuration
    if cfg.filter_sketch:
        assert cfg.dataset == 'Sketchy'
    if cfg.split_eccv_2018:
        assert cfg.dataset == 'Sketchy_extended' or cfg.dataset == 'Sketchy'
    if cfg.gzs_sbir:
        cfg.test = True

    ds_var = None
    if '_' in cfg.dataset:
        token = cfg.dataset.split('_')
        cfg.dataset = token[0]
        ds_var = token[1]

    str_aux = 'None'
    if cfg.split_eccv_2018:
        str_aux = 'split_eccv_2018'
    if cfg.gzs_sbir:
        str_aux = '+'.join([str_aux, 'generalized'])

    cfg.semantic_models = sorted(cfg.semantic_models)
    model_name ='+'.join(cfg.semantic_models)

    assert cfg.seed == 0
    path_feature_pretrained = f'model/{cfg.dataset}_'

    model_name += f'_c2f_{cfg.c2f}'

    path_checkpoint = f"checkpoint/tip_{cfg.dataset}_{ds_var}_{str_aux.replace('+generalized', '')}_{model_name}_{cfg.dim_out}"
    path_log = f'log/tip_{cfg.dataset}_{ds_var}_{str_aux}_{model_name}_{cfg.dim_out}'
    path_result = f'result/tip_{cfg.dataset}_{ds_var}_{str_aux}_{model_name}_{cfg.dim_out}'

    files_semantic_labels = []
    files_semantic_dims = []
    sem_dim = 0
    for f in cfg.semantic_models:
        fi = os.path.join('dataset', cfg.dataset, f + '.npy')
        files_semantic_labels.append(fi)
        files_semantic_dims.append(list(np.load(fi, allow_pickle=True).item().values())[0].shape[0])
        sem_dim += files_semantic_dims[-1]
    print(files_semantic_dims)

    print('Checkpoint path: {}'.format(path_checkpoint))
    print('Logger path: {}'.format(path_log))
    print('Result path: {}'.format(path_result))

    # Parameters for transforming the images
    transform_image = T.Compose([T.Resize((cfg.image_size, cfg.image_size)), T.ToTensor()])
    transform_sketch = T.Compose([T.Resize((cfg.sketch_size, cfg.sketch_size)), T.ToTensor()])

    # Load the dataset
    print('Loading data...', end='')

    if cfg.dataset == 'Sketchy':
        if ds_var == 'extended':
            photo_dir = 'extended_photo'  # photo or extended_photo
            photo_sd = ''
        else:
            photo_dir = 'photo'
            photo_sd = 'tx_000000000000'
        sketch_dir = 'sketch'
        sketch_sd = 'tx_000000000000'
        splits = load_files_sketchy_zeroshot(root_path=f'{cfg.dataset_root}/{cfg.dataset}',
                                             split_eccv_2018=cfg.split_eccv_2018,
                                             photo_dir=photo_dir, sketch_dir=sketch_dir, photo_sd=photo_sd,
                                             sketch_sd=sketch_sd, seed=cfg.seed)
    elif cfg.dataset == 'TU-Berlin':
        photo_dir = 'images'
        sketch_dir = 'sketches'
        photo_sd = ''
        sketch_sd = ''
        splits = load_files_tuberlin_zeroshot(root_path=f'{cfg.dataset_root}/{cfg.dataset}',
                                              photo_dir=photo_dir, sketch_dir=sketch_dir,
                                              photo_sd=photo_sd, sketch_sd=sketch_sd, seed=cfg.seed)
    else:
        raise Exception('Wrong dataset.')

    # Combine the valid and test set into test set
    splits['te_fls_sk'] = np.concatenate((splits['va_fls_sk'], splits['te_fls_sk']), axis=0)
    splits['te_clss_sk'] = np.concatenate((splits['va_clss_sk'], splits['te_clss_sk']), axis=0)
    splits['te_fls_im'] = np.concatenate((splits['va_fls_im'], splits['te_fls_im']), axis=0)
    splits['te_clss_im'] = np.concatenate((splits['va_clss_im'], splits['te_clss_im']), axis=0)

    if cfg.gzs_sbir:
        _, idx_im = np.unique(splits['tr_all_fls_im'], return_index=True)
        tr_all_fls_im_ = splits['tr_all_fls_im'][idx_im]
        tr_all_clss_im_ = splits['tr_all_clss_im'][idx_im]

        splits['te_fls_im'] = np.concatenate((tr_all_fls_im_, splits['te_fls_im']), axis=0)
        splits['te_clss_im'] = np.concatenate((tr_all_clss_im_, splits['te_clss_im']), axis=0)

    # class dictionary
    dict_clss = misc.create_dict_texts(splits['tr_clss_im'])

    def worker_init_fn(worker_id):
        np.random.seed(cfg.seed + worker_id)

    data_train = DataGeneratorPaired(cfg.dataset, f'{cfg.dataset_root}/{cfg.dataset}', photo_dir, sketch_dir, photo_sd,
                                         sketch_sd, splits['tr_fls_sk'], splits['tr_fls_im'], splits['tr_clss_im'],
                                         transforms_sketch=transform_sketch, transforms_image=transform_image)
    data_valid_sketch = DataGeneratorSketch(cfg.dataset, f'{cfg.dataset_root}/{cfg.dataset}', sketch_dir, sketch_sd,
                                            splits['va_fls_sk'], splits['va_clss_sk'], transforms=transform_sketch)
    data_valid_image = DataGeneratorImage(cfg.dataset, f'{cfg.dataset_root}/{cfg.dataset}', photo_dir, photo_sd,
                                          splits['va_fls_im'], splits['va_clss_im'], transforms=transform_image)

    data_test_sketch = DataGeneratorSketch(cfg.dataset, f'{cfg.dataset_root}/{cfg.dataset}', sketch_dir, sketch_sd,
                                           splits['te_fls_sk'], splits['te_clss_sk'], transforms=transform_sketch)
    data_test_image = DataGeneratorImage(cfg.dataset, f'{cfg.dataset_root}/{cfg.dataset}', photo_dir, photo_sd,
                                         splits['te_fls_im'], splits['te_clss_im'], transforms=transform_image)
    print('Done')

    train_sampler = WeightedRandomSampler(data_train.get_weights(), num_samples=cfg.epoch_size * cfg.batch_size,
                                          replacement=True)

    # PyTorch train loader
    train_loader = DataLoader(dataset=data_train, batch_size=cfg.batch_size, sampler=train_sampler,
                              num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    # PyTorch valid loader for sketch
    valid_loader_sketch = DataLoader(dataset=data_valid_sketch, batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    # PyTorch valid loader for image
    valid_loader_image = DataLoader(dataset=data_valid_image, batch_size=cfg.batch_size, shuffle=False,
                                        num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    # PyTorch test loader for sketch
    test_loader_sketch = DataLoader(dataset=data_test_sketch, batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    # PyTorch test loader for image
    test_loader_image = DataLoader(dataset=data_test_image, batch_size=cfg.batch_size, shuffle=False,
                                        num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    # Model parameters
    params_model = dict()
    # Dimensions
    params_model['dim_out'] = cfg.dim_out
    params_model['sem_dim'] = sem_dim
    params_model['feature_size'] = cfg.feature_size
    # Number of classes
    params_model['num_clss'] = len(dict_clss)
    # Weight (on losses) parameters
    params_model['lambda_rec'] = cfg.lambda_rec
    params_model['lambda_gen_adv'] = cfg.lambda_gen_adv
    params_model['lambda_ret_cls'] = cfg.lambda_ret_cls
    params_model['lambda_disc_se'] = cfg.lambda_disc_se
    params_model['lambda_mm_euc'] = cfg.lambda_mm_euc
    params_model['drop'] = cfg.drop
    # Optimizers' parameters
    params_model['lr'] = cfg.lr
    params_model['momentum'] = cfg.momentum
    params_model['milestones'] = cfg.milestones
    params_model['gamma'] = cfg.gamma
    # Files with semantic labels
    params_model['files_semantic_labels'] = files_semantic_labels
    params_model['files_semantic_dims'] = files_semantic_dims
    # Class dictionary
    params_model['dict_clss'] = dict_clss
    params_model['device'] = torch.device(f'cuda:{cfg.gpu_id}')
    params_model['path_feature_pretrained'] = path_feature_pretrained
    params_model['c2f'] = cfg.c2f

    # Model
    net = Baseline(params_model)

    # Logger
    print('Setting logger...', end='')
    logger = Logger(path_log, force=True)
    print('Done')

    # Check cuda
    print('Checking cuda...', end='')
    # Check if CUDA is enabled
    if cfg.ngpu > 0 & torch.cuda.is_available():
        print('*Cuda exists*...', end='')
        net = net.to(torch.device(f'cuda:{cfg.gpu_id}'))
    print('Done')

    best_map = 0
    early_stop_counter = 0

    # Epoch for loop
    if not cfg.test:
        print('***Train***')

        print('***First: Train model***')
        for epoch in range(cfg.epochs):
            net.scheduler_gen.step()
            net.scheduler_disc.step()

            # train on training set
            losses = train(train_loader, net, epoch, cfg)

            # evaluate on validation set, map_ since map is already there
            print('***Validation***')
            valid_data = validate(test_loader_sketch, test_loader_image, net, epoch, False, cfg)
            # H mean
            map_ =np.mean(valid_data['aps@all'])
           

            del valid_data

            if map_ > best_map:
                best_map = map_
                early_stop_counter = 0
                misc.save_checkpoint({'exp_seed': cfg.seed, 'epoch': epoch + 1, 'state_dict': net.state_dict(), 'best_map':
                    best_map}, directory=path_checkpoint)
            else:
                if cfg.early_stop == early_stop_counter:
                    break
                early_stop_counter += 1

            # Logger step
            logger.add_scalar('generator classification loss', losses['ret_cls'].avg)
            logger.add_scalar('multimodal euclidean loss', losses['mm_euc'].avg)
            logger.add_scalar('generator loss', losses['gen'].avg)
            logger.add_scalar('discriminator loss', losses['disc'].avg)
            logger.add_scalar('mean average precision', map_)
            logger.step()

    # load the best model yet
    best_model_file = os.path.join(path_checkpoint, 'model_best.pth')
    if os.path.isfile(best_model_file):
        print("Loading best model from '{}'".format(best_model_file))
        checkpoint = torch.load(best_model_file)
        epoch = checkpoint['epoch']
        best_map = checkpoint['best_map']
        exp_seed = checkpoint['exp_seed']

        model_dict_pretrained = checkpoint['state_dict']
        model_dict_org = net.state_dict()
        model_dict_pretrained = {k: v for k, v in model_dict_pretrained.items() if k in model_dict_org}
        model_dict_org.update(model_dict_pretrained)
        # net.load_state_dict(model_dict_pretrained)

        print("Loaded best model '{0}' (epoch {1}; mAP@all {2:.4f}) with seed {3}".format(best_model_file, epoch, best_map, exp_seed))
        print('***Test***')
        valid_data = validate(test_loader_sketch, test_loader_image, net, epoch, False, cfg)

        if not os.path.exists(f'result/tip1.baseline_c2f_{cfg.dataset}.txt'):
            fr = open(f'result/tip1.baseline_c2f_{cfg.dataset}.txt', 'w+')
        else:
            fr = open(f'result/tip1.baseline_c2f_{cfg.dataset}.txt', 'a+')
        print('lambda_ret_cls={2}, lambda_mm_euc={3}, c2f={4}, drop={5} '
              'Results on test set: mAP@all = {1:.4f}, Prec@100 = {0:.4f} \n\n'
              .format(valid_data['prec@100'], np.mean(valid_data['aps@all']),cfg.lambda_ret_cls, cfg.lambda_mm_euc, cfg.c2f, cfg.drop), file=fr)
        fr.close()
        # print('Saving qualitative results...', end='')
        # path_qualitative_results = os.path.join(path_result, 'qualitative_results')
        # misc.save_qualitative_results(f'{cfg.dataset_root}/{cfg.dataset}', sketch_dir, sketch_sd, photo_dir, photo_sd,
        #                               splits['te_fls_sk'], splits['te_fls_im'], path_qualitative_results, valid_data['aps@all'],
        #                                valid_data['sim_euc'], valid_data['str_sim'], save_image=cfg.save_image_results,
        #                                nq=cfg.number_qualit_results, best=cfg.save_best_results)
        print('Done')
    else:
        print("No best model found at '{}'. Exiting...".format(best_model_file))
        exit()


def train(train_loader, net, epoch, cfg):
    # Switch to train mode
    net.train()

    batch_time = AverageMeter()
    losses_ret_cls = AverageMeter()
    losses_mm_euc = AverageMeter()
    losses_gen = AverageMeter()
    losses_disc = AverageMeter()
    losses_rec = AverageMeter()

    # Start counting time
    time_start = time.time()

    for i, (sk, im, cl) in enumerate(train_loader):
        # Transfer sk and im to cuda
        if torch.cuda.is_available():
            sk, im = sk.to(torch.device(f'cuda:{cfg.gpu_id}')), im.to(torch.device(f'cuda:{cfg.gpu_id}'))

        # Optimize parameters
        loss = net.optimize_params(sk, im, cl)

        # Store losses for visualization
        losses_ret_cls.update(loss['ret_cls'].item(), sk.size(0))
        losses_mm_euc.update(loss['mm_euc'].item(), sk.size(0))
        losses_gen.update(loss['gen'].item(), sk.size(0))
        losses_disc.update(loss['disc'].item(), sk.size(0))
        losses_rec.update(loss['disc'].item(), sk.size(0))
        # time
        time_end = time.time()
        batch_time.update(time_end - time_start)
        time_start = time_end

        if (i + 1) % cfg.log_interval == 0:
            print('[Train] Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Ret. Cls Loss {loss_ret_cls.val:.4f} ({loss_ret_cls.avg:.4f})\t'
                  'MM. Euc Loss {loss_mm_euc.val:.4f} ({loss_mm_euc.avg:.4f})\t'
                  'Gen. Loss {loss_gen.val:.4f} ({loss_gen.avg:.4f})\t'
                  'Disc. Loss {loss_disc.val:.4f} ({loss_disc.avg:.4f})\t'
                  'Rec. Loss {loss_rec.val:.4f} ({loss_rec.avg:.4f})\t'
                  .format(epoch + 1, i + 1, len(train_loader), batch_time=batch_time, loss_ret_cls=losses_ret_cls,
                          loss_mm_euc=losses_mm_euc, loss_gen=losses_gen, loss_disc=losses_disc, loss_rec = losses_disc))

    losses = {'gen': losses_gen, 'disc': losses_disc, 'ret_cls': losses_ret_cls, 'mm_euc': losses_mm_euc, 'rec': losses_rec}
    return losses


if __name__ == '__main__':
    main()
