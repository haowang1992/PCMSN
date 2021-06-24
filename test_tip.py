#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist

# pytorch, torch vision
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as T

# user defined
from util import itq
from config import Config
from util import misc
from util.logger import Logger, AverageMeter
from model.tip_model import Baseline
from dataset.data import DataGeneratorSketch, DataGeneratorImage, \
    load_files_sketchy_zeroshot, load_files_tuberlin_zeroshot


def main():
    # Parse options
    cfg = Config().get_config()

    cfg.test = True
    if cfg.filter_sketch:
        assert cfg.dataset == 'Sketchy'
    if cfg.split_eccv_2018:
        assert cfg.dataset == 'Sketchy_extended' or cfg.dataset == 'Sketchy'


    # modify the log and check point paths
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
    model_name = '+'.join(cfg.semantic_models)

    assert cfg.seed == 0
    path_feature_pretrained = f'model/{cfg.dataset}_'

    path_checkpoint = f"checkpoint/{cfg.dataset}_{ds_var}_{str_aux.replace('+generalized', '')}_{model_name}_{cfg.dim_out}"
    path_result = f'result/{cfg.dataset}_{ds_var}_{str_aux}_{model_name}_{cfg.dim_out}'
    best_model_file = os.path.join(path_checkpoint, 'model_best.pth')
    if os.path.isfile(best_model_file):
        checkpoint = torch.load(best_model_file)
        cfg.seed = checkpoint['exp_seed']
    print(f'Experiment configurations are: {str(cfg)}')

    files_semantic_labels = []
    sem_dim = 0
    for f in cfg.semantic_models:
        fi = os.path.join('dataset', cfg.dataset, f + '.npy')
        files_semantic_labels.append(fi)
        sem_dim += list(np.load(fi, allow_pickle=True).item().values())[0].shape[0]

    print('Checkpoint path: {}'.format(path_checkpoint))
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

    if cfg.gzs_sbir > 0:
        perc = 0.2
        _, idx_sk = np.unique(splits['tr_fls_sk'], return_index=True)
        tr_fls_sk_ = splits['tr_fls_sk'][idx_sk]
        tr_clss_sk_ = splits['tr_clss_sk'][idx_sk]
        _, idx_im = np.unique(splits['tr_fls_im'], return_index=True)
        tr_fls_im_ = splits['tr_fls_im'][idx_im]
        tr_clss_im_ = splits['tr_clss_im'][idx_im]
        if cfg.dataset == 'Sketchy' and cfg.filter_sketch:
            _, idx_sk = np.unique([f.split('-')[0] for f in tr_fls_sk_], return_index=True)
            tr_fls_sk_ = tr_fls_sk_[idx_sk]
            tr_clss_sk_ = tr_clss_sk_[idx_sk]
        idx_sk = np.sort(np.random.choice(tr_fls_sk_.shape[0], int(perc * splits['te_fls_sk'].shape[0]), replace=False))
        idx_im = np.sort(np.random.choice(tr_fls_im_.shape[0], int(perc * splits['te_fls_im'].shape[0]), replace=False))
        splits['te_fls_sk'] = np.concatenate((tr_fls_sk_[idx_sk], splits['te_fls_sk']), axis=0)
        splits['te_clss_sk'] = np.concatenate((tr_clss_sk_[idx_sk], splits['te_clss_sk']), axis=0)
        splits['te_fls_im'] = np.concatenate((tr_fls_im_[idx_im], splits['te_fls_im']), axis=0)
        splits['te_clss_im'] = np.concatenate((tr_clss_im_[idx_im], splits['te_clss_im']), axis=0)

    # class dictionary
    dict_clss = misc.create_dict_texts(splits['tr_clss_im'])

    data_test_sketch = DataGeneratorSketch(cfg.dataset, f'{cfg.dataset_root}/{cfg.dataset}', sketch_dir, sketch_sd,
                                           splits['te_fls_sk'], splits['te_clss_sk'], transforms=transform_sketch)
    data_test_image = DataGeneratorImage(cfg.dataset, f'{cfg.dataset_root}/{cfg.dataset}', photo_dir, photo_sd,
                                         splits['te_fls_im'], splits['te_clss_im'], transforms=transform_image)
    print('Done')

    # PyTorch test loader for sketch
    test_loader_sketch = DataLoader(dataset=data_test_sketch, batch_size=cfg.batch_size, shuffle=False,
                                    num_workers=cfg.num_workers, pin_memory=True)
    # PyTorch test loader for image
    test_loader_image = DataLoader(dataset=data_test_image, batch_size=cfg.batch_size, shuffle=False,
                                   num_workers=cfg.num_workers, pin_memory=True)

    # Model parameters
    params_model = dict()
    # Dimensions
    params_model['dim_out'] = cfg.dim_out
    params_model['sem_dim'] = sem_dim
    params_model['feature_size'] = cfg.feature_size
    # Number of classes
    params_model['num_clss'] = len(dict_clss)
    # Weight (on losses) parameters
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
    # Class dictionary
    params_model['dict_clss'] = dict_clss
    params_model['path_feature_pretrained'] = path_feature_pretrained
    params_model['device'] = torch.device(f'cuda:{cfg.gpu_id}')
    params_model['c2f'] = cfg.c2f

    # Model
    net = Baseline(params_model)

    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = False

    # Check cuda
    print('Checking cuda...', end='')
    # Check if CUDA is enabled
    if cfg.ngpu > 0 & torch.cuda.is_available():
        print('*Cuda exists*...', end='')
        net = net.to(torch.device(f'cuda:{cfg.gpu_id}'))
    print('Done')

    # load the best model yet
    best_model_file = os.path.join(path_checkpoint, 'model_best.pth')
    if os.path.isfile(best_model_file):
        print("Loading best model from '{}'".format(best_model_file))
        checkpoint = torch.load(best_model_file)
        epoch = checkpoint['epoch']
        best_map = checkpoint['best_map']
        exp_seed = checkpoint['exp_seed']
        net.load_state_dict(checkpoint['state_dict'])
        print("Loaded best model '{0}' (epoch {1}; mAP@all {2:.4f} with seed {3})".format(best_model_file, epoch, best_map, exp_seed))
        print('***Test***')
        valid_data = validate(test_loader_sketch, test_loader_image, net, epoch, cfg)
        print('Results on test set: mAP@all = {1:.4f}, Prec@100 = {0:.4f}, mAP@200 = {3:.4f}, Prec@200 = {2:.4f}, '
              'Time = {4:.6f} || mAP@all (binary) = {6:.4f}, Prec@100 (binary) = {5:.4f}, mAP@200 (binary) = {8:.4f}, '
              'Prec@200 (binary) = {7:.4f}, Time (binary) = {9:.6f} '
              .format(valid_data['prec@100'], np.mean(valid_data['aps@all']), valid_data['prec@200'],
                      np.mean(valid_data['aps@200']), valid_data['time_euc'], valid_data['prec@100_bin'],
                      np.mean(valid_data['aps@all_bin']), valid_data['prec@200_bin'], np.mean(valid_data['aps@200_bin'])
                      , valid_data['time_bin']))
        print('Saving qualitative results...', end='')
        path_qualitative_results = os.path.join(path_result, 'qualitative_results')
        misc.save_qualitative_results(f'{cfg.dataset_root}/{cfg.dataset}', sketch_dir, sketch_sd, photo_dir, photo_sd,
                                       splits['te_fls_sk'], splits['te_fls_im'], path_qualitative_results, valid_data['aps@all'],
                                       valid_data['sim_euc'], valid_data['str_sim'], save_image=cfg.save_image_results,
                                       nq=cfg.number_qualit_results, best=cfg.save_best_results)
        print('Done')
    else:
        print("No best model found at '{}'. Exiting...".format(best_model_file))
        exit()


def validate(valid_loader_sketch, valid_loader_image, net, epoch, best, cfg):
    # Switch to test mode
    net.eval()
    batch_time = AverageMeter()

    # Start counting time
    time_start = time.time()

    for i, (sk, cls_sk) in enumerate(valid_loader_sketch):
        if torch.cuda.is_available():
            sk = sk.to(torch.device(f'cuda:{cfg.gpu_id}'))

        # Sketch embedding into a semantic space
        sk_em = net.get_sketch_embeddings(sk)

        # Accumulate sketch embedding
        if i == 0:
            acc_sk_em = sk_em.cpu().data.numpy()
            acc_cls_sk = cls_sk
        else:
            acc_sk_em = np.concatenate((acc_sk_em, sk_em.cpu().data.numpy()), axis=0)
            acc_cls_sk = np.concatenate((acc_cls_sk, cls_sk), axis=0)

        # time
        time_end = time.time()
        batch_time.update(time_end - time_start)
        time_start = time_end

        if (i + 1) % cfg.log_interval == 0:
            print('[Test][Sketch] Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  .format(epoch + 1, i + 1, len(valid_loader_sketch), batch_time=batch_time))
    
    for i, (im, cls_im) in enumerate(valid_loader_image):
        if torch.cuda.is_available():
            im = im.to(torch.device(f'cuda:{cfg.gpu_id}'))

        # Image embedding into a semantic space
        im_em = net.get_image_embeddings(im)

        # Accumulate sketch embedding
        if i == 0:
            acc_im_em = im_em.cpu().data.numpy()
            acc_cls_im = cls_im
        else:
            acc_im_em = np.concatenate((acc_im_em, im_em.cpu().data.numpy()), axis=0)
            acc_cls_im = np.concatenate((acc_cls_im, cls_im), axis=0)

        # time
        time_end = time.time()
        batch_time.update(time_end - time_start)
        time_start = time_end

        if (i + 1) % cfg.log_interval == 0:
            print('[Test][Image] Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  .format(epoch + 1, i + 1, len(valid_loader_image), batch_time=batch_time))
    # Compute mAP
    print('Computing evaluation metrics...', end='')

    # Compute similarity
    t = time.time()
    sim_euc = np.exp(-cdist(acc_sk_em, acc_im_em, metric='euclidean'))
    time_euc = (time.time() - t) / acc_cls_sk.shape[0]

    # similarity of classes or ground truths
    # Multiplied by 1 for boolean to integer conversion
    str_sim = (np.expand_dims(acc_cls_sk, axis=1) == np.expand_dims(acc_cls_im, axis=0)) * 1

    apsall = misc.apsak(sim_euc, str_sim)
    prec100, _ = misc.precak(sim_euc,str_sim,k=100)
    valid_data = {'aps@all': apsall, 'prec@100':prec100}
    print('Done')

    return valid_data


if __name__ == '__main__':
    main()
