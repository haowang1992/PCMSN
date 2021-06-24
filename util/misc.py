import argparse
import os
import shutil
import multiprocessing
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

import torch


def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]" % (x,))
    return x


def create_dict_texts(texts):
    texts = sorted(list(set(texts)))
    d = {l: i for i, l in enumerate(texts)}
    return d


def numeric_classes(tags_classes, dict_tags):
    num_classes = np.array([dict_tags.get(t) for t in tags_classes])
    return num_classes


def prec(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    if k is not None:
        pr = len(act_set & pred_set) / min(k, len(pred_set))
    else:
        pr = len(act_set & pred_set) / max(len(pred_set), 1)
    return pr


def rec(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    re = len(act_set & pred_set) / max(len(act_set), 1)
    return re


def precak(sim, str_sim, k=None):
    act_lists = [np.nonzero(s)[0] for s in str_sim]
    pred_lists = np.argsort(-sim, axis=1)
    num_cores = min(multiprocessing.cpu_count(), 32)
    nq = len(act_lists)
    preck = Parallel(n_jobs=num_cores)(delayed(prec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    reck = Parallel(n_jobs=num_cores)(delayed(rec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    return np.mean(preck), np.mean(reck)


def aps(sim, str_sim):
    nq = str_sim.shape[0]
    num_cores = min(multiprocessing.cpu_count(), 32)
    aps = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim[iq]) for iq in range(nq))
    return aps


def apsak(sim, str_sim, k=None):
    idx = (-sim).argsort()[:, :k]
    sim_k = np.array([sim[i, id] for i, id in enumerate(idx)])
    str_sim_k = np.array([str_sim[i, id] for i, id in enumerate(idx)])
    idx_nz = np.where(str_sim_k.sum(axis=1) != 0)[0]
    sim_k = sim_k[idx_nz]
    str_sim_k = str_sim_k[idx_nz]
    aps_ = np.zeros((sim.shape[0]), dtype=np.float)
    aps_[idx_nz] = aps(sim_k, str_sim_k)
    return aps_


def save_checkpoint(state, directory, prefix=None):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    if prefix:
        checkpoint_file = os.path.join(directory, prefix+'checkpoint.pth')
        best_model_file = os.path.join(directory, prefix+'model_best.pth')
    else:
        checkpoint_file = os.path.join(directory, 'checkpoint.pth')
        best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    shutil.copyfile(checkpoint_file, best_model_file)


def save_qualitative_results(root, sketch_dir, sketch_sd, photo_dir, photo_sd, fls_sk, fls_im, dir_op, aps, sim,
                             str_sim, nq=50, nim=10, im_sz=(256, 256), best=False, save_image=False):
    # Set directories according to dataset
    dir_sk = os.path.join(root, sketch_dir, sketch_sd)
    dir_im = os.path.join(root, photo_dir, photo_sd)

    if not os.path.isdir(dir_op):
        os.makedirs(dir_op)
    else:
        clean_folder(dir_op)

    if best:
        ind_sk = np.argsort(-aps)[:nq]
    else:
        np.random.seed(0)
        ind_sk = np.random.choice(len(aps), nq, replace=False)

    # create a text file for results
    fp = open(os.path.join(dir_op, "Results.txt"), "w")

    for i, isk in enumerate(ind_sk):
        fp.write("{0}, ".format(fls_sk[isk]))
        if save_image:
            sdir_op = os.path.join(dir_op, str(i + 1))
            if not os.path.isdir(sdir_op):
                os.makedirs(sdir_op)
            sk = Image.open(os.path.join(dir_sk, fls_sk[isk])).convert(mode='RGB').resize(im_sz)
            sk.save(os.path.join(sdir_op, fls_sk[isk].split('/')[0] + '.png'))
        ind_im = np.argsort(-sim[isk])[:nim]
        for j, iim in enumerate(ind_im):
            if j < len(ind_im)-1:
                fp.write("{0} {1}, ".format(fls_im[iim], str_sim[isk][iim]))
            else:
                fp.write("{0} {1}".format(fls_im[iim], str_sim[isk][iim]))
            if save_image:
                im = Image.open(os.path.join(dir_im, fls_im[iim])).convert(mode='RGB').resize(im_sz)
                im.save(os.path.join(sdir_op, str(j + 1) + '_' + str(str_sim[isk][iim]) + '.png'))
        fp.write("\n")
    fp.close()


def clean_folder(folder):
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        try:
            if os.path.isfile(p):
                os.unlink(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)
        except Exception as e:
            print(e)

