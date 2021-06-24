import os
import random
import glob
import itertools
import numpy as np
from PIL import Image, ImageOps

import torch.utils.data as data


class DataGeneratorPaired(data.Dataset):
    def __init__(self, dataset, root, photo_dir, sketch_dir, photo_sd, sketch_sd, fls_sk, fls_im, clss,
                 transforms_sketch=None, transforms_image=None):
        self.dataset = dataset
        self.root = root
        self.photo_dir = photo_dir
        self.sketch_dir = sketch_dir
        self.photo_sd = photo_sd
        self.sketch_sd = sketch_sd
        self.fls_sk = fls_sk
        self.fls_im = fls_im
        self.clss = clss
        self.transforms_sketch = transforms_sketch
        self.transforms_image = transforms_image

    def __getitem__(self, item):
        sk = ImageOps.invert(Image.open(os.path.join(self.root, self.sketch_dir, self.sketch_sd, self.fls_sk[item]))).\
            convert(mode='RGB')
        im = Image.open(os.path.join(self.root, self.photo_dir, self.photo_sd, self.fls_im[item])).convert(mode='RGB')
        cls = self.clss[item]
        if self.transforms_image is not None:
            im = self.transforms_image(im)
        if self.transforms_sketch is not None:
            sk = self.transforms_sketch(sk)
        return sk, im, cls

    def __len__(self):
        return len(self.clss)

    def get_weights(self):
        weights = np.zeros(self.clss.shape[0])
        uniq_clss = np.unique(self.clss)
        for cls in uniq_clss:
            idx = np.where(self.clss == cls)[0]
            weights[idx] = 1 / idx.shape[0]
        return weights


class DataGeneratorSketch(data.Dataset):
    def __init__(self, dataset, root, sketch_dir, sketch_sd, fls_sk, clss_sk, transforms=None):
        self.dataset = dataset
        self.root = root
        self.sketch_dir = sketch_dir
        self.sketch_sd = sketch_sd
        self.fls_sk = fls_sk
        self.clss_sk = clss_sk
        self.transforms = transforms

    def __getitem__(self, item):
        sk = ImageOps.invert(Image.open(os.path.join(self.root, self.sketch_dir, self.sketch_sd, self.fls_sk[item]))).\
            convert(mode='RGB')
        cls_sk = self.clss_sk[item]
        if self.transforms is not None:
            sk = self.transforms(sk)
        return sk, cls_sk

    def __len__(self):
        return len(self.fls_sk)

    def get_weights(self):
        weights = np.zeros(self.clss_sk.shape[0])
        uniq_clss = np.unique(self.clss_sk)
        for cls in uniq_clss:
            idx = np.where(self.clss_sk == cls)[0]
            weights[idx] = 1 / idx.shape[0]
        return weights


class DataGeneratorImage(data.Dataset):
    def __init__(self, dataset, root, photo_dir, photo_sd, fls_im, clss_im, transforms=None):
        self.dataset = dataset
        self.root = root
        self.photo_dir = photo_dir
        self.photo_sd = photo_sd
        self.fls_im = fls_im
        self.clss_im = clss_im
        self.transforms = transforms

    def __getitem__(self, item):
        im = Image.open(os.path.join(self.root, self.photo_dir, self.photo_sd, self.fls_im[item])).convert(mode='RGB')
        cls_im = self.clss_im[item]
        if self.transforms is not None:
            im = self.transforms(im)
        return im, cls_im

    def __len__(self):
        return len(self.fls_im)

    def get_weights(self):
        weights = np.zeros(self.clss_im.shape[0])
        uniq_clss = np.unique(self.clss_im)
        for cls in uniq_clss:
            idx = np.where(self.clss_im == cls)[0]
            weights[idx] = 1 / idx.shape[0]
        return weights


def get_coarse_grained_samples(classes, fls_im, fls_sk, set_type='train', filter_sketch=True, seed=0):
    idx_im_ret = np.array([], dtype=np.int)
    idx_sk_ret = np.array([], dtype=np.int)
    clss_im = np.array([f.split('/')[-2] for f in fls_im])
    clss_sk = np.array([f.split('/')[-2] for f in fls_sk])
    names_sk = np.array([f.split('-')[0] for f in fls_sk])
    for i, c in enumerate(classes):
        idx1 = np.where(clss_im == c)[0]
        idx2 = np.where(clss_sk == c)[0]
        if set_type == 'train':
            idx_cp = list(itertools.product(idx1, idx2))
            if len(idx_cp) > 100000:
                random.seed(i+seed)
                idx_cp = random.sample(idx_cp, 100000)
            idx1, idx2 = zip(*idx_cp)
        else:
            # remove duplicate sketches
            if filter_sketch:
                names_sk_tmp = names_sk[idx2]
                idx_tmp = np.unique(names_sk_tmp, return_index=True)[1]
                idx2 = idx2[idx_tmp]
        idx_im_ret = np.concatenate((idx_im_ret, idx1), axis=0)
        idx_sk_ret = np.concatenate((idx_sk_ret, idx2), axis=0)
    return idx_im_ret, idx_sk_ret


def load_files_sketchy_zeroshot(root_path, split_eccv_2018=False, filter_sketch=False, photo_dir='photo',
                                sketch_dir='sketch', photo_sd='tx_000000000000', sketch_sd='tx_000000000000', seed=0):
    # paths of sketch and image
    path_im = os.path.join(root_path, photo_dir, photo_sd)
    path_sk = os.path.join(root_path, sketch_dir, sketch_sd)

    # all the image and sketch files together with classes and core names
    fls_sk = np.array(['/'.join(f.split('/')[-2:]) for f in glob.glob(os.path.join(path_sk, '*/*.png'))])
    fls_im = np.array(['/'.join(f.split('/')[-2:]) for f in glob.glob(os.path.join(path_im, '*/*.jpg'))])

    print(f'Total {len(fls_sk)} sketches and {len(fls_im)} imagaes...', end='')

    # classes for image and sketch
    clss_sk = np.array([f.split('/')[0] for f in fls_sk])
    clss_im = np.array([f.split('/')[0] for f in fls_im])

    # all the unique classes
    classes = sorted(os.listdir(path_sk))

    # divide the classes
    if split_eccv_2018:
        # According to Yelamarthi et al., "A Zero-Shot Framework for Sketch Based Image Retrieval", ECCV 2018.
        cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(cur_path, "test_classes_eccv_2018.txt")) as fp:
            te_classes = fp.read().splitlines()
            va_classes = te_classes
            tr_classes = np.setdiff1d(classes, np.union1d(te_classes, va_classes))
    else:
        # According to Shen et al., "Zero-Shot Sketch-Image Hashing", CVPR 2018.
        np.random.seed(seed)
        tr_classes = np.random.choice(classes, int(0.8 * len(classes)), replace=False)
        va_classes = np.random.choice(np.setdiff1d(classes, tr_classes), int(0.1 * len(classes)), replace=False)
        te_classes = np.setdiff1d(classes, np.union1d(tr_classes, va_classes))

    print(f'test classes are: {sorted(te_classes.tolist() + va_classes.tolist())}')

    idx_tr_im, idx_tr_sk = get_coarse_grained_samples(tr_classes, fls_im, fls_sk, set_type='train',
                                                      filter_sketch=filter_sketch, seed=seed)
    idx_va_im, idx_va_sk = get_coarse_grained_samples(va_classes, fls_im, fls_sk, set_type='valid',
                                                      filter_sketch=filter_sketch, seed=seed)
    idx_te_im, idx_te_sk = get_coarse_grained_samples(te_classes, fls_im, fls_sk, set_type='test',
                                                      filter_sketch=filter_sketch, seed=seed)

    idx_all_tr_im, idx_all_tr_sk = get_coarse_grained_samples(tr_classes, fls_im, fls_sk, set_type='gzsl_test',
                                                      filter_sketch=filter_sketch, seed=seed)

    splits = dict()

    splits['tr_fls_sk'] = fls_sk[idx_tr_sk]
    splits['va_fls_sk'] = fls_sk[idx_va_sk]
    splits['te_fls_sk'] = fls_sk[idx_te_sk]

    splits['tr_clss_sk'] = clss_sk[idx_tr_sk]
    splits['va_clss_sk'] = clss_sk[idx_va_sk]
    splits['te_clss_sk'] = clss_sk[idx_te_sk]

    splits['tr_fls_im'] = fls_im[idx_tr_im]
    splits['va_fls_im'] = fls_im[idx_va_im]
    splits['te_fls_im'] = fls_im[idx_te_im]

    splits['tr_all_fls_im'] = fls_im[idx_all_tr_im]
    splits['tr_all_clss_im'] = clss_im[idx_all_tr_im]
    splits['tr_all_fls_sk'] = fls_sk[idx_all_tr_sk]
    splits['tr_all_clss_sk'] = clss_sk[idx_all_tr_sk]

    splits['tr_clss_im'] = clss_im[idx_tr_im]
    splits['va_clss_im'] = clss_im[idx_va_im]
    splits['te_clss_im'] = clss_im[idx_te_im]
    return splits


def load_files_tuberlin_zeroshot(root_path, photo_dir='images', sketch_dir='sketches', photo_sd='', sketch_sd='', seed=0):
    path_im = os.path.join(root_path, photo_dir, photo_sd)
    path_sk = os.path.join(root_path, sketch_dir, sketch_sd)

    # image files and classes
    fls_im = glob.glob(os.path.join(path_im, '*', '*.jpg'))
    fls_im = np.array([os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_im])
    clss_im = np.array([f.split('/')[-2] for f in fls_im])

    # sketch files and classes
    fls_sk = glob.glob(os.path.join(path_sk, '*', '*.png'))
    fls_sk = np.array([os.path.join(f.split('/')[-2], f.split('/')[-1]) for f in fls_sk])
    clss_sk = np.array([f.split('/')[-2] for f in fls_sk])

    print(f'Total {len(fls_sk)} sketches and {len(fls_im)} imagaes...', end='')

    # all the unique classes
    classes = np.unique(clss_im)

    # divide the classes, done according to the "Zero-Shot Sketch-Image Hashing" paper
    np.random.seed(seed)
    tr_classes = np.random.choice(classes, int(0.88 * len(classes)), replace=False)
    va_classes = np.random.choice(np.setdiff1d(classes, tr_classes), int(0.06 * len(classes)), replace=False)
    te_classes = np.setdiff1d(classes, np.union1d(tr_classes, va_classes))

    print(f'test classes are: {sorted(te_classes.tolist() + va_classes.tolist())}')

    idx_tr_im, idx_tr_sk = get_coarse_grained_samples(tr_classes, fls_im, fls_sk, set_type='train', seed=seed)
    idx_va_im, idx_va_sk = get_coarse_grained_samples(va_classes, fls_im, fls_sk, set_type='valid', seed=seed)
    idx_te_im, idx_te_sk = get_coarse_grained_samples(te_classes, fls_im, fls_sk, set_type='test', seed=seed)
    idx_all_tr_im, idx_all_tr_sk = get_coarse_grained_samples(tr_classes, fls_im, fls_sk, set_type='gzsl_test', seed=seed)

    splits = dict()

    splits['tr_fls_sk'] = fls_sk[idx_tr_sk]
    splits['va_fls_sk'] = fls_sk[idx_va_sk]
    splits['te_fls_sk'] = fls_sk[idx_te_sk]

    splits['tr_clss_sk'] = clss_sk[idx_tr_sk]
    splits['va_clss_sk'] = clss_sk[idx_va_sk]
    splits['te_clss_sk'] = clss_sk[idx_te_sk]

    splits['tr_fls_im'] = fls_im[idx_tr_im]
    splits['va_fls_im'] = fls_im[idx_va_im]
    splits['te_fls_im'] = fls_im[idx_te_im]

    splits['tr_all_fls_im'] = fls_im[idx_all_tr_im]
    splits['tr_all_clss_im'] = clss_im[idx_all_tr_im]
    splits['tr_all_fls_sk'] = fls_sk[idx_all_tr_sk]
    splits['tr_all_clss_sk'] = clss_sk[idx_all_tr_sk]

    splits['tr_clss_im'] = clss_im[idx_tr_im]
    splits['va_clss_im'] = clss_im[idx_va_im]
    splits['te_clss_im'] = clss_im[idx_te_im]
    return splits
