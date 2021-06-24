import argparse
from util import misc


class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Zero-Shot Sketch-based Image Retrieval')

        # dataset specific
        parser.add_argument('--dataset', required=True, choices=['Sketchy', 'Sketchy_extended', 'TU-Berlin'], help='Name of the dataset')
        parser.add_argument('--dataset_root', type=str, required=True)
        parser.add_argument('--image_size', default=224, type=int, help='Image size')
        parser.add_argument('--sketch_size', default=224, type=int, help='Sketch size')
        parser.add_argument('--dim_out', default=64, type=int, help='Output dimension of sketch and image')
        parser.add_argument('--feature_size', type=int, default=512, help='Sketch or Image feature size')

        # split specific
        parser.add_argument('--split_eccv_2018', action="store_true", default=False,
                            help='Whether to use the splits of ECCV 2018 paper')
        parser.add_argument('--gzs_sbir', action="store_true", default=False,
                            help='Generalized zero-shot sketch based image retrieval')
        parser.add_argument('--filter_sketch', action="store_true", default=False,
                            help='Allows only one sketch per image (only for Sketchy)')

        # semantic specific
        parser.add_argument('--semantic_models', nargs='+', default=['word2vec-google-news', 'hieremb-path'],
                            type=str, help='Semantic model')
        parser.add_argument('--c2f', action='store_true', default=False)

        # hyper-parameters specific
        parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
        parser.add_argument('--epoch_size', default=100, type=int, help='Epoch size')
        parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU, 1 = CUDA, 1 < DataParallel')
        parser.add_argument('--gpu_id', type=int, default=0)
        parser.add_argument('--num_workers', type=int, default=4, help='Number of workers in data loader')

        parser.add_argument('--epochs', type=int, default=100, metavar='N',
                            help='Number of epochs to train (default: 100)')
        parser.add_argument('--lr', type=lambda x: misc.restricted_float(x, [1e-5, 0.5]), default=0.0001, metavar='LR',
                            help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--milestones', type=int, nargs='+', default=[], help='Milestones for scheduler')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule steps.')

        # Weight (on loss) parameters
        parser.add_argument('--lambda_rec', default=0.5, type=float, help='Weight on rec')
        parser.add_argument('--lambda_gen_adv', default=1.0, type=float, help='Weight on adversarial loss (gen)')
        parser.add_argument('--lambda_ret_cls', default=0.4, type=float, help='Weight on classification loss (retrieval)')
        parser.add_argument('--lambda_disc_se', default=1.0, type=float, help='Weight on semantic loss (disc)')
        parser.add_argument('--lambda_mm_euc', default=1.0, type=float, help='Euclidean distance for multimodality')
        parser.add_argument('--lambda_domain_cls', default=1.0, type=float, help='Weight on domain classification loss')
        parser.add_argument('--drop', default=0.5, type=float)

        # log specific
        parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                            help='How many batches to wait before logging training status')
        parser.add_argument('--save_image-results', action="store_true", default=False, help='Whether to save image '
                                                                                             'results')
        parser.add_argument('--number_qualit_results', type=int, default=2000, help='Number of qualitative results to be'
                                                                                   ' saved')
        parser.add_argument('--save_best_results', action="store_true", default=False, help='Whether to save the best '
                                                                                            'results')

        parser.add_argument('--test', action="store_true", default=False, help='Test only flag')
        parser.add_argument('--early_stop', type=int, default=5, help='Early stopping epochs.')
        parser.add_argument('--seed', type=int, default=-1)

        self.parser = parser

    def get_config(self):
        return self.parser.parse_args()

