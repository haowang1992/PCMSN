import os
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.models as M

from util import misc, loss



class VGGNetFeats(nn.Module):
	def __init__(self, pretrained=True, finetune=True):
		super(VGGNetFeats, self).__init__()
		model = M.vgg16(pretrained=pretrained)
		for param in model.parameters():
			param.requires_grad = finetune
		self.features = model.features
		self.classifier = nn.Sequential(
			*list(model.classifier.children())[:-1],
			nn.Linear(4096, 512)
		)

	def forward(self, x):
		x = self.features(x)
		x = self.classifier(x.view(x.size(0), -1))
		return x


class GaussianNoiseLayer(nn.Module):
	def __init__(self, mean=0.0, std=0.2, device=torch.device('cpu')):
		super(GaussianNoiseLayer, self).__init__()
		self.mean = mean
		self.std = std
		self.device = device

	def forward(self, x):
		if self.training:
			noise = x.data.new(x.size()).normal_(self.mean, self.std)
			if x.is_cuda:
				noise = noise.to(self.device)
			x = x + noise
		return x


class Generator(nn.Module):
	def __init__(self, in_dim=512, out_dim=300, noise=True, use_batchnorm=True, use_dropout=False, use_prelu=False, device=torch.device('cpu'), drop=0.5):
		super(Generator, self).__init__()
		hid_dim = (in_dim + out_dim) // 2

		# main model
		modules = list()

		modules.append(nn.Linear(in_dim, hid_dim))
		if use_batchnorm:
			modules.append(nn.BatchNorm1d(hid_dim))
		if use_prelu:
			modules.append(nn.PReLU())
		else:
			modules.append(nn.LeakyReLU(0.2, inplace=True))
		if noise:
			modules.append(GaussianNoiseLayer(mean=0.0, std=0.2, device=device))
		if use_dropout:
			modules.append(nn.Dropout(p=drop))

		modules.append(nn.Linear(hid_dim, hid_dim))
		if use_batchnorm:
			modules.append(nn.BatchNorm1d(hid_dim))
		if use_prelu:
			modules.append(nn.PReLU())
		else:
			modules.append(nn.LeakyReLU(0.2, inplace=True))
		if noise:
			modules.append(GaussianNoiseLayer(mean=0.0, std=0.2, device=device))
		if use_dropout:
			modules.append(nn.Dropout(p=drop))

		modules.append(nn.Linear(hid_dim, out_dim))

		self.gen = nn.Sequential(*modules)

	def forward(self, x):
		return self.gen(x)


class Discriminator(nn.Module):
	def __init__(self, in_dim=300, out_dim=1, noise=True, use_batchnorm=True, use_dropout=False,
				 use_sigmoid=False, use_prelu=False, device=torch.device('cpu'), drop=0.5):
		super(Discriminator, self).__init__()
		hid_dim = (in_dim) // 2

		modules = list()

		if noise:
			modules.append(GaussianNoiseLayer(mean=0.0, std=0.3, device=device))
		modules.append(nn.Linear(in_dim, hid_dim))
		if use_batchnorm:
			modules.append(nn.BatchNorm1d(hid_dim))
		if use_prelu:
			modules.append(nn.PReLU())
		else:
			modules.append(nn.LeakyReLU(0.2, inplace=True))
		if use_dropout:
			modules.append(nn.Dropout(p=drop))

		modules.append(nn.Linear(hid_dim, hid_dim))
		if use_batchnorm:
			modules.append(nn.BatchNorm1d(hid_dim))
		if use_prelu:
			modules.append(nn.PReLU())
		else:
			modules.append(nn.LeakyReLU(0.2, inplace=True))
		if use_dropout:
			modules.append(nn.Dropout(p=drop))
		modules.append(nn.Linear(hid_dim, out_dim))
		if use_sigmoid:
			modules.append(nn.Sigmoid())

		self.disc = nn.Sequential(*modules)

	def forward(self, x):
		return self.disc(x)


class Baseline(nn.Module):
	def __init__(self, params_model):
		super(Baseline, self).__init__()

		print('Initializing model variables...', end='')
		# Dimension of embedding
		self.dim_out = params_model['dim_out']
		# Dimension of semantic embedding
		self.sem_dim = params_model['sem_dim']
		# Number of classes
		self.num_clss = params_model['num_clss']
		self.path_feature_pretrained = params_model['path_feature_pretrained']
		self.drop = params_model['drop']
		# Sketch or Image feature size
		self.feature_size = params_model['feature_size']
		# Sketch model: pre-trained on ImageNet
		self.sketch_model = VGGNetFeats(pretrained=True, finetune=False)
		sketch_model_dict_pretrained = torch.load(self.path_feature_pretrained+'sketch.pth', map_location='cpu')['state_dict_sketch']
		self.sketch_model.load_state_dict(sketch_model_dict_pretrained)

		# Image model: pre-trained on ImageNet
		self.image_model = VGGNetFeats(pretrained=True, finetune=False)
		image_model_dict_pretrained = torch.load(self.path_feature_pretrained + 'image.pth', map_location='cpu')['state_dict_image']
		self.image_model.load_state_dict(image_model_dict_pretrained)

		self.device = params_model['device']

		# Semantic model embedding
		self.sem = []
		for f in params_model['files_semantic_labels']:
			self.sem.append(np.load(f, allow_pickle=True).item())
		self.files_semantic_dims = params_model['files_semantic_dims']
		self.dict_clss = params_model['dict_clss']
		print('Done')

		print('Initializing trainable models...', end='')
		# Generators
		self.sem_enc = Generator(in_dim=self.feature_size, out_dim=self.sem_dim, noise=False, use_dropout=True,
								   use_prelu=True, device=self.device, drop=0.5)
								   
								   
		self.ret = nn.Sequential(
				nn.Linear(in_features=self.sem_dim, out_features=1024),
				nn.BatchNorm1d(num_features=1024),
				nn.PReLU(),
				nn.Linear(in_features=1024, out_features=1024),
				nn.BatchNorm1d(num_features=1024),
				nn.PReLU(),
				nn.Dropout(p=self.drop),
				nn.Linear(in_features=1024, out_features=self.dim_out)
				)
								  
		# Discriminators
		# Common semantic discriminator
		self.disc_se = Discriminator(in_dim = self.sem_dim, noise = True, use_batchnorm = True, use_prelu = True, 
									 device = self.device, drop = 0.5)
		

		# Classifier
		self.classifier_se = nn.Linear(self.dim_out, self.num_clss, bias=False)
		for param in self.classifier_se.parameters():
			param.requires_grad = False
		with torch.no_grad():
			self.classifier_se.weight.div_(torch.norm(self.classifier_se.weight, dim=1, keepdim=True))
	

		# Optimizers
		print('Defining optimizers...', end='')
		self.lr = params_model['lr']
		self.gamma = params_model['gamma']
		self.momentum = params_model['momentum']
		self.milestones = params_model['milestones']

		self.optimizer_gen = optim.Adam(list(self.sem_enc.parameters()) + list(self.ret.parameters()),
										lr=self.lr)
		self.optimizer_disc = optim.SGD(list(self.disc_se.parameters()), lr=self.lr, momentum=self.momentum)
		self.scheduler_gen = optim.lr_scheduler.MultiStepLR(self.optimizer_gen, milestones=self.milestones,
															gamma=self.gamma)
		self.scheduler_disc = optim.lr_scheduler.MultiStepLR(self.optimizer_disc, milestones=self.milestones,
															 gamma=self.gamma)
		print('Done')

		# Loss function
		print('Defining losses...', end='')
		self.lambda_rec = params_model['lambda_rec']
		self.lambda_gen_adv = params_model['lambda_gen_adv']
		self.lambda_ret_cls = params_model['lambda_ret_cls']
		self.lambda_domain_cls = params_model['lambda_domain_cls']
		self.lambda_disc_se = params_model['lambda_disc_se']
		self.criterion_gan = loss.GANLoss(use_lsgan=True, device=self.device)
		self.criterion_cls = nn.CrossEntropyLoss()
		#self.criterion_rec = nn.L1Loss()
		self.criterion_rec = nn.MSELoss(reduce=True, size_average=True)
		print('Done')

		# Intermediate variables
		print('Initializing variables...', end='')
		self.img_rec_1 = torch.zeros(1)
		self.img_rec_2 = torch.zeros(1)
		self.ske_rec_1 = torch.zeros(1)
		self.ske_rec_2 = torch.zeros(1)

		self.sketch_feature = torch.zeros(1)
		self.sketch_semantic = torch.zeros(1)
		self.image_feature = torch.zeros(1)
		self.image_semantic = torch.zeros(1)
		self.img_domain = torch.zeros(1)
		self.ske_domain = torch.zeros(1)
		self.ske_ret = torch.zeros(1)
		self.img_ret = torch.zeros(1)
		print('Done')

	def forward(self, sketch, image, semantic, label_cls):
	
		self.sketch_feature = self.sketch_model(sketch)
		self.image_feature = self.image_model(image)

		self.sketch_semantic = self.sem_enc(self.sketch_feature)
		self.image_semantic = self.sem_enc(self.image_feature)
		self.ske_ret = self.ret(self.sketch_semantic)
		self.img_ret = self.ret(self.image_semantic)

	def backward(self, semantic, label_cls):
		# Adversarial loss with flipped labels (false -> true)
		loss_gen_adv = self.criterion_gan(self.disc_se(self.sketch_semantic), True) + \
			self.criterion_gan(self.disc_se(self.image_semantic), True)
		loss_gen_adv = self.lambda_gen_adv * loss_gen_adv
		# Classification loss
		loss_ret_cls = self.criterion_cls(self.classifier_se(self.ske_ret), label_cls) + \
					   self.criterion_cls(self.classifier_se(self.img_ret), label_cls)

		loss_cls = self.lambda_ret_cls * loss_ret_cls


		loss_gen = loss_gen_adv + loss_cls
		self.optimizer_gen.zero_grad()
		loss_gen.backward(retain_graph=True)
		self.optimizer_gen.step()
		
		# Discriminator loss
		self.optimizer_disc.zero_grad()
		loss_disc_se = self.criterion_gan(self.disc_se(semantic), True) + \
					   0.5 * self.criterion_gan(self.disc_se(self.sketch_semantic), False) + \
					   0.5 * self.criterion_gan(self.disc_se(self.image_semantic), False)
		loss_disc_se = self.lambda_disc_se * loss_disc_se
		loss_disc_se.backward()

		loss_disc = loss_disc_se
		self.optimizer_disc.step()

		loss = {'gen': loss_gen, 'disc': loss_disc, 'gen_adv': loss_gen_adv, 'ret_cls': loss_ret_cls, 'domain_cls': loss_ret_cls, 'rec': loss_ret_cls}
		return loss

	def optimize_params(self, sk, im, cl):
		# Get numeric classes
		num_cls = torch.from_numpy(misc.numeric_classes(cl, self.dict_clss)).to(self.device)

		# Get the semantic embeddings for cl
		se = np.zeros((len(cl), self.sem_dim), dtype=np.float32)
		for i, c in enumerate(cl):
			se_c = np.array([], dtype=np.float32)
			for s in self.sem:
				se_c = np.concatenate((se_c, s.get(c).astype(np.float32)), axis=0)
			se[i] = se_c
		se = torch.from_numpy(se)
		if torch.cuda.is_available:
			se = se.to(self.device)

		# Forward pass
		self.forward(sk, im, se, num_cls)
		# Backward pass
		loss = self.backward(se, num_cls)
		return loss

	def get_sketch_embeddings(self, sk):
		# sketch embedding
		sketch_feature = self.sketch_model(sk)
		sk_em = self.ret(self.sem_enc(sketch_feature))
		return sk_em

	def get_image_embeddings(self, im):
		# image embedding
		image_feature = self.image_model(im)
		im_em = self.ret(self.sem_enc(image_feature))
		return im_em
