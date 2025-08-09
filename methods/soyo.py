import time
import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy_domain, count_parameters
from models.soyo_vit import soyo_vit
from models.soyo_clip import soyo_clip
from utils.soyo_utils import feature_compression, random_sampling, soyo_network


class SOYO(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args['net_type'] == 'soyo_vit':
            self._network = soyo_vit(args)
        elif args['net_type'] == 'soyo_clip':
            self._network = soyo_clip(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args['net_type']))
        self.soyo_network = soyo_network(args).to(self._device)

        self.args = args
        self.EPSILON = args['EPSILON']
        self.init_epoch = args['init_epoch']
        self.init_lr = args['init_lr']
        self.init_lr_decay = args['init_lr_decay']
        self.init_weight_decay = args['init_weight_decay']
        self.epochs = args['epochs']
        self.lr = args['lr']
        self.lr_decay = args['lr_decay']
        self.batch_size = args['batch_size']
        self.weight_decay = args['weight_decay']
        self.num_workers = args['num_workers']
        self.dataset = args['dataset']
        
        self.topk = 2
        self.class_num = self._network.class_num
        self.n_components = 2
        self.soyo_epoch = args['soyo_epoch']
        self.soyo_lr = args['soyo_lr']
        self.soyo_weight_decay = args['soyo_weight_decay']
        self.domain_compression = []
    
    def incremental_train(self, data_manager):
        ### Loading dataset
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes) # numtask += 1
        logging.info('')
        logging.info(f'==> Training task {self._cur_task}, Learning on {self._known_classes}-{self._total_classes}')

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                 source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, 
                                       batch_size=self.batch_size, 
                                       shuffle=True,
                                       drop_last=False,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), 
                                                source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, 
                                      batch_size=self.batch_size, 
                                      shuffle=False,
                                      num_workers=self.num_workers)
        logging.info(f'    len(train_dataset): {len(train_dataset)}')
        logging.info(f'    len(test_dataset): {len(test_dataset)}')

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        self.compress_train_resample(self.train_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        
        logging.info('==> Checking the parameter')
        # numtask = 1 ... n
        if len(self._multiple_gpus) > 1:
            _network_numtask = self._network.module.numtask
        else:
            _network_numtask = self._network.numtask
        
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if 'prompt_pool' + '.' + str(_network_numtask - 1) in name:
                param.requires_grad_(True)
            if 'classifier' + '.' + str(_network_numtask - 1) in name:
                param.requires_grad_(True)
            if 'down_pool' + '.' + str(_network_numtask - 1) in name:
                param.requires_grad_(True)
            if 'up_pool' + '.' + str(_network_numtask - 1) in name:
                param.requires_grad_(True)
        logging.info(f'    Total parameters: {count_parameters(self._network)}')
        logging.info(f'    Trainable parameters: {count_parameters(self._network, trainable=True)}')
        logging.info(f'    Blocks:')
        for name, module in self._network.named_children():
            logging.info(f'        {name}: {count_parameters(module)}')
        logging.info(f'    Training:')
        for name, param in self._network.named_parameters():
            if param.requires_grad: 
                logging.info(f'        {name}: {param.shape}')

        ### lr and optimizer
        # self._cur_task = 0 ... n-1
        trainable_params = [param for param in self._network.parameters() if param.requires_grad]
        
        if self._cur_task == 0:
            optimizer = optim.SGD(trainable_params, momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.init_epoch)
            self.run_epoch = self.init_epoch
            self.train_function(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(trainable_params, momentum=0.9, lr=self.lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs)
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)


    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        for _, epoch in enumerate(tqdm(range(self.run_epoch))):
            print('epoch', epoch)
            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            total_steps = len(train_loader)
            
            for step, (_, inputs, targets) in enumerate(tqdm(train_loader), start=1):
                # [bs, 3, 224, 224], [bs]
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1) 
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes
                
                outputs = self._network(inputs)    # [bs, 3, 224, 224]
                logits = outputs['logits']         # [bs, 345]
                loss = F.cross_entropy(logits, targets)
                
                losses += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # tqdm.write(f"Step {step}/{total_steps}, Current loss: {loss.item():.4f}")
                
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy_domain(self._network, test_loader)
            # test_acc = -1
            logging.info(f"Task {self._cur_task}, "
                         f"Epoch [{epoch+1}/{self.run_epoch}] "
                         f"lr {scheduler.get_last_lr()[0]:.5f} "
                         f"Loss {losses/len(train_loader):.3f}, "
                         f"Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}")
    
    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(tqdm(loader)):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)
                logits = outputs['logits']

            _, predicts = torch.max(logits, dim=1)
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


    def compress_train_resample(self, dataloader):
        logging.info('==> Start compress_train_resample')
        ##############################
        ### resample train set from old domains
        if len(self.domain_compression) > 0:
            rs_features = []
            rs_domain_targets = []
            for domain_index, compression in enumerate(self.domain_compression):
                n_samples = len(dataloader.dataset)
                rs_feature = random_sampling(n_samples=n_samples, compression=compression, n_components=self.n_components)
                rs_domain_target = np.full((n_samples), domain_index)
                rs_features.append(rs_feature)
                rs_domain_targets.append(rs_domain_target)
            rs_features = torch.tensor(np.vstack(rs_features), dtype=torch.float32)
            rs_domain_targets = torch.tensor(np.hstack(rs_domain_targets))
        
        num_new = dataloader.batch_size // (1 + len(self.domain_compression)) # 128 / (k+1)
        num_rs = dataloader.batch_size - num_new  # 128 * k / (k+1)
        logging.info(f'    use {num_rs} old + {num_new} new samples for each batch')
        
        
        ##############################
        ### SOYO optimizer
        logging.info('==> Checking the SOYO parameter')
        logging.info('    SOYO Training:')
        training_list = ['soyo_clf']
        for name, param in self.soyo_network.named_parameters():
            param.requires_grad_(False)
            for training_name in training_list:
                if training_name in name:
                    param.requires_grad_(True)
                    logging.info(f'        {name}: {param.shape}')
        logging.info('    SOYO Total params: {}'.format(count_parameters(self.soyo_network)))
        logging.info('    SOYO Trainable params: {}'.format(count_parameters(self.soyo_network, True)))
        
        optimizer_soyo = optim.SGD(self.soyo_network.parameters(), momentum=0.9, lr=self.soyo_lr, weight_decay=self.soyo_weight_decay)
        scheduler_soyo = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_soyo,T_max=self.soyo_epoch)
        
        
        ##############################
        ### train SOYO
        logging.info(f"==> Start SOYO training")
        for _, epoch in enumerate(range(self.soyo_epoch)):
            # self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(tqdm(dataloader)):
                # [bs, 3, 224, 224], [bs]
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask) # select on the dim 0 with mask
                with torch.no_grad():
                    if isinstance(self._network, nn.DataParallel):
                        feature = self._network.module.extract_vector(inputs).to(torch.float32)
                    else:
                        feature = self._network.extract_vector(inputs).to(torch.float32)
                # feature = feature.detach()
                domain_targets = targets // self.class_num
                
                # mix old and new samples after the second
                if len(self.domain_compression) > 0:
                    select_new = torch.randperm(len(feature))[:num_new]
                    select_rs = torch.randperm(len(rs_features))[:num_rs]
                    final_features = torch.cat((feature[select_new], rs_features[select_rs].to(self._device)))
                    final_domain_targets = torch.cat((domain_targets[select_new], rs_domain_targets[select_rs].to(self._device)))
                else:
                    final_features = feature
                    final_domain_targets = domain_targets
                
                domain_logits = self.soyo_network(final_features) # only support single GPU now
                
                loss = F.cross_entropy(domain_logits, final_domain_targets)
                losses += loss.item()
                optimizer_soyo.zero_grad()
                loss.backward()
                optimizer_soyo.step()
                
                _, domain_preds = torch.max(domain_logits, dim=1)
                correct += domain_preds.eq(final_domain_targets.expand_as(domain_preds)).cpu().sum()
                total += len(final_domain_targets)
            
            scheduler_soyo.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            logging.info(f"SOYO training: "
                        f"Task {self._cur_task+1}, "
                        f"Epoch [{epoch+1}/{self.soyo_epoch}] "
                        f"lr {scheduler_soyo.get_last_lr()[0]:.5f} "
                        f"Loss {losses/len(dataloader):.3f}, "
                        f"Train_acc {train_acc:.2f}")

        ##############################
        ### save compression of image features for soyo training
        logging.info('==> Saving compression...')
        features = []
        for i, (_, inputs, targets) in enumerate(tqdm(dataloader)):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask) # select on the dim 0 with mask
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        compression = feature_compression(features, n_components=self.n_components)
        self.domain_compression.append(compression)
        logging.info(f'    used features: {features.shape}')
        logging.info(f'    saving compression: {[i.shape for i in compression]}')
        logging.info(f'    have saved {len(self.domain_compression)} groups of compression')


    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader) # [N, topk], [N]
        cnn_accy = self._evaluate(y_pred, y_true)

        return cnn_accy


    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        
        for _, (_, inputs, targets) in enumerate(tqdm(loader)):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            
            # predict the domain label
            with torch.no_grad():
                # predict domain labels
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs).to(torch.float32)
                else:
                    feature = self._network.extract_vector(inputs).to(torch.float32)
                domain_logits = self.soyo_network(feature)
                selection = torch.max(domain_logits, dim=1)[1]
                
                # forward
                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs, selection)
                else:
                    outputs = self._network.interface(inputs, selection)
            
            predicts = torch.topk(outputs['logits'], k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())   # [bs, topk]
            y_true.append(targets.cpu().numpy())    # [bs]
        
        return np.concatenate(y_pred), np.concatenate(y_true)   # [N, topk], [N]


    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy_domain(y_pred.T[0], y_true, 
                                  self._known_classes, 
                                  increment=self.class_num, 
                                  class_num=self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']

        return ret


    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
