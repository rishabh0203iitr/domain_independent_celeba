import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from data.util import get_dataset, IdxDataset, ZippedDataset
from module.util import get_model
from models import basenet
from models import dataloader
import utils

class CifarModel():
    def __init__(self, opt):
        super(CifarModel, self).__init__()
        self.epoch = 0
        self.device = opt['device']
        self.save_path = opt['save_folder']
        self.print_freq = opt['print_freq']
        self.init_lr = opt['optimizer_setting']['lr']
        self.log_writer = SummaryWriter(os.path.join(self.save_path, 'logfile'))

        self.dataset_tag = 'CelebA'
        self.model_tag = 'ResNet18'
        self.target_attr_idx = 18
        self.bias_attr_idx = 20
        self.main_num_steps = 636 * 200
        self.main_valid_freq = 636
        self.main_batch_size = 256
        self.main_learning_rate = 1e-4
        self.main_weight_decay = 1e-4
        # main_tag = 'CelebA-{}-{}'.format(target_attr_idx, bias_attr_idx)
        # log_dir = os.path.join(log_dir, 'celeba')
        print('init')

        self.train_dataset = get_dataset(
            'CelebA',
            data_dir="/raid/ysharma_me/fair_lr/domain_independent/datasets/debias",
            dataset_split="train",
            transform_split="train"
        )

        self.valid_dataset = get_dataset(
            'CelebA',
            data_dir="/raid/ysharma_me/fair_lr/domain_independent/datasets/debias",
            dataset_split="eval",
            transform_split="eval"
        )

        print('inout')
        
        train_target_attr = self.train_dataset.attr[:, self.target_attr_idx]
        train_bias_attr = self.train_dataset.attr[:, self.bias_attr_idx]
        attr_dims = []
        attr_dims.append(torch.max(train_target_attr).item() + 1)
        attr_dims.append(torch.max(train_bias_attr).item() + 1)
        self.num_classes = attr_dims[0]
        self.num_domains = attr_dims[1]

        self.set_network(opt)
        self.set_data(opt)
        self.set_optimizer(opt)

    def forward(self, x):
        out = self.network(x)
        return out

    def set_network(self, opt):
        """Define the network"""
        model_tag = 'ResNet18'
        self.network = get_model(model_tag, self.num_classes* self.num_domains).to(self.device)

    def set_data(self, opt):
        """Set up the dataloaders"""

        # train_target_attr = train_dataset.attr[:, target_attr_idx]
        # train_bias_attr = train_dataset.attr[:, bias_attr_idx]
        # attr_dims = []
        # attr_dims.append(torch.max(train_target_attr).item() + 1)
        # attr_dims.append(torch.max(train_bias_attr).item() + 1)
        # num_classes = attr_dims[0]
        # print(self.train_dataset[0])
        # iirtr
        # self.train_dataset = IdxDataset(self.train_dataset)
        # self.valid_dataset = IdxDataset(self.valid_dataset)


        
        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=main_batch_size,
        #     shuffle=True,
        #     num_workers=16,
        #     pin_memory=True,
        # )

        # valid_loader = DataLoader(
        #     valid_dataset,
        #     batch_size=256,
        #     shuffle=False,
        #     num_workers=16,
        #     pin_memory=True,
        # )
        print('datain')
        
        train_data = dataloader.CelebaDatasetLff(self.train_dataset, self.target_attr_idx, self.bias_attr_idx, self.num_classes, self.num_domains)
        # iitr
        test_data = dataloader.CelebaDatasetLff_test(self.train_dataset, self.target_attr_idx, self.bias_attr_idx, self.num_classes, self.num_domains)
        # test_gray_data = dataloader.CelebaDataset(data_setting['test_gray_path'], 
        #                                          data_setting['test_label_path'],
        #                                          transform_test)
        print('yo')

        self.train_loader = torch.utils.data.DataLoader(
                                 train_data, batch_size=opt['batch_size'],
                                 shuffle=True, num_workers=1)
        print('no')
        # iitr
        self.test_loader = torch.utils.data.DataLoader(
                                      test_data, batch_size=opt['batch_size'],
                                      shuffle=False, num_workers=1)
        # self.test_gray_loader = torch.utils.data.DataLoader(
        #                              test_gray_data, batch_size=opt['batch_size'],
        #                              shuffle=False, num_workers=1)
    
    def set_optimizer(self, opt):
        # optimizer_setting = opt['optimizer_setting']
        # torch.optim.Adam(
        #     params=list(F_E.parameters()) + list(C.parameters()),
        #     lr=1e-3,
        #     weight_decay=main_weight_decay,
        # )
        self.optimizer = torch.optim.Adam( 
                            params=self.network.parameters(), 
                            lr=1e-4,
                            weight_decay=1e-4
                            )
        
    def _criterion(self, output, target):
        return F.cross_entropy(output, target)
        
    def state_dict(self):
        state_dict = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }  
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epoch = state_dict['epoch']

    def log_result(self, name, result, step):
        self.log_writer.add_scalars(name, result, step)

    def adjust_lr(self):
        lr = self.init_lr * (0.1 ** (self.epoch // 50))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _train(self, loader):
        """Train the model for one epoch"""
        
        self.network.train()
        self.adjust_lr()
        
        train_loss = 0
        total = 0
        correct = 0
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.forward(images)
            loss = self._criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = correct*100. / total

            train_result = {
                'accuracy': correct*100. / total,
                'loss': loss.item(),
            }
            self.log_result('Train iteration', train_result,
                            len(loader)*self.epoch + i)

            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {}: [{}|{}], loss:{}, accuracy:{}'.format(
                    self.epoch, i+1, len(loader), loss.item(), accuracy
                ))

        self._train_accuracy = accuracy
        self.epoch += 1

    def _test(self, loader):
        """Test the model performance"""
        
        self.network.eval()

        total = 0
        correct = 0
        test_loss = 0
        output_list = []
        feature_list = []
        predict_list = []
        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.forward(images)
                loss = self._criterion(outputs, targets)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                predict_list.extend(predicted.tolist())
                output_list.append(outputs.cpu().numpy())
                # feature_list.append(features.cpu().numpy())

        test_result = {
            'accuracy': correct*100. / total,
            'predict_labels': predict_list,
            'outputs': np.vstack(output_list),
        }
        return test_result

    def train(self):
        self._train(self.train_loader)
        utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'ckpt.pth'))

    def test(self):
        # Test and save the result
        test_color_result = self._test(self.test_color_loader)
        test_gray_result = self._test(self.test_gray_loader)
        utils.save_pkl(test_color_result, os.path.join(self.save_path, 'test_color_result.pkl'))
        utils.save_pkl(test_gray_result, os.path.join(self.save_path, 'test_gray_result.pkl'))
        
        # Output the classification accuracy on test set
        info = ('Test on color images accuracy: {}\n' 
                'Test on gray images accuracy: {}'.format(test_color_result['accuracy'],
                                                          test_gray_result['accuracy']))
        utils.write_info(os.path.join(self.save_path, 'test_result.txt'), info)
        
    


            
