import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from data.util import get_dataset, IdxDataset, ZippedDataset
from module.util import get_model
from models import basenet
from models import dataloader
from models.cifar_core import CifarModel
import utils

class CifarDomainIndependent(CifarModel):
    def __init__(self, opt):
        super(CifarDomainIndependent, self).__init__(opt)
    
    def _criterion(self, output, target):
        class_num = output.size(1) // 2
        logprob_first_half = F.log_softmax(output[:, :class_num], dim=1)
        logprob_second_half = F.log_softmax(output[:, class_num:], dim=1)
        output = torch.cat((logprob_first_half, logprob_second_half), dim=1)
        return F.nll_loss(output, target)
        
    def _test(self, loader, test_on_color=True):
        """Test the model performance"""
        
        self.network.eval()

        total = 0
        correct = 0
        test_loss = 0
        output_list = []
        feature_list = []
        target_list = []
        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                images, targets = images.to(self.device), targets.to(self.device)
                outputs, features = self.forward(images)
                loss = self._criterion(outputs, targets)
                test_loss += loss.item()

                output_list.append(outputs)
                feature_list.append(features)
                target_list.append(targets)
                
        outputs = torch.cat(output_list, dim=0)
        features = torch.cat(feature_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        
        # accuracy_conditional = self.compute_accuracy_conditional(outputs, targets, test_on_color)
        accuracy_sum_out = self.compute_accuracy_sum_out(outputs, targets)
        
        test_result = {
            # 'accuracy_conditional': accuracy_conditional,
            'accuracy_sum_out': accuracy_sum_out,
            'outputs': outputs.cpu().numpy(),
            'features': features.cpu().numpy()
        }
        return test_result
    
    def compute_accuracy_conditional(self, outputs, targets, test_on_color):
        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()
        
        class_num = outputs.shape[1] // 2
        if test_on_color:
            outputs = outputs[:, :class_num]
        else:
            outputs = outputs[:, class_num:]
        predictions = np.argmax(outputs, axis=1)
        accuracy = (predictions == targets).mean() * 100.
        return accuracy
    
    def compute_accuracy_sum_out(self, outputs, targets):
        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()
        
        class_num = outputs.shape[1] // 2
        predictions = np.argmax(outputs[:, :class_num] + outputs[:, class_num:], axis=1)
        accuracy = (predictions == targets).mean() * 100.
        return accuracy

    def test(self):
        # Test and save the result
        # state_dict = torch.load(os.path.join(self.save_path, 'ckpt.pth'))
        # self.load_state_dict(state_dict)
        # test_color_result = self._test(self.test_color_loader, test_on_color=True)
        # test_gray_result = self._test(self.test_gray_loader, test_on_color=False)
        # utils.save_pkl(test_color_result, os.path.join(self.save_path, 'test_color_result.pkl'))
        # utils.save_pkl(test_gray_result, os.path.join(self.save_path, 'test_gray_result.pkl'))
        
        # # Output the classification accuracy on test set for different inference
        # # methods
        # info = ('Test on color images accuracy conditional: {}\n' 
        #         'Test on color images accuracy sum out: {}\n'
        #         'Test on gray images accuracy conditional: {}\n'
        #         'Test on gray images accuracy sum out: {}\n'
        #         .format(test_color_result['accuracy_conditional'],
        #                 test_color_result['accuracy_sum_out'],
        #                 test_gray_result['accuracy_conditional'],
        #                 test_gray_result['accuracy_sum_out']))
        # utils.write_info(os.path.join(self.save_path, 'test_result.txt'), info)

        state_dict = torch.load(os.path.join(self.save_path, 'ckpt.pth'))
        self.load_state_dict(state_dict)
        test_result = self._test(self.test_loader, test_on_color=True)
        # test_gray_result = self._test(self.test_gray_loader, test_on_color=False)
        utils.save_pkl(test_result, os.path.join(self.save_path, 'test_result.pkl'))
        # utils.save_pkl(test_gray_result, os.path.join(self.save_path, 'test_gray_result.pkl'))
        
        # import numpy as np
        # class_num = 10
        # gray_out=test_gray_result['outputs']
        # color_out=test_color_result['outputs']
        # bias=0
        # gray_res = np.argmax(gray_out[:, :class_num] + gray_out[:, class_num:], axis=1)
        # color_res = np.argmax(color_out[:, :class_num] + color_out[:, class_num:], axis=1)
        # for i in range(10):
        #   Gr=(np.array(gray_res)==i).sum()
        #   Col=(np.array(color_res)==i).sum()
        #   bias=bias+((max(Gr,Col)/(Gr+Col))-0.5)
        # bias=bias/10
        
        # Output the classification accuracy on test set for different inference
        # methods
        info = ('Test accuracy sum out: {}\n'.format(test_result['accuracy_sum_out']))
        utils.write_info(os.path.join(self.save_path, 'test_result.txt'), info)
    
    