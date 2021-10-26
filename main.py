import os
import sys
import torch
import torchvision
from torch import nn
from torch import optim
import argparse
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms
import torchvision.datasets
from torchvision.datasets import ImageFolder
import logging
import datetime
import time
from visdom import Visdom
from torch.optim import lr_scheduler
import numpy as np
import copy
import gc
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from itertools import cycle
from models import *
from center_loss import CenterLoss
from options.train_options import TrainOptions
from cyclegan_train import cycleGAN
from loss.center_loss import CenterLoss
from loss.mmd_loss import MMD_loss
from util import *
from util.util import get_logger
from util.util import AverageMeter
from util.util import generate_gandataset_realAB
from util.util import generate_finetune_dataset
from util.util import get_cross_dataloader
from util.util import evaluate_accuracy
from util.util import evaluate_fake_accuracy
from util.util import save_model
from util.util import get_all_dataloader
from util.majority_voting import calculate_single_prediction
from util.majority_voting import voting_prediction

from data import create_dataset
from data import generation_dataset
from data.deal_dataset import FakeDataset

from models.base_model import FineTuneVGG16

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.backends.cudnn.enabled = False
opt = TrainOptions().parse()   # get training options   
logger = get_logger(logdir=opt.logdir)

device = torch.device("cpu")
opt.use_gpu = True if (device == 'cuda:0') else False
opt.batch_size = 32

temp_name = "finetune_net_params.pkl"
opt.cross_data_root = './lung'
save_path = 'results'

opt.use_pre_gan = True

def train(net, gan_model, criterion_xent, criterion_cent, criterion_cent_cross, criterion_mmd, optimizer_model, optimizer_centloss, optimizer_centloss_cross):
    net = net.to(device)
    print("training on ", device)
    scheduler = lr_scheduler.StepLR(optimizer_model, step_size=1, gamma=0.5)
    total_iters = 0
    best_acc = -100.0
    best_acc_cross = -100.0
    weight_cent = 1.0
    weight_mmd = 2.0
    passing_threshold = 0.75
    log_interval = 5
    best_model = net
    IsEmpty = False
    for epoch in range(opt.epochs):
        epoch_start_time = time.time()
        realA_loader, testA_loader, realB_loader = get_all_dataloader(opt, opt.img_size)
        all_cycle = max(len(realA_loader), len(realB_loader))
        logger.info(all_cycle)
        realA_train_acc_sum, realB_acc_sum, An, Bn, start = 0.0, 0.0, 0, 0, time.time()
        batch_count, epoch_iter, val_acc = 0, 0, 0.0
        all_fake_A_data_list = []
        all_real_B_data_list = []
        all_real_B_labels_list = []
        gan_model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for step_all, tupdata in enumerate(zip(cycle(realA_loader), cycle(realB_loader))):
            if(batch_count == all_cycle):
                break
            cross_log_interval = len(realB_loader)
            realA_data, realA_labels, realB_data, realB_labels = tupdata[0][0], tupdata[0][1], tupdata[1][0], tupdata[1][1]
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            batch_count += 1
            gan_dataset = generate_gandataset_realAB(realA_data, realA_labels, realB_data, realB_labels) # dataloader
            iter_data_time = time.time()
            dataset_size = len(gan_dataset)    # get the number of images in the dataset.
            print('The number of training images = %d' % dataset_size)
            logger.info('The number of training images = %d' % dataset_size)
            real_A_list, real_B_list, fake_A_list, rec_A_list, idt_B_list = [], [], [], [], []
            for i, gan_data in enumerate(gan_dataset):
              gan_data['A'] = torch.unsqueeze(gan_data['A'], 0)
              gan_data['B'] = torch.unsqueeze(gan_data['B'], 0)
              real_A, real_B, fake_A, rec_A, idt_B = cycleGAN(opt, gan_data, gan_model, dataset_size, total_iters, epoch_iter, iter_start_time)
              real_A_list.append(real_A.data)
              real_B_list.append(real_B.data)
              fake_A_list.append(fake_A.data)
              rec_A_list.append(rec_A.data)
              idt_B_list.append(idt_B.data)
              all_fake_A_data_list.append(fake_A.data)
              all_real_B_data_list.append(real_B.data)
            real_A_list = torch.cat([real_A_list[i] for i in range(len(real_A_list))], 0)
            real_B_list = torch.cat([real_B_list[i] for i in range(len(real_B_list))], 0)
            fake_A_list = torch.cat([fake_A_list[i] for i in range(len(fake_A_list))], 0)
            rec_A_list = torch.cat([rec_A_list[i] for i in range(len(rec_A_list))], 0)
            idt_B_list = torch.cat([idt_B_list[i] for i in range(len(idt_B_list))], 0)
            all_real_B_labels_list.append(realB_labels)
            
            # classify data with finetune
            net.train() 
            A_data, A_labels = generate_finetune_dataset(real_A_list, realA_labels, rec_A_list, idt_B_list)
            A_data = A_data.to(device)
            A_labels = A_labels.to(device)
            A_outputs_cross, A_labels_cross = A_data, A_labels
            A_outputs = net(A_data)
            #print(A_outputs, A_labels)
            loss_xent = criterion_xent(A_outputs, A_labels)
            loss_cent = criterion_cent(A_outputs, A_labels)
            loss_cent *= weight_cent
            loss = loss_xent + loss_cent
            optimizer_model.zero_grad()
            optimizer_centloss.zero_grad()
            loss.backward()
            optimizer_model.step()
            # by doing so, weight_cent would not impact on the learning of centers
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / weight_cent)
            optimizer_centloss.step()
            
            A_prediction = (A_outputs.argmax(dim=1) == A_labels).sum().cpu().item()
            realA_train_acc_sum += A_prediction
            An += A_labels.shape[0]
            realA_train_acc = realA_train_acc_sum/float(An)

            # classify cross-domin data
            if(IsEmpty == False):
              torch.save(net, temp_name)
              net_cross = torch.load(temp_name)
              net_cross = net_cross.to(device)
              optimizer_model_cross = torch.optim.SGD(net_cross.parameters(), lr=0.001, weight_decay=5e-04, momentum=0.9)
              scheduler_cross = lr_scheduler.StepLR(optimizer_model_cross, step_size=1, gamma=0.5)
              if(val_acc > passing_threshold):
                IsEmpty = True
                logger.info('Training by itself and stopping passing parameters')
            net_cross.train() 
            real_A_list = real_A_list.to(device)
            realA_labels = realA_labels.to(device)
            realA_outputs = net_cross(real_A_list)
            
            real_B_list = real_B_list.to(device)
            realB_labels = realB_labels.to(device)
            realB_outputs = net_cross(real_B_list)

            A_outputs_data = net_cross(A_outputs_cross)
            loss_xent_cross = criterion_xent(A_outputs_data, A_labels_cross)
            loss_cent_cross = criterion_cent_cross(A_outputs_data, A_labels_cross)
            loss_mmd = criterion_mmd(realA_outputs, realB_outputs)
            
            loss_cent_cross *= weight_cent
            loss_mmd *= weight_mmd
            loss_cross = loss_xent_cross + loss_cent_cross + loss_mmd
            
            optimizer_model_cross.zero_grad()
            optimizer_centloss_cross.zero_grad()
            loss_cross.backward()
            optimizer_model_cross.step()
            # by doing so, weight_cent would not impact on the learning of centers
            for param in criterion_cent_cross.parameters():
                param.grad.data *= (1. / weight_cent)
            optimizer_centloss_cross.step()
            
            B_prediction = (realB_outputs.argmax(dim=1) == realB_labels).sum().cpu().item()
            realB_acc_sum += B_prediction
            Bn += realB_labels.shape[0]
            realB_acc = realB_acc_sum / float(Bn)
            with torch.no_grad():
              if(batch_count % log_interval == 0 or batch_count == all_cycle or batch_count == 1):
                print("Epoch {} Batch {}/{}\t Loss_finetune {:.6f} XentLoss_finetune {:.6f} CenterLoss {:.6f} Loss_cross {:.6f} XentLoss_cross {:.6f} CentLoss_cross {:.6f} MMDLoss {:.6f} RealA_TrainAcc {:.6f}, RealB_Acc {:.6f}".format(epoch + 1, batch_count, all_cycle, loss, loss_xent, loss_cent, loss_cross, loss_xent_cross, loss_cent_cross, loss_mmd, realA_train_acc, realB_acc))
                
                logger.info("Epoch {} Batch {}/{}\t Loss_finetune {:.6f} XentLoss_finetune {:.6f} CenterLoss {:.6f} Loss_cross {:.6f} XentLoss_cross {:.6f} CentLoss_cross {:.6f} MMDLoss {:.6f} RealA_TrainAcc {:.6f}, RealB_Acc {:.6f}".format(epoch + 1, batch_count, all_cycle, loss, loss_xent, loss_cent, loss_cross, loss_xent_cross, loss_cent_cross, loss_mmd, realA_train_acc, realB_acc))
                net.eval()
                val_acc, val_cnt0, val_cnt1 = evaluate_accuracy(testA_loader, net)
                
                print('Epoch %d, Batch %d/%d\t, val acc %.6f, val_0 %d, val_1 %d, time %.3f sec'
                      % (epoch + 1, batch_count, all_cycle, val_acc, val_cnt0, val_cnt1, time.time() - start))
                logger.info('Epoch %d, Batch %d/%d\t, val acc %.6f, val_0 %d, val_1 %d, time %.3f sec'
                      % (epoch + 1, batch_count, all_cycle, val_acc, val_cnt0, val_cnt1, time.time() - start))
                
                # save model
                if val_acc >= best_acc:
                    best_model = net
                    best_acc = val_acc
                    print('Best acc of finetune = %.6f' % (best_acc))
                    logger.info('Best acc of finetune = %.6f' % (best_acc))
                    save_model(save_path, best_acc, epoch, net, optimizer_model, 'FINETUNE')
            # save cross model
              if(batch_count % cross_log_interval == 0 or batch_count % (log_interval*10) == 0 or batch_count == all_cycle or batch_count == 1):
                all_real_B_labels_temp = torch.cat([all_real_B_labels_list[i] for i in range(len(all_real_B_labels_list))], 0)
                all_fake_A_dataloader = get_cross_dataloader(all_fake_A_data_list, all_real_B_labels_temp)
                all_real_B_dataloader = get_cross_dataloader(all_real_B_data_list, all_real_B_labels_temp)

                pre_ff_list, prob_ff_list = [], []
                pre_fc_list, prob_fc_list = [], []
                pre_rf_list, prob_rf_list = [], []
                pre_rc_list, prob_rc_list = [], []
                net.eval()
                net_cross.eval()
                for step, (A_x, A_y) in enumerate(all_fake_A_dataloader):
                  pre, prob = calculate_single_prediction(A_x, net, device)
                  pre_ff_list.append(pre)
                  prob_ff_list.append(prob)
                for step, (B_x, B_y) in enumerate(all_real_B_dataloader):
                  pre_cross, prob_cross = calculate_single_prediction(B_x, net_cross, device)
                  pre_rc_list.append(pre_cross)
                  prob_rc_list.append(prob_cross)
               
                cross_acc, cross_cnt0, cross_cnt1 = voting_prediction(opt.use_gpu, all_real_B_labels_temp, pre_ff_list, prob_ff_list, pre_rc_list, prob_rc_list)  
                
                if(cross_cnt0 == 0 or cross_cnt1 == 0): #aviod overfitting to some degree
                    IsEmpty = False
                if(batch_count % cross_log_interval == 0):
                  logger.info('satisfy cross_log_interval!!')
                  all_fake_A_data_list, all_real_B_data_list = [], []
                  all_real_B_labels_list = []
                
                del all_fake_A_dataloader
                del all_real_B_dataloader
                del all_real_B_labels_temp
                gc.collect()
                print('Epoch %d, Batch %d/%d\t, cross acc %.6f, cross_0 %d, cross_1 %d, time %.3f sec' % (epoch + 1, batch_count, all_cycle, cross_acc, cross_cnt0, cross_cnt1, time.time() - start))
                logger.info('Epoch %d, Batch %d/%d\t, cross acc %.6f, cross_0 %d, cross_1 %d, time %.3f sec' % (epoch + 1, batch_count, all_cycle, cross_acc, cross_cnt0, cross_cnt1, time.time() - start))
                if (cross_acc >= best_acc_cross and batch_count > 1):
                    best_acc_cross = cross_acc
                    print('Best acc of cross = %.6f' % (best_acc_cross))
                    logger.info('Best acc of cross = %.6f' % (best_acc_cross))
                    save_model(save_path, best_acc_cross, epoch, net, optimizer_model_cross, 'MIXED_FINETUNE')
                    save_model(save_path, best_acc_cross, epoch, net_cross, optimizer_model_cross, 'MIXED_CROSS')
                if batch_count == all_cycle:
                    print('Acc of the final batch = %.6f' % (cross_acc))
                    logger.info('Acc of the final batch = %.6f' % (cross_acc))
                    save_model(save_path, cross_acc, epoch, net, optimizer_model_cross, 'Final_FINETUNE')
                    save_model(save_path, cross_acc, epoch, net_cross, optimizer_model_cross, 'Final_CROSS')
        scheduler.step()
        scheduler_cross.step()
        gc.collect()
        val_acc, val_cnt0, val_cnt1 = evaluate_accuracy(testA_loader, net)
        print('Epoch %d, val acc %.6f, val_0 %d, val_1 %d, time %.3f sec'
              % (epoch + 1, val_acc, val_cnt0, val_cnt1, time.time() - start))
        logger.info('Epoch %d, val acc %.6f, val_0 %d, val_1 %d, time %.3f sec'
              % (epoch + 1, val_acc, val_cnt0, val_cnt1, time.time() - start))
        # save CycleGAN model
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        gan_model.save_networks('latest')
        gan_model.save_networks(temp_name)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)) 
        if val_acc >= best_acc:
            best_model = net
            best_acc = val_acc
            save_model(save_path, best_acc, epoch, net, optimizer_model, 'FINETUNE')
            print('Best acc of finetune = %.6f' % (best_acc))
            logger.info('Best acc of finetune = %.6f' % (best_acc))

    test_acc, test_cnt0, test_cnt1 = evaluate_accuracy(testA_loader, best_model)    
    print('test acc %.6f, test_0 %d, test_1 %d, time %.3f sec'
              % (test_acc, test_cnt0, test_cnt1, time.time() - start))
    logger.info('test acc %.6f, test_0 %d, test_1 %d, time %.3f sec'
              % (test_acc, test_cnt0, test_cnt1, time.time() - start))
    
def main():
    used_model = FineTuneVGG16(num_class = opt.class_num)
    for para in list(used_model.parameters())[:-6]:
        para.requires_grad = False
    opt.img_size = 224
    
    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(opt.use_gpu, num_classes=opt.class_num, feat_dim=2)
    criterion_cent_cross = CenterLoss(opt.use_gpu, num_classes=opt.class_num, feat_dim=2)
    criterion_mmd = MMD_loss()
    
    optimizer_model = torch.optim.SGD(used_model.parameters(), lr=0.001, weight_decay=5e-04, momentum=0.9)
    optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=0.5)
    optimizer_centloss_cross = torch.optim.SGD(criterion_cent.parameters(), lr=0.5)

    gan_model = create_model(opt)      # create a model given opt.model and other options
    gan_model.setup(opt)               # regular setup: load and print networks; create schedulers
    # Train
    train(used_model, gan_model, criterion_xent, criterion_cent, criterion_cent_cross, criterion_mmd, optimizer_model, optimizer_centloss, optimizer_centloss_cross)

if __name__=='__main__':
    main()