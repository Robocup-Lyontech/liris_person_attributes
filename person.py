# -*- coding: utf-8 -*-
# *****************************************************************************
# Author: Christian Wolf
#
# Classify person attributes
#
# Changelog:
# 11.04.18 cw: begin development
# *****************************************************************************

from __future__ import print_function, division

# Torch stuff
import torch
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import sys
import time

import math
import numpy

# Our own stuff
try:
  from . import loader_rapdataset_yiqiang
  from . import loader_peta_dataset
  from . import loader_rap_plus_peta_dataset
except:
  import loader_rapdataset_yiqiang
  import loader_peta_dataset
  import loader_rap_plus_peta_dataset


# ************************************************************************
# Parameters
# ************************************************************************

Param_Batchsize  = 50 #1  #50
Param_Epochs     = 50 #10 #50
Param_PrintStats = 100
Param_Nb_Workers = 2  #0  #2
NO_ATTRIBUTES    = 0 #92


# ************************************************************************
# The model
# ************************************************************************

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
        # class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # - in_channels (int) – Number of channels in the input image
        # - out_channels (int) – Number of channels produced by the convolution
        # - kernel_size (int or tuple) – Size of the convolving kernel
        # - stride (int or tuple, optional) – Stride of the convolution. Default: 1
        # - padding (int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
        # - dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        # - groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        # - bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        #ELO- self.conv4 = nn.Conv2d(32, 32, 2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, (1,3), padding=0) # conv sz w3xh1
        #   ??? conv2d or conv1d for C4 and c5 ????
        #ELO- self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 150, (1,3), padding=0) # conv sz w3xh1 ; normalement sortie = 150 channels

        # https://pytorch.org/docs/stable/nn.html?highlight=maxpool2d#torch.nn.MaxPool2d
        # class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # - kernel_size – the size of the window to take a max over
        # - stride – the stride of the window. Default value is kernel_size
        # - padding – implicit zero padding to be added on both sides
        # - dilation – a parameter that controls the stride of elements in the window
        # - return_indices – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later
        # - ceil_mode – when True, will use ceil instead of floor to compute the output shape

        self.pool1 = nn.MaxPool2d(2, 2) 
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.pool3 = nn.MaxPool2d(2, 2) 
        #ELO- self.pool4 = nn.MaxPool2d(2, 2) # ajoute par CW

        #ELO- self.d = nn.Dropout(p=0.5)

        #ELO- self.fc1 = nn.Linear(64*8*3, 500)
        self.fc1 = nn.Linear(150*16*2, 500) #  150x16x2 = C5 output size ?
        self.fc2 = nn.Linear(500, 500)
        assert (NO_ATTRIBUTES != 0)
        self.fc3 = nn.Linear(500, NO_ATTRIBUTES)

        self.sig = nn.Sigmoid()

        self.bn1 = nn.BatchNorm2d(32) # after C1
        self.bn2 = nn.BatchNorm2d(32) # after C2
        self.bn3 = nn.BatchNorm2d(32) # after C3
        #ELO- self.bn4 = nn.BatchNorm2d(64) # after C4
        self.bn4 = nn.BatchNorm2d(32) # after C4
        #ELO- self.bn5 = nn.BatchNorm1d(500) # after C5
        self.bn5 = nn.BatchNorm2d(150) # after C5, ??? 1d ???
        #ELO- self.bn6 = nn.BatchNorm1d(500) # ???
        #ELO- self.bn7 = nn.BatchNorm1d(NO_ATTRIBUTES) # ???

    # Input Image is Batch x 3 x 128 x 48       
    def forward(self, x):
        #print ("   DBG  input:",x.size())

        # C1 + P1
        # Input images are of size 128x48x3
        #ELO- x = F.relu(self.pool1(self.conv1(x)))      
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        #print ("   DBG  C1+P1 output:",x.size())

        # C2 + P2
        # Input size to this layer: 64x24
        #ELO- x = F.relu(self.pool2(self.bn1(self.conv2(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        #print ("   DBG  C2+P2 output:",x.size())

        # C3 + P3
        # Input size to this layer: 32x12
        #ELO- x = F.relu(self.pool3(self.bn2(self.conv3(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        #print ("   DBG  C3+P3 output:",x.size())

        # C4
        # Input size to this layer: 8x3
        #ELO- x = F.relu(self.pool4(self.bn3(self.conv4(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        #print ("   DBG  C4 output:",x.size())

        # C5
        # Input size to this layer: (150x) 8x3
        #ELO- x = F.relu(self.bn4(self.conv5(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        #print ("   DBG  C5 output:",x.size())

        #ELO- # Feature maps are now of size 16*6*32
        #ELO- x = x.view(-1, 64*8*3)
        #ELO- x = F.relu(self.bn5(self.fc1(x)))

        #ELO- # x = self.d(x) # Will NEED TO TURN IT OFF DURING TESTING !!!!

        #ELO- # Output size is now 500
        #ELO- x = F.relu(self.bn6(self.fc2(x)))

        #ELO+ beg
        x = x.view(-1, 150*16*2)  # tensor to vector before fully connected
        # here x.size -> torch.Size([50, 150*16*2=4800])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #ELO+ end

        # Multi-class prediction 
        #ELO- x = self.sig(self.bn7(self.fc3(x)))
        #ELO  disabling sigmoid because BCEWithLogitsLoss() applies the Sigmoid
        #     internally
        #ELO- x = self.sig(x)
        #x = self.bn7(self.fc3(x))

        return x


# ************************************************************************
# Replace NAN values in tensor by 1
# ************************************************************************

def replace_nan_by_one(torch_tensor):
  nan_mask = torch.isnan(torch_tensor)
  torch_tensor[nan_mask] = 1


# ************************************************************************
# Calculate the error of a model on data from a given loader
# ************************************************************************

def calcError (net, dataloader):

  #ELO+
  assert (NO_ATTRIBUTES != 0)
  total = 0
  correct = 0
  batch_nbr = 0
  per_attrib_total = torch.zeros([NO_ATTRIBUTES], dtype=torch.int64) # size [92]
  per_attrib_correct = torch.zeros([NO_ATTRIBUTES], dtype=torch.int64) # size [92]
  per_attrib_1_pred = torch.zeros([NO_ATTRIBUTES], dtype=torch.int64) # size [92]
  per_attrib_class_accuracy = torch.zeros([NO_ATTRIBUTES], dtype=torch.float) # size [92]
  if doGPU:
    per_attrib_total = per_attrib_total.cuda()
    per_attrib_correct = per_attrib_correct.cuda()
    per_attrib_1_pred = per_attrib_1_pred.cuda()
    per_attrib_class_accuracy = per_attrib_class_accuracy.cuda()
    
  with torch.no_grad():
    # loop over batches
    # accumulate per-attribute and total number of correct predictions
    for i_batch, sample_batched in enumerate(dataloader):
      assert (sample_batched['image'].shape[1:] == (3,128,48)), "wrong image size"
      batch_nbr += 1
      real_batch_size = sample_batched['image'].shape[0]
      total += real_batch_size * NO_ATTRIBUTES
      per_attrib_total += real_batch_size # size [92]
      assert (per_attrib_total.sum().item() == total)
      try:
        assert (batch_nbr == math.ceil(per_attrib_total[0].item()/Param_Batchsize))
      except AssertionError:
        #import ipbd ; ipdb.set_trace()
        pass


      # prepare data for prediction
      if doGPU:
          inp = Variable(sample_batched['image'].float().cuda())
      else:
          inp = Variable(sample_batched['image'].float())

      lab_gt = sample_batched['label']
      lab_gtv = Variable(lab_gt)

      if doGPU:
          lab_gtv = lab_gtv.cuda()

      # do prediction
      logits = net.forward(inp)  # output without Sigmoid
      predictions = (logits > 0).int() # size [50, 92]

      # accumulate total number of correct predictions
      correct += (lab_gtv == predictions).sum()

      # accumulate per-attribute number of correct predictions
      per_batch_and_attrib_correct = (lab_gtv == predictions) # size [50, 92]
      #if doGPU:
      #  per_batch_and_attrib_correct = per_batch_and_attrib_correct.cpu()
      per_attrib_correct += per_batch_and_attrib_correct.sum(0) # size [92]
      assert (per_attrib_correct.sum().item() == correct)

      # accumulate number of 1 predictions for each attribute
      per_attrib_1_pred += predictions.sum(0) # size [92]

      # accumulate for class-accuracy
      per_batch_and_attrib_1_good_prediction = (predictions.byte() * per_batch_and_attrib_correct).sum(0) #size [92]
      per_batch_and_attrib_0_good_prediction = ((1 - predictions.byte()) * per_batch_and_attrib_correct).sum(0) #size [92]
      assert torch.equal(per_batch_and_attrib_1_good_prediction + per_batch_and_attrib_0_good_prediction, per_batch_and_attrib_correct.sum(0))
      per_batch_and_attrib_1_ground_truth = lab_gtv.sum(0) #size [92]
      per_batch_and_attrib_0_ground_truth = (1 - lab_gtv).sum(0) #size [92]
      try:
        assert torch.equal(per_batch_and_attrib_1_ground_truth + per_batch_and_attrib_0_ground_truth, torch.tensor([real_batch_size] * NO_ATTRIBUTES).cuda())
      except AssertionError:
        print("per_batch_and_attrib_1_ground_truth + per_batch_and_attrib_0_ground_truth=")
        print(per_batch_and_attrib_1_ground_truth + per_batch_and_attrib_0_ground_truth)
        #import ipbd ; ipdb.set_trace()
        pass

      per_batch_and_attrib_recall_1 = per_batch_and_attrib_1_good_prediction.float() / per_batch_and_attrib_1_ground_truth.float() #size [92]
      # nan values appear when ground_truth number of 1 value is 0
      # in this case, good_prediction can not be different of 0
      # (there can not be a good prediction of 1 because there is not
      # any 1 in the ground truth)
      # so a nan appears only when recall = 0 good pred / 0 case in ground truth
      # so recall=nan can be safely replaced by a recall=1
      replace_nan_by_one(per_batch_and_attrib_recall_1)
      per_batch_and_attrib_recall_0 = per_batch_and_attrib_0_good_prediction.float() / per_batch_and_attrib_0_ground_truth.float() #size [92]
      replace_nan_by_one(per_batch_and_attrib_recall_0)
      # class_accuracy = mean(recall_of_0, recall_of_1)
      per_batch_and_attrib_class_accuracy = (per_batch_and_attrib_recall_0 + per_batch_and_attrib_recall_1) / 2.0 #size [92]
      per_attrib_class_accuracy += per_batch_and_attrib_class_accuracy #size [92]

    assert (total == (dataloader.dataset.__len__() * NO_ATTRIBUTES))
    
    if doGPU:
      per_attrib_total = per_attrib_total.cpu()
      per_attrib_correct = per_attrib_correct.cpu()
      per_attrib_1_pred = per_attrib_1_pred.cpu()
      per_attrib_class_accuracy = per_attrib_class_accuracy.cpu()

    # compute per-attribute and global average prediction error
    err = 100*(1.0-correct.item()/total)
    per_attrib_err = 100.0*(1.0 - (per_attrib_correct.to(dtype=torch.float) / per_attrib_total.to(dtype=torch.float))) # size [92]
    numpy.testing.assert_allclose(per_attrib_err.mean().item(), err, rtol=1e-5)

    # compute per-attribute number of 1 predictions
    per_attrib_1_pred_rate = 100 * (per_attrib_1_pred.to(dtype=torch.float) / per_attrib_total.to(dtype=torch.float)) # size [92]

    # compute mean class_accuracy over batches
    per_attrib_class_accuracy = per_attrib_class_accuracy * 1.0 / batch_nbr 

    return err, per_attrib_err, per_attrib_1_pred_rate, per_attrib_class_accuracy



# ************************************************************************
# Save model to file for future inference
# ************************************************************************

def save_model(net, filename):

  # https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
  torch.save(net.state_dict(), filename)



# ************************************************************************
# Load model from file for inference
# ************************************************************************

def load_model(net, filename):

  # https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
  net.load_state_dict(torch.load(filename))
  net.eval()
      


# ************************************************************************
# Training
# ************************************************************************

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description = 'Robocup Gender Studies')
  parser.add_argument('-gpu', type=str, default='yes')
  parser.add_argument('-LR', type=float, default=0.1)
  parser.add_argument('-dataset', type=str, default='rap')
  args = parser.parse_args()
  print (args)

  doGPU = (args.gpu == 'yes')
  optLearningRate = args.LR
  optDataset = args.dataset

  print ("doGPU .............:", doGPU)
  print ("Learning Rate .....:", optLearningRate)

  print ("Batch size ........:", Param_Batchsize)
  print ("Epochs number .....:", Param_Epochs)
  print ("Batch between stats:", Param_PrintStats)
  print ("Workers number ....:", Param_Nb_Workers)
  print ("Dataset ...........:", optDataset)

  transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  
  if optDataset == 'rap':
    # training = 33268 images = 665 batches of 50 images + 1 batch of 18 images
    dataset_train = loader_rapdataset_yiqiang.RAPDataset(0,True,'/storage/Datasets/Rap-PedestrianAttributeRecognition/',transform)
    # validation = 8317 images = 166 batches of 50 images + 1 batch of 17 images
    dataset_valid = loader_rapdataset_yiqiang.RAPDataset(0,False,'/storage/Datasets/Rap-PedestrianAttributeRecognition/',transform)
    labels = loader_rapdataset_yiqiang.ATTRIBUTES
    NO_ATTRIBUTES = 92
  elif optDataset == 'peta':
    dataset_train = loader_peta_dataset.PETADataset(True, '/storage/Datasets/PETA-PEdesTrianAttribute', transform)
    dataset_valid = loader_peta_dataset.PETADataset(False, '/storage/Datasets/PETA-PEdesTrianAttribute', transform)
    labels = loader_peta_dataset.ATTRIBUTES
    NO_ATTRIBUTES = 104
  elif optDataset == 'rappeta':
    dataset_train = loader_rap_plus_peta_dataset.RAPPlusPETADataset(True, '/storage/Datasets/Rap-PedestrianAttributeRecognition/', '/storage/Datasets/PETA-PEdesTrianAttribute', transform)
    dataset_valid = loader_rap_plus_peta_dataset.RAPPlusPETADataset(False, '/storage/Datasets/Rap-PedestrianAttributeRecognition/', '/storage/Datasets/PETA-PEdesTrianAttribute', transform)
    labels = [peta_label for rap_label,peta_label in loader_rap_plus_peta_dataset.ATTRIBUTES]
    NO_ATTRIBUTES = 49

  print ("Dataset train size :", dataset_train.__len__())
  print ("Dataset valid size :", dataset_valid.__len__())
  print ("Attributes number .:", NO_ATTRIBUTES)
  assert (NO_ATTRIBUTES != 0)
  assert (len(labels) == NO_ATTRIBUTES)

  net = Net()
  if doGPU:
      net = Net().cuda()

  dataloader_train = DataLoader(dataset_train, batch_size=Param_Batchsize, shuffle=True, num_workers=Param_Nb_Workers)
  dataloader_valid = DataLoader(dataset_valid, batch_size=Param_Batchsize, shuffle=True, num_workers=Param_Nb_Workers)
  # note: see drop_last option of Dataloader to drop incomplete batch

  #ELO+  see https://stackoverflow.com/questions/52855843/multi-label-classification-in-pytorch
  #ELO-  multilabelloss = nn.MultiLabelSoftMarginLoss(size_average=False)
  #ELO+  torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
  #multilabelloss = nn.BCEWithLogitsLoss()
  optimizer = optim.SGD(net.parameters(), lr=optLearningRate, momentum=0.9)

  model_basename = 'model_' + time.strftime("%y%m%d-%H%M%S")
  first_time = True
  total_batch_nbr = 0

  # compute the validation error before training
  valid_err, valid_attr_err, valid_attr_1_rate, class_accuracy = calcError (net, dataloader_valid)

  # display global error header
  #print('STAT   Epoch   train-loss   train-err   valid-err')
  print('STAT   Batch   train-loss   train-err   valid-err')
  print('STAT   %5d   %10.3f   %9.3f   %9.3f' % (0, 0.0, 0.0, valid_err))
  # display per-attrib train error header
  print('ATTRIBTRAINERROR   Batch', end='')
  for label in labels:
    print('   ' + label, end='')
  print()
  # display per-attrib valid error header
  print('ATTRIBVALIDERROR   Batch', end='')
  for label in labels:
    print('   ' + label, end='')
  print()
  # display per-attrib valid error
  print('ATTRIBVALIDERROR   %5d' % total_batch_nbr, end='')
  for e in valid_attr_err:
    print('   %9.3f' % e, end='')
  print()
  # display per-attrib 1 prediction on validation base header
  print('ATTRIBVALID1RATE   Batch', end='')
  for label in labels:
    print('   ' + label, end='')
  print()
  # display per-attrib 1 prediction on validation base
  print('ATTRIBVALID1RATE   %5d' % total_batch_nbr, end='')
  for e in valid_attr_1_rate:
    print('   %9.3f' % e, end='')
  print()
  # display per-attrib class accuracy on validation base header
  print('ATTRIBCLASSACCURACY   Batch', end='')
  for label in labels:
    print('   ' + label, end='')
  print()
  # display per-attrib class accuracy on validation base
  print('ATTRIBCLASSACCURACY   %5d' % total_batch_nbr, end='')
  for e in class_accuracy:
    print('   %9.3f' % e, end='')
  print()
  sys.stdout.flush()


  # loop over epochs
  for epoch in range(Param_Epochs):

    print ("Starting epoch ", epoch, " on ", time.strftime("%c"))
    running_loss = 0.0
    running_count = 0
    running_correct = 0
    per_attrib_correct = torch.zeros([NO_ATTRIBUTES], dtype=torch.int64) # size [92]
    per_attrib_count = torch.zeros([NO_ATTRIBUTES], dtype=torch.int64) # size [92]

    # loop over batches
    for i_batch, sample_batched in enumerate(dataloader_train):
      #assert (sample_batched['image'].shape[1:] == (3,128,48)), "wrong image size"
      # print ("[%d]" % i_batch)
      total_batch_nbr += 1

      if doGPU:
          inp = Variable(sample_batched['image'].float().cuda())
      else:
          inp = Variable(sample_batched['image'].float())

      lab_gt = sample_batched['label']
      lab_gt_f = sample_batched['label_f']
      lab_gtv = Variable(lab_gt)
      lab_gtv_f = Variable(lab_gt_f)

      if doGPU:
          lab_gtv = lab_gtv.cuda()
          lab_gtv_f = lab_gtv_f.cuda()

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      logits = net.forward(inp)  # output without Sigmoid
      multilabelloss = nn.BCEWithLogitsLoss()
      if first_time:
        print("version 2 - BCEWithLogitsLoss")
        first_time = False

      ######## WIP ######### begin
      ###  loss = multilabelloss(logits, lab_gtv_f)  # size [] aka single value
      multilabelloss2 = nn.BCEWithLogitsLoss(reduction='none')
      loss2 = multilabelloss2(logits, lab_gtv_f) # size [50, 92]
      loss2 = loss2.mean(0) # size [92]
      if False:
        # boost Female attribute
        attribute_weights = torch.tensor([1.0] * NO_ATTRIBUTES).cuda()
        attribute_weights[0] = 10.0 #100.0 #5.0
        loss2 = loss2 * attribute_weights
      loss = loss2.mean()
      ######## WIP ######### end

      loss.backward() # compute the gradient
      optimizer.step() # update the weights
      predictions = (logits > 0).int()
      # predictions.shape -> torch.Size([50, 92])
      # lab_gtv.shape -> torch.Size([50, 92])
      diff = (lab_gtv == predictions).sum()
      per_attrib_diffs = (lab_gtv == predictions) # size [50, 92]
      per_attrib_diffs = per_attrib_diffs.sum(0) # size [92]

      if doGPU:
          diff = diff.cpu()
          per_attrib_diffs = per_attrib_diffs.cpu()

      running_correct += diff.data.numpy()
      running_count += (Param_Batchsize*NO_ATTRIBUTES)
      per_attrib_correct += per_attrib_diffs
      per_attrib_count += Param_Batchsize
      assert (per_attrib_correct.sum().item() == running_correct)
      assert (per_attrib_count.sum().item() == running_count)

      running_loss += loss.data.item()
      #print("   DBG  i_batch=", i_batch, "  loss.data[0]=", loss.data[0], "  running_loss=", running_loss)

      # Print statistics
      if ((i_batch+1) % Param_PrintStats) == 0:
        # compute loss and train-err for this epoch
        train_err = 100.0*(1.0 - (running_correct/running_count))
        loss_avg = running_loss / Param_PrintStats
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        # compute per attribute error
        train_attr_err = 100.0*(1.0 - (per_attrib_correct.to(dtype=torch.float) / per_attrib_count.to(dtype=torch.float))) # size [92]
        numpy.testing.assert_allclose(train_attr_err.mean().item(), train_err, rtol=1e-5)
        per_attrib_correct = torch.zeros([NO_ATTRIBUTES], dtype=torch.int64)
        per_attrib_count = torch.zeros([NO_ATTRIBUTES], dtype=torch.int64)

        # compute the validation error for this epoch
        valid_err, valid_attr_err, valid_attr_1_rate, class_accuracy = calcError(net, dataloader_valid)

        # display stats
        print('STAT   %5d   %10.3f   %9.3f   %9.3f' % (total_batch_nbr, loss_avg, train_err, valid_err))
        sys.stdout.flush()

        # display per-attribute train error
        print('ATTRIBTRAINERROR   %5d' % total_batch_nbr, end='')
        for e in train_attr_err:
          print('   %9.3f' % e, end='')
        print()
        # display per-attrib valid error
        print('ATTRIBVALIDERROR   %5d' % total_batch_nbr, end='')
        for e in valid_attr_err:
          print('   %9.3f' % e, end='')
        print()
        # display per-attrib 1 prediction on valid base
        print('ATTRIBVALID1RATE   %5d' % total_batch_nbr, end='')
        for e in valid_attr_1_rate:
          print('   %9.3f' % e, end='')
        print()
        # display per-attrib class accuracy on validation base
        print('ATTRIBCLASSACCURACY   %5d' % total_batch_nbr, end='')
        for e in class_accuracy:
          print('   %9.3f' % e, end='')
        print()
        sys.stdout.flush()


        
    # compute and print epoch stats
    """
    # compute loss and train-err for this epoch
    train_err = 100.0*(1.0-running_correct / running_count)
    loss_avg = running_loss / (i_batch+1)
    running_loss = 0.0
    running_correct = 0.0
    running_count = 0.0

    # compute the validation error for this epoch
    valid_err = calcError (net, dataloader_valid)

    # display stats
    print('STAT   %5d   %10.3f   %9.3f   %9.3f' % (epoch+1, loss_avg, train_err, valid_err))
    sys.stdout.flush()
    """

    # save model
    filename = model_basename + '_epoch-{:02d}_batch-{:05d}'.format(epoch, total_batch_nbr)
    print('saving model to file "' + filename + '"')
    save_model(net, filename)

