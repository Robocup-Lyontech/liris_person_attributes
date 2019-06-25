# -*- coding: utf-8 -*-
# *****************************************************************************
#
# Evaluate RAPPETA trained model on RAP or PETA or RAPPETA dataset
#
# *****************************************************************************

from __future__ import print_function, division

import sys
import math
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

import person
import loader_rapdataset_yiqiang
import loader_peta_dataset
import loader_rap_plus_peta_dataset


# ************************************************************************
# Parameters
# ************************************************************************

Param_Batchsize  = 50 #1  #50
Param_Nb_Workers = 2  #0  #2


# ************************************************************************
# Functions
# ************************************************************************

#################################################

def init_model(model_filename, doGPU):
  """
  Initialize model and model attributes.
  """
  # set model attributes list
  ##print("Model-dataset =", model_ds_name)
  ##if model_ds_name == 'modelRAP':
  ##  model_labels = loader_rapdataset_yiqiang.ATTRIBUTES
  ##elif model_ds_name == 'modelPETA':
  ##  model_labels = loader_peta_dataset.ATTRIBUTES
  ##elif model_ds_name == 'modelRAPPETA':
  ##  model_labels = [peta_label for rap_label,peta_label in loader_rap_plus_peta_dataset.ATTRIBUTES]
  ##else:
  ##  print("ERROR: unknown model-dataset.")
  ##  sys.exit()
  model_labels = loader_rap_plus_peta_dataset.ATTRIBUTES
  assert (len(model_labels) == 49)

  # create model
  person.NO_ATTRIBUTES = len(model_labels) #TODO-elo: ugly, attr. nbr should be a parameter of person.Net.__init__()
  net = person.Net()
  if doGPU:
    net = person.Net().cuda()

  # load model
  print('loading model "' + model_filename + '"')
  person.load_model(net, model_filename)

  return net, model_labels

#################################################

def init_dataset(validation_dataset_name):
  """
  Initialize dataloader and dataset attributes list.
  """
  transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  
  if validation_dataset_name == 'datasetRAP':
    # validation = 8317 images = 166 batches of 50 images + 1 batch of 17 images
    dataset_valid = loader_rapdataset_yiqiang.RAPDataset(0,False,'/storage/Datasets/Rap-PedestrianAttributeRecognition/',transform)
    labels = loader_rapdataset_yiqiang.ATTRIBUTES
    datset_attr_nbr = 92
  elif validation_dataset_name == 'datasetPETA':
    dataset_valid = loader_peta_dataset.PETADataset(False, '/storage/Datasets/PETA-PEdesTrianAttribute', transform)
    labels = loader_peta_dataset.ATTRIBUTES
    datset_attr_nbr = 104
  elif validation_dataset_name == 'datasetRAPPETA':
    dataset_valid = loader_rap_plus_peta_dataset.RAPPlusPETADataset(False, '/storage/Datasets/Rap-PedestrianAttributeRecognition/', '/storage/Datasets/PETA-PEdesTrianAttribute', transform)
    labels = [peta_label for rap_label,peta_label in loader_rap_plus_peta_dataset.ATTRIBUTES]
    datset_attr_nbr = 49

  print ("Dataset valid size :", dataset_valid.__len__())
  print ("Dataset Attributes number :", datset_attr_nbr)
  assert (len(labels) == datset_attr_nbr)

  dataloader_valid = DataLoader(dataset_valid, batch_size=Param_Batchsize, shuffle=True, num_workers=Param_Nb_Workers)

  return dataloader_valid, dataset_valid

#################################################

def calcError(net, net_labels, dataset_name, dataloader, dataset, doGPU):
  """
  Calculate prediction error and class accuracy.
  """
  # note: net_labels is a list of pairs (RAP_name, PETA_name) of attribute names
  net_attr_nbr = len(net_labels)
  assert (net_attr_nbr == 49)
  
  total = 0
  correct = 0
  batch_nbr = 0
  per_attrib_total = torch.zeros([net_attr_nbr], dtype=torch.int64) # size [92]
  per_attrib_correct = torch.zeros([net_attr_nbr], dtype=torch.int64) # size [92]
  per_attrib_1_pred = torch.zeros([net_attr_nbr], dtype=torch.int64) # size [92]
  per_attrib_class_accuracy = torch.zeros([net_attr_nbr], dtype=torch.float) # size [92]
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
      total += real_batch_size * net_attr_nbr
      per_attrib_total += real_batch_size # size [net_attr_nbr]
      assert (per_attrib_total.sum().item() == total)
      try:
        assert (batch_nbr == math.ceil(per_attrib_total[0].item()/Param_Batchsize))
      except AssertionError:
        ipdb.set_trace()
        pass


      # prepare data for prediction
      if doGPU:
          inp = Variable(sample_batched['image'].float().cuda())
      else:
          inp = Variable(sample_batched['image'].float())

      # retrieve ground truth
      dataset_lab_gt = sample_batched['label'] # shape == [50,NB_ATTRIB]

      # convert ground truth to model attributes
      if dataset_name == 'datasetRAPPETA':
        assert (dataset_lab_gt.shape[1] == 49)
        # no conversion needed, use ground truth as it is
        lab_gt = dataset_lab_gt
      elif dataset_name == 'datasetRAP':
        assert (dataset_lab_gt.shape[1] == 92)
        # note: in the line below dataset_lab_gt.shape[0] is better than 
        #       Param_Batchsize because the last batch may be incomplete
        lab_gt = torch.zeros((dataset_lab_gt.shape[0],net_attr_nbr), dtype=dataset_lab_gt.dtype)
        net_labels_RAP = [rap_label for rap_label,peta_label in net_labels]
        for attr_idx,attr_name in enumerate(net_labels_RAP):
          lab_gt[:,attr_idx] = dataset_lab_gt[:,dataset.index_of(attr_name)]
      elif dataset_name == 'datasetPETA':
        assert (dataset_lab_gt.shape[1] == 104)
        # note: in the line below dataset_lab_gt.shape[0] is better than 
        #       Param_Batchsize because the last batch may be incomplete
        lab_gt = torch.zeros((dataset_lab_gt.shape[0],net_attr_nbr), dtype=dataset_lab_gt.dtype)
        net_labels_PETA = [peta_label for rap_label,peta_label in net_labels]
        for attr_idx,attr_name in enumerate(net_labels_PETA):
          lab_gt[:,attr_idx] = dataset_lab_gt[:,dataset.index_of(attr_name)]
      else:
        print('Unknown dataset \'' + dataset_name + '\'')
        sys.exit(1)

      # 'format' ground truth for Torch
      lab_gtv = Variable(lab_gt)
      if doGPU:
          lab_gtv = lab_gtv.cuda()

      # do prediction
      logits = net.forward(inp)  # output without Sigmoid
      predictions = (logits > 0).int() # size [50, net_attr_nbr]
      assert (net_attr_nbr == predictions.shape[1])

      # accumulate total number of correct predictions
      correct += (lab_gtv == predictions).sum()

      # accumulate per-attribute number of correct predictions
      per_batch_and_attrib_correct = (lab_gtv == predictions) # size [50, net_attr_nbr]
      #if doGPU:
      #  per_batch_and_attrib_correct = per_batch_and_attrib_correct.cpu()
      per_attrib_correct += per_batch_and_attrib_correct.sum(0) # size [net_attr_nbr]
      assert (per_attrib_correct.sum().item() == correct)

      # accumulate number of 1 predictions for each attribute
      per_attrib_1_pred += predictions.sum(0) # size [net_attr_nbr]

      # accumulate for class-accuracy
      per_batch_and_attrib_1_good_prediction = (predictions.byte() * per_batch_and_attrib_correct).sum(0) #size [net_attr_nbr]
      per_batch_and_attrib_0_good_prediction = ((1 - predictions.byte()) * per_batch_and_attrib_correct).sum(0) #size [net_attr_nbr]
      assert torch.equal(per_batch_and_attrib_1_good_prediction + per_batch_and_attrib_0_good_prediction, per_batch_and_attrib_correct.sum(0))
      per_batch_and_attrib_1_ground_truth = lab_gtv.sum(0) #size [net_attr_nbr]
      per_batch_and_attrib_0_ground_truth = (1 - lab_gtv).sum(0) #size [net_attr_nbr]
      try:
        assert torch.equal(per_batch_and_attrib_1_ground_truth + per_batch_and_attrib_0_ground_truth, torch.tensor([real_batch_size] * net_attr_nbr).cuda())
      except AssertionError:
        print("per_batch_and_attrib_1_ground_truth + per_batch_and_attrib_0_ground_truth=")
        print(per_batch_and_attrib_1_ground_truth + per_batch_and_attrib_0_ground_truth)
        ipdb.set_trace()
        pass

      per_batch_and_attrib_recall_1 = per_batch_and_attrib_1_good_prediction.float() / per_batch_and_attrib_1_ground_truth.float() #size [net_attr_nbr]
      # nan values appear when ground_truth number of 1 value is 0
      # in this case, good_prediction can not be different of 0
      # (there can not be a good prediction of 1 because there is not
      # any 1 in the ground truth)
      # so a nan appears only when recall = 0 good pred / 0 case in ground truth
      # so recall=nan can be safely replaced by a recall=1
      person.replace_nan_by_one(per_batch_and_attrib_recall_1)
      per_batch_and_attrib_recall_0 = per_batch_and_attrib_0_good_prediction.float() / per_batch_and_attrib_0_ground_truth.float() #size [net_attr_nbr]
      person.replace_nan_by_one(per_batch_and_attrib_recall_0)
      # class_accuracy = mean(recall_of_0, recall_of_1)
      per_batch_and_attrib_class_accuracy = (per_batch_and_attrib_recall_0 + per_batch_and_attrib_recall_1) / 2.0 #size [net_attr_nbr]
      per_attrib_class_accuracy += per_batch_and_attrib_class_accuracy #size [net_attr_nbr]

    assert (total == (dataloader.dataset.__len__() * net_attr_nbr))
    
    if doGPU:
      per_attrib_total = per_attrib_total.cpu()
      per_attrib_correct = per_attrib_correct.cpu()
      per_attrib_1_pred = per_attrib_1_pred.cpu()
      per_attrib_class_accuracy = per_attrib_class_accuracy.cpu()

    # compute per-attribute and global average prediction error
    err = (1.0-correct.item()/total)
    per_attrib_err = (1.0 - (per_attrib_correct.to(dtype=torch.float) / per_attrib_total.to(dtype=torch.float))) # size [net_attr_nbr]
    np.testing.assert_allclose(per_attrib_err.mean().item(), err, rtol=1e-5)

    # compute per-attribute number of 1 predictions
    per_attrib_1_pred_rate = 100 * (per_attrib_1_pred.to(dtype=torch.float) / per_attrib_total.to(dtype=torch.float)) # size [net_attr_nbr]

    # compute mean class_accuracy over batches
    per_attrib_class_accuracy = per_attrib_class_accuracy * 1.0 / batch_nbr 

    return err, per_attrib_err, per_attrib_1_pred_rate, per_attrib_class_accuracy

#################################################

def main():
  # process parameters
  if len(sys.argv) < 3:
    print("Usage: " + sys.argv[0] + "  RAPPETA_model_filename  datasetRAP|datasetPETA|datasetRAPPETA")
    sys.exit(1)

  ##model_ds_name = sys.argv[1]
  model_filename = sys.argv[1]
  validation_dataset_name = sys.argv[2]
  doGPU = True

  # initialize model
  net, net_labels = init_model(model_filename, doGPU)

  # initialize dataset
  dataloader, dataset = init_dataset(validation_dataset_name)

  # compute the validation error and class accuracy of the model against the dataset
  valid_err, valid_attr_err, valid_attr_1_rate, class_accuracy = calcError(net, net_labels, validation_dataset_name, dataloader, dataset, doGPU)

  # print per-attribute validation-error and class-accuracy
  print("attribute               validation-error  class-accuracy")
  for attr_names, v_error, c_accuracy in zip(net_labels, valid_attr_err, class_accuracy):
    print('{:25s} {:9.2f} {:15.2f}'.format(attr_names[0], v_error, c_accuracy)) 

#################################################

if __name__ == "__main__":
  main()

