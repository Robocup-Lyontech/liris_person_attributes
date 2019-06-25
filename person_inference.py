# -*- coding: utf-8 -*-
# *****************************************************************************
#
# Apply person attributes prediction
#
# *****************************************************************************

from __future__ import print_function, division
import sys
import numpy as np
import torch
from torch.autograd import Variable
from scipy.misc import imread, imsave, imshow
from skimage.transform import rescale, resize
from torchvision import transforms

try:
  from . import person
  from . import loader_rapdataset_yiqiang
  from . import loader_peta_dataset
  from . import loader_rap_plus_peta_dataset
except:
  import person
  import loader_rapdataset_yiqiang
  import loader_peta_dataset
  import loader_rap_plus_peta_dataset
  


#attributes_sorted_by_accuracy = ['0.897_hs-BaldHead', '0.895_occlusionDown', '0.881_faceBack', '0.880_occlusionLeft', '0.876_occlusion-Environment', '0.873_action-Pusing', '0.872_low-Blue', '0.867_lb-Jeans', '0.865_occlusion-Other', '0.865_faceFront', '0.864_occlusionRight', '0.852_low-Black', '0.851_shoes-Cloth', '0.842_up-Blue', '0.840_occlusionUp', '0.839_up-Orange', '0.832_AgeLess16', '0.830_up-Black', '0.828_Female', '0.813_lb-LongTrousers', '0.812_low-Yellow', '0.812_hs-Muffler', '0.809_faceRight', '0.808_up-Red', '0.807_up-Purple', '0.806_Clerk', '0.790_shoes-Green', '0.788_up-Pink', '0.788_shoes-Black', '0.786_attach-Backpack', '0.785_faceLeft', '0.775_shoes-White', '0.774_low-Red', '0.768_occlusion-Attachment', '0.766_hs-LongHair', '0.764_shoes-Leather', '0.760_Customer', '0.756_occlusion-Person', '0.756_lb-TightTrousers', '0.750_up-Yellow', '0.747_low-Green', '0.746_attach-PaperBag', '0.739_shoes-Blue', '0.731_action-Pulling', '0.727_up-Green', '0.726_shoes-Red', '0.723_ub-Vest', '0.722_attach-HandTrunk', '0.718_up-Mixture', '0.716_shoes-Boots', '0.715_ub-ShortSleeve', '0.710_hs-Hat', '0.709_up-Gray', '0.709_lb-Skirt', '0.708_low-Mixture', '0.708_lb-ShortSkirt', '0.708_lb-Dress', '0.705_ub-Shirt', '0.694_ub-SuitUp', '0.690_shoes-Sport', '0.689_ub-Cotton', '0.684_up-White', '0.683_attach-HandBag', '0.680_shoes-Yellow', '0.677_up-Brown', '0.673_shoes-Brown', '0.669_ub-Jacket', '0.667_low-White', '0.665_action-Holding', '0.664_shoes-Mixture', '0.664_action-CarrybyArm', '0.663_attach-PlasticBag', '0.659_hs-BlackHair', '0.658_low-Gray', '0.648_Age31-45', '0.641_Age17-30', '0.638_action-Calling', '0.619_attach-Box', '0.604_action-CarrybyHand', '0.602_ub-TShirt', '0.602_shoes-Gray', '0.600_action-Talking', '0.594_attach-Other', '0.593_ub-Tight', '0.568_hs-Glasses', '0.567_shoes-Casual', '0.558_BodyFat', '0.556_attach-SingleShoulderBag', '0.546_ub-Sweater', '0.545_action-Gathering', '0.517_BodyThin', '0.515_BodyNormal']


# # # # # # # # # # #
class Person_attributes:

  # # # # # # # # # # #
  def __init__(self, dataset_name, model_filename, doGPU):
    """
    Initialize model.
    """
    # set attributes list
    print("initializing labels for dataset ", dataset_name)
    if dataset_name == 'RAP':
      self.labels = loader_rapdataset_yiqiang.ATTRIBUTES
    elif dataset_name == 'PETA':
      self.labels = loader_peta_dataset.ATTRIBUTES
    elif dataset_name == 'RAPPETA':
      self.labels = [peta_label for rap_label,peta_label in loader_rap_plus_peta_dataset.ATTRIBUTES]
    else:
      print("ERROR: unknown dataset.")
      sys.exit()

    self.doGPU = doGPU
    self.transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # create model
    person.NO_ATTRIBUTES = len(self.labels) #TODO-elo: ugly, attr. nbr should be a parameter of person.Net.__init__()
    if self.doGPU:
      self.net = person.Net().cuda()
    else:
      self.net = person.Net()

    # load model
    print('loading model "' + model_filename + '"')
    person.load_model(self.net, model_filename)
        

  # # # # # # # # # # #
  def predict(self, image):
    """
    Do prediction on one image.
    """
    # preprocess image
    inp = self.preprocess_image(image)

    # do inference
    with torch.no_grad():
      if self.doGPU:
          inp = Variable(inp.float().cuda())
      else:
          inp = Variable(inp.float())
      # at this point inp.shape = torch.Size([1, 3, 128, 48])

      logits = self.net.forward(inp)
      predictions = (logits > 0).int()

    #import ipdb ; ipdb.set_trace()
    # build prediction result list [(label, prediction, confidence)]
    assert (predictions.shape[1] == len(self.labels))
    pred_list = []
    for label, prediction, confidence in zip(self.labels, predictions[0], logits[0]):
      pred_list.append((label, prediction.item(), confidence.item()))

    return pred_list
 

  # # # # # # # # # # #
  def preprocess_image(self, image):
    """
    Preprocess one image.
    """
    # resize and transform image
    image = resize(image, (128,48), anti_aliasing=True, mode='constant')
    image = 255 * image
    image = image.astype(np.uint8)
    image = self.transform(image)

    # save transformed image for debugging
    if False:
      image_tmp = image.numpy() # Torch tensor to Numpy array
      #image_tmp = self.normalize(image_tmp) # shape is (3,128,48)
      image_tmp = np.moveaxis(image_tmp, 0, -1) # shape is (128,48,3)
      imsave('image_transformed.png', image_tmp)

    image = image[None,:,:,:] # simulate a batch of size 1
    return image


# # # # # # # # # # #
def main():
  """
  Do prediction on images.
  """
  # process parameters
  if len(sys.argv) < 3:
    print("Usage: " + sys.argv[0] + "  RAP|PETA|RAPPETA  model_filename  image1 image2 ...")
    sys.exit(1)

  ds_name = sys.argv[1]
  model_filename = sys.argv[2]
  image_files = sys.argv[3:]
  doGPU = True

  person_attributes = Person_attributes(ds_name, model_filename, doGPU)  

  # loop over input images
  for image_file in image_files:
    print('processing image ', image_file)

    # load input image
    image = imread(image_file)

    # do prediction
    prediction = person_attributes.predict(image)

    # print prediction
    for p in prediction:
      label, pred, confidence = p
      #print(label + ' ' + str(pred))
      print('{:s} {:d} {:.2f}'.format(label, pred, confidence))
    print()

    # display image
    imshow(image)
    try:
      # fails on Python27
      input('Enter to continue...')
    except:
      pass


# # # # # # # # # # #
if __name__ == "__main__":
  main()

