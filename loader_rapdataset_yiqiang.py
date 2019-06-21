import sys
import numpy as np
import scipy.io as sio
from scipy.misc import imread, imsave, imshow
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.transform import rescale, resize
from skimage import img_as_ubyte
#import ipdb
import PIL  # info: provided by Pillow


#############################

ATTRIBUTES = ['Female', 'AgeLess16', 'Age17-30', 'Age31-45', 'BodyFat', 'BodyNormal', 'BodyThin', 'Customer', 'Clerk', 'hs-BaldHead', 'hs-LongHair', 'hs-BlackHair', 'hs-Hat', 'hs-Glasses', 'hs-Muffler', 'ub-Shirt', 'ub-Sweater', 'ub-Vest', 'ub-TShirt', 'ub-Cotton', 'ub-Jacket', 'ub-SuitUp', 'ub-Tight', 'ub-ShortSleeve', 'lb-LongTrousers', 'lb-Skirt', 'lb-ShortSkirt', 'lb-Dress', 'lb-Jeans', 'lb-TightTrousers', 'shoes-Leather', 'shoes-Sport', 'shoes-Boots', 'shoes-Cloth', 'shoes-Casual', 'attach-Backpack', 'attach-SingleShoulderBag', 'attach-HandBag', 'attach-Box', 'attach-PlasticBag', 'attach-PaperBag', 'attach-HandTrunk', 'attach-Other', 'action-Calling', 'action-Talking', 'action-Gathering', 'action-Holding', 'action-Pusing', 'action-Pulling', 'action-CarrybyArm', 'action-CarrybyHand', 'faceFront', 'faceBack', 'faceLeft', 'faceRight', 'occlusionLeft', 'occlusionRight', 'occlusionUp', 'occlusionDown', 'occlusion-Environment', 'occlusion-Attachment', 'occlusion-Person', 'occlusion-Other', 'up-Black', 'up-White', 'up-Gray', 'up-Red', 'up-Green', 'up-Blue', 'up-Yellow', 'up-Brown', 'up-Purple', 'up-Pink', 'up-Orange', 'up-Mixture', 'low-Black', 'low-White', 'low-Gray', 'low-Red', 'low-Green', 'low-Blue', 'low-Yellow', 'low-Mixture', 'shoes-Black', 'shoes-White', 'shoes-Gray', 'shoes-Red', 'shoes-Green', 'shoes-Blue', 'shoes-Yellow', 'shoes-Brown', 'shoes-Mixture']
# 92 attributes


#############################
class RAPDataset(Dataset):

  #############################
  def __init__(self, num_partition, train, data_dir, transform=None):
    self.RAPdataset_path = os.path.join(data_dir,'RAP_dataset/')
    self.transform = transform
    RAPannotation_path = os.path.join(data_dir+'RAP_annotation/')

    annotation = sio.loadmat(os.path.join(RAPannotation_path,'RAP_annotation.mat'))['RAP_annotation']

    image_names = annotation['imagesname'][0][0]
    label = annotation['label'][0][0]
    partition =  annotation['partion'][0][0]
    if train:
      #train index
      idx = partition[num_partition][0][0][0][0][0]-1
    else:
      #test index
      idx = partition[num_partition][0][0][0][1][0]-1
    
    self.image_names = image_names[idx]

    #TODO-elo: check if data are ints or floats here
    self.label = label[idx]  # dtype=int32
    self.label_f = self.label.astype(np.float32)  # dtype=float32

    attribute_eng_raw = annotation['attribute_eng']
    self.attribute_eng = [x[0] for x in attribute_eng_raw[0][0][:,0]]
    #print("  DBG self.attribute_eng =", self.attribute_eng)

    # dictionnary attribute_name -> attribute_index
    self.attribute_to_idx = { attr: idx for idx, attr in enumerate(ATTRIBUTES) }

    # write ground truth for all dataset images in one file
    if False:
      if train:
        # training dataset
        filename = 'training.dataset.ground_truth'
      else:
        # validation dataset
        filename = 'validation.dataset.ground_truth'
     
      print('Creating file \'' + filename + '\'')
      with open(filename, 'w') as f:
        # loop over dataset images
        assert self.image_names.shape[0] == self.label.shape[0]
        for image_name, gtlabel in zip(self.image_names, self.label):
          image_name = image_name[0][0] # array to string
          f.write(self.RAPdataset_path + '/' + image_name + '\n')
          assert len(self.attribute_eng) == gtlabel.shape[0]
          for attribute, value in zip(self.attribute_eng, gtlabel):
            f.write(attribute + ' ' + str(value) + '\n')
          f.write('\n\n')
    

  #############################
  def __len__(self):
    return len(self.image_names)


  #############################
  def __getitem__(self,idx):
    img_short_name = self.image_names[idx][0][0]
    img_name = os.path.join(self.RAPdataset_path, img_short_name)

    # CW hardcode image resizeing to standard size
    #print("  DBG reading image...", img_name)
    image = imread(img_name)
    elo_debug_input = False
    if elo_debug_input: #TODO-elo-rm-dbg
      #ipdb.set_trace()
      imsave('TMP/' + img_short_name, image)
    # image.dtype -> uint8, image.shape -> (200, 74, 3), np.amin(image) -> 0, np.amax(image) -> 253
    image = resize(image, (128,48), anti_aliasing=True, mode='constant')
    # image.dtype -> float64, image.shape -> (128, 48, 3), np.amin(image) -> 0.24, np.amax(image) -> 0.789
    # 
    image = 255 * image
    # image.dtype -> float64, image.shape -> (128, 48, 3), np.amin(image) -> 61.22, np.amax(image) -> 201.26
    image = image.astype(np.uint8)
    # image.dtype -> uint8, image.shape -> (128, 48, 3), np.amin(image) -> 61, np.amax(image) -> 201
    
    #imshow(image)
    #sleep(5)
    if elo_debug_input: #TODO-elo-rm-dbg
      imsave('TMP/' + img_short_name + '_1resized.png', image)

    if self.transform:
      image = self.transform(image)
      if elo_debug_input: #TODO-elo-rm-dbg
        #imshow(image)
        image_tmp = image.numpy() # Torch tensor to Numpy array
        image_tmp = self.normalize(image_tmp) # shape is (3,128,48)
        image_tmp = np.moveaxis(image_tmp, 0, -1) # shape is (128,48,3)
        imsave('TMP/' + img_short_name + '_2transformed.png', image_tmp)

    if elo_debug_input: #TODO-elo-rm-dbg
      self.write_attributes_to_file('TMP/' + img_short_name + '_3attributes.txt', self.label[idx]) 

    sample = {'image':image, 'label':self.label[idx], 'label_f':self.label_f[idx]}

    return sample


  #############################
  def write_attributes_to_file(self, filename, label):
    # label is a <class 'numpy.ndarray'> of shape (92,)
    # self.attribute_eng is a <class 'list'> of length 92 
    #ipdb.set_trace()
    assert label.shape[0] == len(self.attribute_eng)
    with open(filename, 'w') as f:
      for attribute, value in zip(self.attribute_eng, label):
        f.write(attribute + ' ' + str(value) + '\n')


  #############################
  def normalize(self, array):
      max_val = np.amax(array)
      min_val = np.amin(array)
      normalized_array = (array - min_val)/(max_val - min_val)
      return normalized_array


  #############################
  def index_of(self, attribute_name):
    """
    Return the index of the attribute in the attribute list.
    This index can be used to retrive attribute gound truth from
    the gound truth vector, and the attribute prediction from
    the prediction vector.
    """
    return self.attribute_to_idx[attribute_name]
    

#############################
if __name__ == '__main__':
  #simple test
  transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  data = RAPDataset(0,True,'/storage/Datasets/Rap-PedestrianAttributeRecognition/',transform)
  print('data[0] =', data[0])
  print('data[0] =')
  for attr in ATTRIBUTES:
    print(attr, data[0]['label'][data.index_of(attr)])

  # test index_of()
  print('idx   attribute')
  for idx, attr in enumerate(ATTRIBUTES):
    print('{0:3d}   {1}'.format(idx, attr))
    assert (data.index_of(attr) == idx)

  
