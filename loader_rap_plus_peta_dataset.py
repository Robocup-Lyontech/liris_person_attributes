#
# Combined RAP+PETA dataset
#

from . import loader_rapdataset_yiqiang
from . import loader_peta_dataset

from torch.utils.data import Dataset, DataLoader
import numpy as np


#############################

ATTRIBUTES = [('Female','personalFemale'), ('AgeLess16','personalLess15'), ('Age17-30','personalLess30'), ('Age31-45','personalLess45'), ('hs-BaldHead','hairBald'), ('hs-LongHair','hairLong'), ('hs-BlackHair','hairBlack'), ('hs-Hat','accessoryHat'), ('hs-Muffler','accessoryMuffler'), ('ub-Sweater','upperBodySweater'), ('ub-TShirt','upperBodyTshirt'), ('ub-Jacket','upperBodyJacket'), ('ub-SuitUp','upperBodySuit'), ('ub-ShortSleeve','upperBodyShortSleeve'), ('lb-LongTrousers','lowerBodyTrousers'), ('lb-Skirt','lowerBodyLongSkirt'), ('lb-ShortSkirt','lowerBodyShortSkirt'), ('lb-Jeans','lowerBodyJeans'), ('shoes-Leather','footwearLeatherShoes'), ('shoes-Boots','footwearBoots'), ('attach-Backpack','carryingBackpack'), ('attach-SingleShoulderBag','carryingMessengerBag'), ('attach-PlasticBag','carryingPlasticBags'), ('up-Black','upperBodyBlack'), ('up-White','upperBodyWhite'), ('up-Gray','upperBodyGrey'), ('up-Red','upperBodyRed'), ('up-Green','upperBodyGreen'), ('up-Blue','upperBodyBlue'), ('up-Yellow','upperBodyYellow'), ('up-Brown','upperBodyBrown'), ('up-Purple','upperBodyPurple'), ('up-Pink','upperBodyPink'), ('up-Orange','upperBodyOrange'), ('low-Black','lowerBodyBlack'), ('low-White','lowerBodyWhite'), ('low-Gray','lowerBodyGrey'), ('low-Red','lowerBodyRed'), ('low-Green','lowerBodyGreen'), ('low-Blue','lowerBodyBlue'), ('low-Yellow','lowerBodyYellow'), ('shoes-Black','footwearBlack'), ('shoes-White','footwearWhite'), ('shoes-Gray','footwearGrey'), ('shoes-Red','footwearRed'), ('shoes-Green','footwearGreen'), ('shoes-Blue','footwearBlue'), ('shoes-Yellow','footwearYellow'), ('shoes-Brown','footwearBrown')]
# 49 attributes

#############################
class RAPPlusPETADataset(Dataset):
  """
  Richly Annotated Pedestrian (RAP) dataset
  and
  PEdesTrian Attribute (PETA) dataset.
  """


  #############################
  def __init__(self, train, rap_root_dir, peta_root_dir, transform=None):
    """
    Args:
        train (boolean): true to get the training part of the datasets,
                         false to get the validation part.
        rap_root_dir (string): root directory of the RAP dataset.
        peta_root_dir (string): root directory of the PETA dataset with
                                all its subsets.
        transform (callable, optional): optional transform to be applied
                                        on a sample.
    """
    self.train = train
    self.rap_dataset = None
    self.peta_dataset = None
    if train:
      # training dataset
      # training = 33268 RAP + 17100 PETA images
      #          = 50368 images
      #          = 1007 batches of 50 images + 1 batch of 18 images
      self.rap_dataset = loader_rapdataset_yiqiang.RAPDataset(0, True, rap_root_dir, transform)
      self.peta_dataset = loader_peta_dataset.PETADataset(True, peta_root_dir, transform)
    else:
      # validation dataset
      # validation = 8317 RAP + 1900 PETA images
      #            = 10217 images
      #            = 204 batches of 50 images + 1 batch of 17 images
      self.rap_dataset = loader_rapdataset_yiqiang.RAPDataset(0, False, rap_root_dir, transform)
      self.peta_dataset = loader_peta_dataset.PETADataset(False, peta_root_dir, transform)


  #############################
  def __len__(self):
    total_len = self.rap_dataset.__len__() + self.peta_dataset.__len__()
    return total_len


  #############################
  def __getitem__(self, idx):
    # note: idx is in range [0, __len__()-1]
    rap_flag = False   # is item from RAP?
    peta_flag = False  # is item from PETA?

    # get item from RAP or PETA
    ds_name,raw_item = self.get_raw_item(idx)
    assert (ds_name in ['rap', 'peta'])

    # keep only RAP-PETA common attributes
    raw_gt_vect_i = raw_item['label']
    raw_gt_vect_f = raw_item['label_f']
    gt_vect_i = np.zeros((len(ATTRIBUTES),), raw_gt_vect_i.dtype)
    gt_vect_f = np.zeros((len(ATTRIBUTES),), raw_gt_vect_f.dtype)

    if ds_name == 'rap':
      # data from RAP dataset
      for i, attr_names in enumerate(ATTRIBUTES):
        # attr_names is a pair (rap_name, peta_name)
        gt_vect_i[i] = raw_gt_vect_i[self.rap_dataset.index_of(attr_names[0])]
        gt_vect_f[i] = raw_gt_vect_f[self.rap_dataset.index_of(attr_names[0])]
    else:
      # data from PETA dataset
      for i, attr_names in enumerate(ATTRIBUTES):
        # attr_names is a pair (rap_name, peta_name)
        gt_vect_i[i] = raw_gt_vect_i[self.peta_dataset.index_of(attr_names[1])]
        gt_vect_f[i] = raw_gt_vect_f[self.peta_dataset.index_of(attr_names[1])]

    # build sample dict
    image = raw_item['image']
    sample = {'image':image, 'label':gt_vect_i, 'label_f':gt_vect_f}
    return sample


  #############################
  def get_raw_item(self, idx):
    """
    Returns a couple (dataset_name, sample)
    """
    # note: idx is in range [0, __len__()-1]

    # idx  < rap_dataset.__len__() -> RAP
    # idx >= rap_dataset.__len__() -> PETA
    rap_len = self.rap_dataset.__len__()
    if idx < rap_len:
      return 'rap',self.rap_dataset.__getitem__(idx)
    else:
      return 'peta',self.peta_dataset.__getitem__(idx-rap_len)


#############################
#
# test
#
if __name__ == '__main__':
  """
  Test image and ground truth loading:
  - load dataset image list
  - load ground truth
  - display 10 random images + ground truth
  """
  #transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  transform = None

  # test get_raw_item()
  dataset_train = RAPPlusPETADataset(True, '/storage/Datasets/Rap-PedestrianAttributeRecognition/',  '/storage/Datasets/PETA-PEdesTrianAttribute', transform)
  print("RAPPlusPETADataset train size =", dataset_train.__len__())
  for idx in [0, 33268-1, 33268, 33268+17100-1]:
    print('idx =', idx)
    ds_name,item = dataset_train.get_raw_item(idx)
    label_i = item['label']
    label_f = item['label_f']
    if idx < 33268:
      assert (ds_name == 'rap')
      assert (label_i.shape == (92,))
      assert (label_f.shape == (92,))
    else:
      assert (ds_name == 'peta')
      assert (label_i.shape == (104,))
      assert (label_f.shape == (104,))

  dataset_valid = RAPPlusPETADataset(False, '/storage/Datasets/Rap-PedestrianAttributeRecognition/',  '/storage/Datasets/PETA-PEdesTrianAttribute', transform)
  print("RAPPlusPETADataset valid size =", dataset_valid.__len__())
  for idx in [0, 8317-1, 8317, 8317+1900-1]:
    print('idx =', idx)
    ds_name,item = dataset_valid.get_raw_item(idx)
    label_i = item['label']
    label_f = item['label_f']
    if idx < 8317:
      assert (ds_name == 'rap')
      assert (label_i.shape == (92,))
      assert (label_f.shape == (92,))
    else:
      assert (ds_name == 'peta')
      assert (label_i.shape == (104,))
      assert (label_f.shape == (104,))
 
  # test __getitem__() #1
  sample = dataset_train.__getitem__(0)
  assert (sample['label'].tolist() == [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
  assert (sample['label_f'].tolist() == [0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.])

  # test __getitem__() #2
  sample = dataset_train.__getitem__(33268)
  assert (sample['label'].tolist() == [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
  assert (sample['label_f'].tolist() == [0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0])

  # test __getitem__() #3
  ds_name,raw_sample = dataset_train.get_raw_item(42)
  sample = dataset_train.__getitem__(42)
  for i,attr_names in enumerate(ATTRIBUTES):
    assert (sample['label'][i] == raw_sample['label'][dataset_train.rap_dataset.index_of(attr_names[0])])
    assert (sample['label_f'][i] == raw_sample['label_f'][dataset_train.rap_dataset.index_of(attr_names[0])])

  # test __getitem__() #4
  ds_name,raw_sample = dataset_train.get_raw_item(33268+42)
  sample = dataset_train.__getitem__(33268+42)
  for i,attr_names in enumerate(ATTRIBUTES):
    assert (sample['label'][i] == raw_sample['label'][dataset_train.peta_dataset.index_of(attr_names[1])])
    assert (sample['label_f'][i] == raw_sample['label_f'][dataset_train.peta_dataset.index_of(attr_names[1])])




