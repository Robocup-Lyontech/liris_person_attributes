import sys
#from torch.utils.data import Dataset #, DataLoader
import pathlib
import random
from PIL import Image  # Pillow package

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from scipy.misc import imread, imsave, imshow
from skimage.transform import rescale, resize
import numpy as np


#############################

"""
Full PETA attributes list (106):

accessoryFaceMask accessoryHairBand accessoryHat accessoryHeadphone accessoryKerchief accessoryMuffler accessoryNothing accessoryShawl accessorySunglasses carryingBabyBuggy carryingBackpack carryingFolder carryingLuggageCase carryingMessengerBag carryingNothing carryingOther carryingPlasticBags carryingShoppingTro carryingSuitcase carryingUmbrella footwearBlack footwearBlue footwearBoots footwearBrown footwearGreen footwearGrey footwearLeatherShoes footwearOrange footwearPink footwearPurple footwearRed footwearSandals footwearShoes footwearSneakers footwearStocking footwearWhite footwearYellow hairBald hairBlack hairBrown hairGreen hairGrey hairLong hairOrange hairPurple hairRed hairShort hairWhite hairYellow lowerBodyBlack lowerBodyBlue lowerBodyBrown lowerBodyCapri lowerBodyCasual lowerBodyFormal lowerBodyGreen lowerBodyGrey lowerBodyHotPants lowerBodyJeans lowerBodyLogo lowerBodyLongSkirt lowerBodyOrange lowerBodyPink lowerBodyPlaid lowerBodyPurple lowerBodyRed lowerBodyShorts lowerBodyShortSkirt lowerBodySuits lowerBodyThinStripes lowerBodyTrousers lowerBodyWhite lowerBodyYellow personalFemale personalLarger60 personalLess15 personalLess30 personalLess45 personalLess60 personalMale upperBodyBlack upperBodyBlue upperBodyBrown upperBodyCasual upperBodyFormal upperBodyGreen upperBodyGrey upperBodyJacket upperBodyLogo upperBodyLongSleeve upperBodyNoSleeve upperBodyOrange upperBodyOther upperBodyPink upperBodyPlaid upperBodyPurple upperBodyRed upperBodyShortSleeve upperBodySuit upperBodySweater upperBodyThickStripes upperBodyThinStripes upperBodyTshirt upperBodyVNeck upperBodyWhite upperBodyYellow
"""

# full attributes list minus some irrelevant ones
ATTRIBUTES = ['accessoryHairBand', 'accessoryHat', 'accessoryHeadphone', 'accessoryKerchief', 'accessoryMuffler', 'accessoryNothing', 'accessorySunglasses', 'carryingBabyBuggy', 'carryingBackpack', 'carryingFolder', 'carryingLuggageCase', 'carryingMessengerBag', 'carryingNothing', 'carryingOther', 'carryingPlasticBags', 'carryingShoppingTro', 'carryingSuitcase', 'carryingUmbrella', 'footwearBlack', 'footwearBlue', 'footwearBoots', 'footwearBrown', 'footwearGreen', 'footwearGrey', 'footwearLeatherShoes', 'footwearOrange', 'footwearPink', 'footwearPurple', 'footwearRed', 'footwearSandals', 'footwearShoes', 'footwearSneakers', 'footwearStocking', 'footwearWhite', 'footwearYellow', 'hairBald', 'hairBlack', 'hairBrown', 'hairGreen', 'hairGrey', 'hairLong', 'hairOrange', 'hairPurple', 'hairRed', 'hairShort', 'hairWhite', 'hairYellow', 'lowerBodyBlack', 'lowerBodyBlue', 'lowerBodyBrown', 'lowerBodyCapri', 'lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyGreen', 'lowerBodyGrey', 'lowerBodyHotPants', 'lowerBodyJeans', 'lowerBodyLogo', 'lowerBodyLongSkirt', 'lowerBodyOrange', 'lowerBodyPink', 'lowerBodyPlaid', 'lowerBodyPurple', 'lowerBodyRed', 'lowerBodyShorts', 'lowerBodyShortSkirt', 'lowerBodySuits', 'lowerBodyThinStripes', 'lowerBodyTrousers', 'lowerBodyWhite', 'lowerBodyYellow', 'personalFemale', 'personalLarger60', 'personalLess15', 'personalLess30', 'personalLess45', 'personalLess60', 'personalMale', 'upperBodyBlack', 'upperBodyBlue', 'upperBodyBrown', 'upperBodyCasual', 'upperBodyFormal', 'upperBodyGreen', 'upperBodyGrey', 'upperBodyJacket', 'upperBodyLogo', 'upperBodyLongSleeve', 'upperBodyNoSleeve', 'upperBodyOrange', 'upperBodyOther', 'upperBodyPink', 'upperBodyPlaid', 'upperBodyPurple', 'upperBodyRed', 'upperBodyShortSleeve', 'upperBodySuit', 'upperBodySweater', 'upperBodyThickStripes', 'upperBodyThinStripes', 'upperBodyTshirt', 'upperBodyVNeck', 'upperBodyWhite', 'upperBodyYellow']
# 104 attributes

#PETA_subsets = [ '3DPeS', 'CAVIAR4REID', 'CUHK', 'GRID', 'i-LID', 'MIT', 'PRID', 'SARC3D', 'TownCentre', 'VIPeR' ]


#############################
class PETADataset(Dataset):
  """PEdesTrian Attribute (PETA) dataset."""


  #############################
  def __init__(self, train, root_dir, transform=None):
    """
    Args:
        root_dir (string): Root directory of the dataset with all its subsets.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    print('PETADataset root_dir =', root_dir)

    self.root_dir = root_dir
    self.transform = transform

    # load image list
    img_patterns = [ '*.bmp', '*.png', '*.jpg', '*.jpeg' ]
    self.image_files = []
    for pattern in img_patterns:
      self.image_files += \
          [ f for f in pathlib.Path(self.root_dir).rglob(pattern) ]
    assert (len(self.image_files) == 19000), "PETA dataset: incorrect total image number"

    # select training/validation images
    # validation = one in ten images
    validation_image_files = self.image_files[::10]
    # training = 9 in ten images
    training_image_files = [item for item in self.image_files if item not in validation_image_files]

    if train:
      # PETA train subdataset
       self.image_files = training_image_files
       assert (len(self.image_files) == 19000-1900), "PETA dataset: incorrect training image number"
    else:
      # PETA validation subdataset
       self.image_files = validation_image_files
       assert (len(self.image_files) == 1900), "PETA dataset: incorrect validation image number"

    # build attribute -> index map
    self.attribute_to_idx = { attr: idx for idx, attr in enumerate(ATTRIBUTES) }

    # load ground truth

    # build person label dictionnary, because there can be several images
    # of the same person, with the same attributes
    person_labels = { }
    for label_file in pathlib.Path(self.root_dir).rglob('Label.txt'):
      # read label file
      subset = label_file.parent.parent.name
      print('processing', label_file, 'of subset', subset, '...')
      labels = label_file.read_text().splitlines()
      labels = [ line.split() for line in labels ]
      # labels[0] == '0709 upperBodyBlack lowerBodyBlue hairBrown footwearBlack lowerBodyCasual lowerBodyJeans personalLess45 personalMale upperBodyFormal upperBodyLongSleeve upperBodySuit hairShort footwearLeatherShoes carryingSuitcase accessoryNothing'
      # labels[1] == '0059 upperBodyBlack lowerBodyBlack hairBrown footwearBlack lowerBodyFormal lowerBodyTrousers personalFemale personalLess45 upperBodyFormal upperBodyLongSleeve upperBodyOther hairLong footwearLeatherShoes carryingOther carryingPlasticBags accessoryNothing'
      subset_labels = \
          { label[0]: self.attribute_list_to_vect(label[1:]) for label in labels }
      # subset_labels['0709'] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, ...]
      # subset_labels['0059'] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, ...]
      person_labels[subset] = subset_labels

      # check
      if False:
        # print 1st label line and its vector version
        print(labels[0])
        first_person_id = labels[0][0]
        self.print_attributes_vect(person_labels[subset][first_person_id])

    # associate one attribute vector per image
    self.ground_truth = []
    for img_filename in self.image_files:
      subset = img_filename.parent.parent.name
      # retrieve person id from image name
      if subset == 'CUHK':
        # person id = image filename
        person_id = img_filename.name
      else:
        # person id = xxx_...
        person_id = img_filename.name.split('_')[0]
      if False:
        print('img_filename=', img_filename)
        print('person_id=', person_id)
      # set image ground truth
      self.ground_truth.append(person_labels[subset][person_id])

    # check there is as much ground_truth data as images
    assert len(self.ground_truth) == len(self.image_files)

    # other consistency check
    if False:
      cnt = 0
      cnt_male_equals_female = 0
      for gt_vect in self.ground_truth:
        # check that 'personalFemale' or 'personalMale' is set to 1
        personalFemale_idx = self.attribute_to_idx['personalFemale']
        personalMale_idx = self.attribute_to_idx['personalMale']
        cnt += 1
        print('cnt=', cnt, 'female=', gt_vect[personalFemale_idx], 'male=', gt_vect[personalMale_idx])
        assert ((gt_vect[personalFemale_idx] + gt_vect[personalMale_idx]) <= 1)
        if gt_vect[personalFemale_idx] == gt_vect[personalMale_idx]:
          cnt_male_equals_female += 1
      print('cnt_male_equals_female=', cnt_male_equals_female)
      # note: there is 1 image with female=0 and male=0


  #############################
  def __len__(self):
    return len(self.image_files)


  #############################
  def __getitem__(self, idx):
    # get image file name
    img_name = self.image_files[idx]

    # load image
    # hardcode image resizeing to standard size
    image = imread(img_name)

    # resize image and convert to uint8
    # image.dtype -> uint8, image.shape -> (200, 74, 3), np.amin(image) -> 0, np.amax(image) -> 253
    image = resize(image, (128,48), anti_aliasing=True, mode='constant')
    # image.dtype -> float64, image.shape -> (128, 48, 3), np.amin(image) -> 0.24, np.amax(image) -> 0.789
    image = 255 * image
    # image.dtype -> float64, image.shape -> (128, 48, 3), np.amin(image) -> 61.22, np.amax(image) -> 201.26
    image = image.astype(np.uint8)
    # image.dtype -> uint8, image.shape -> (128, 48, 3), np.amin(image) -> 61, np.amax(image) -> 201

    # pre-process image
    if self.transform:
      image = self.transform(image)

    # get ground truth and convert to numpy array
    gt_vect = self.ground_truth[idx]
    gt_vect_i = np.array(gt_vect, np.int32)  # shape==(104,)  dtype==int32
    gt_vect_f = gt_vect_i.astype(np.float32) # shape==(104,)  dtype==float32

    sample = {'image':image, 'label':gt_vect_i, 'label_f':gt_vect_f}

    return sample


  #############################
  # convert attributes sparse list to a dense vector with 1 for
  # an ON attribute and 0 for an OFF attribute
  def attribute_list_to_vect(self, attr_list):
    v = [0] * len(ATTRIBUTES)
    for a in attr_list:
      try:
        v[self.attribute_to_idx[a]] = 1
      except:
        pass
    return v


  #############################
  # print one attribute dense vector to console
  def print_attributes_vect(self, attr_vect):
    for attr, value in zip(ATTRIBUTES, attr_vect):
      print(attr, value)


  #############################
  # print one attribute dense vector to console (sorted output)
  def print_sorted_attributes_vect(self, attr_vect):
    # build unordered list of strings 'attribute value'
    attr_val = []
    for attr, value in zip(ATTRIBUTES, attr_vect):
      attr_val.append(attr + ' ' + str(value))
    # sort 'attribute value' list
    attr_val.sort()
    # display sorted list
    print('\n'.join(attr_val))


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
# test
if __name__ == '__main__':
  """
  Test image and ground truth loading:
  - load dataset image list
  - load ground truth
  - display 10 random images + ground truth
  """
  peta_root_dir = '/storage/Datasets/PETA-PEdesTrianAttribute'

  if len(sys.argv) == 2:
    if sys.argv[1] == '-h':
      print("Usage:", sys.argv[0], "[-h | PETA_directory]")
      sys.exit(1)
    else:
      peta_root_dir = sys.argv[1]

  # load dataset
  transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  loader = PETADataset(True, peta_root_dir, transform)
  print('len(loader.image_files) =', len(loader.image_files))
  print('loader[0] =', loader[0])
  print('loader[0] =')
  for attr in ATTRIBUTES:
    print(attr, loader[0]['label'][loader.index_of(attr)])
  #print('loader.attribute_to_idx =', loader.attribute_to_idx)

  # test index_of()
  print('idx   attribute')
  for idx, attr in enumerate(ATTRIBUTES):
    print('{0:3d}   {1}'.format(idx, attr))
    assert (loader.index_of(attr) == idx)

  # prepare data samples
  indices = list(range(len(loader.image_files)))
  #random.shuffle(indices)

  # print 20 random images name
  if False:
    print('---------------------')
    for img_id in indices[:20]:
      print(loader.image_files[img_id])
    print('---------------------')

  # display 10 random image and ground truth
  for img_id in indices[:10]:
    print('---------------------')
    print(loader.image_files[img_id])
    # display image
    img = Image.open(loader.image_files[img_id])
    img.show('real size')
    img_big = img.resize((4*img.size[0], 4*img.size[1]))
    img_big.show('big size')
    # display ground truth
    print(loader.ground_truth[img_id])
    loader.print_sorted_attributes_vect(loader.ground_truth[img_id])
    # pause
    input('Enter to continue...')


