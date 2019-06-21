import numpy as np
import scipy.io as sio
from scipy.misc import imread
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class RAPDataset(Dataset):
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
		self.label = label[idx]
		

	def __len__(self):
		return len(self.image_names)

	def __getitem__(self,idx):
		img_name = os.path.join(self.RAPdataset_path,self.image_names[idx][0][0])
		image = imread(img_name)		
		
		if self.transform:
                        image = self.transform(image)
		sample = {'image':image, 'label':self.label[idx]}
		return sample


if __name__ == '__main__':
	#simple test
	transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	data = RAPDataset(0,True,'/storage/Datasets/Rap-PedestrianAttributeRecognition/',transform)
	print data[0]

	


