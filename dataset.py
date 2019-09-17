import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
import os
import torch
import torchvision
import numpy as np
from PIL import Image
import json
import random

def images(batch_size= 16,size = [448,448],data_path = None):
	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomResizedCrop(size=448,scale=(0.8,1.0)),
			# transforms.Resize(size=448),  # Let smaller edge match
			transforms.RandomHorizontalFlip(),
			# transforms.RandomCrop(size=448),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
		]),
		'val': transforms.Compose([
			transforms.Resize(size=(448,448)),
			# transforms.CenterCrop(size=448),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
		]),
	}
	images_path = data_path
	train_sets = torchvision.datasets.ImageFolder(os.path.join(images_path, 'train'), data_transforms['train'])
	train_loader = torch.utils.data.DataLoader(train_sets, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=False)
	val_sets = torchvision.datasets.ImageFolder(os.path.join(images_path, 'val'), data_transforms['val'])
	val_loader = torch.utils.data.DataLoader(val_sets, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=False)
	return train_loader,val_loader

# def test_loader(root, batch_size= 16,size = 224):
# 	preprocess = transforms.Compose([
# 		transforms.Resize(size),
# 		transforms.ToTensor(),
# 		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#
# 	val_datas = MyDataSet(root=root, transform=preprocess)
# 	val_loader = torch.utils.data.DataLoader(dataset=val_datas, batch_size=batch_size, shuffle=False, num_workers=8)
# 	return val_loader

# class IMG(Dataset):
# 	def __init__(self, file, root, transform = None, target_transform=None, n_triplets = 3):
# 		f = open(file,'r')
# 		Img_json = json.load(f)
# 		pics = []
# 		for img in Img_json:
# 			img_name = img['image_id']
# 			img_lable = img['disease_class']
# 			pics.append((img_name, img_lable))
# 		self.pics = pics
# 		self.transform = transform
# 		self.target_transform = target_transform
# 		self.root = root
# 		self.n_triplets = n_triplets
#
# 	def generate_triplets(self, index):
# 		triplets = ''
# 		# while(True):
# 		# 	pic = random.sample(self.pics,1)
# 		# 	# print(pic)
# 		# 	if pic[0][0] != self.pics[index][0] and pic[0][1] == self.pics[index][1]:
# 		# 		# print("pic",pic)
# 		# 		triplets.append(pic[0][0])
# 		# 		break
# 		while(True):
# 			pic = random.sample(self.pics,1)
# 			if pic[0][1] != self.pics[index]:
# 				triplets = pic[0]
# 				# triplets.append(pic[0][0])
# 				break
# 		return triplets
# 	def __getitem__(self, index):
# 		name,label = self.pics[index]
# 		t = index
# 		# name_p, name_n = self.generate_triplets(t)
# 		# name_n, label_n = self.generate_triplets(t)
# 		img = Image.open(self.root + name).convert('RGB')
# 		# img_p = Image.open(self.root + name_p).convert('RGB')
# 		# img_n = Image.open(self.root + name_n).convert('RGB')
# 		if self.transform is not None:
# 			img = self.transform(img)
# 			# img_p = self.transform(img_p)
# 			# img_n = self.transform(img_n)
# 		return img,label#[img, img_n], [label,label_n]#
#
# 	def __len__(self):
# 		return len(self.pics)

# def dataset(batch_size= 16):
# 	root_train = "F:\\yxb\\Crop-Disease-Detection-master\\data\\AgriculturalDisease_trainingset\\images\\"
# 	root_val = "F:\\yxb\\Crop-Disease-Detection-master\\data\\AgriculturalDisease_validationset\\images\\"
# 	train_json = "F:\\yxb\\Crop-Disease-Detection-master\\data\\AgriculturalDisease_trainingset\\AgriculturalDisease_train_annotations.json"
# 	val_json = "F:\\yxb\\Crop-Disease-Detection-master\\data\\AgriculturalDisease_validationset\\AgriculturalDisease_validation_annotations.json"
# 	# root_train = "F:\\yxb\\Crop-Disease-Detection-master\\data\\train_disease\\"
# 	# root_val = "F:\\yxb\\Crop-Disease-Detection-master\\data\\val_disease\\"
# 	# train_json = "F:\\yxb\\Crop-Disease-Detection-master\\data\\train_test_labels.json"
# 	# val_json = "F:\\yxb\\Crop-Disease-Detection-master\\data\\val_test_labels.json"
# 	data_transforms = {
# 		'train': transforms.Compose([
# 			transforms.Resize(384),
# 			transforms.RandomResizedCrop(299),
# 			transforms.RandomHorizontalFlip(),
# 			transforms.ToTensor(),
# 			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# 		]),
# 		'val': transforms.Compose([
# 			transforms.Resize(299),
# 			transforms.CenterCrop(299),
# 			transforms.ToTensor(),
# 			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# 		]),
# 	}
# 	train_data = IMG(file = train_json, root = root_train, transform = data_transforms['train'])
# 	train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True,num_workers=8)
# 	val_data = IMG(file = val_json, root = root_val, transform = data_transforms['val'])
# 	val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False,num_workers=8)
# 	return train_loader, val_loader

# class read_image(Dataset):
# 	def __init__(self, file, root, transform=None):
# 		f = open(file)
# 		file_list = f.readlines()
# 		images = []
# 		for file in file_list:
# 			file_name,file_lable = file.split()
# 			images.append((file_name,int(file_lable)))
# 		self.images = images
# 		self.transform = transform
# 		self.root = root
# 	def __getitem__(self, index):
# 		image_name,label = self.images[index]
# 		img = Image.open(self.root + image_name).convert('RGB')
# 		if self.transform is not None:
# 			img = self.transform(img)
# 		return img,label
#
# 	def __len__(self):
# 		return len(self.images)

# def datasetloader(batch_size= 64,size = [256,224]):
# 	train_root = 'D:\\cygwin64\\home\\user\\tianqijun\\data\\Stanford Cars\\train\\'
# 	val_root = 'D:\\cygwin64\\home\\user\\tianqijun\\data\\Stanford Cars\\val\\'
# 	train_lable = 'D:\\cygwin64\\home\\user\\tianqijun\\data\\Stanford Cars\\train.txt'
# 	val_lable = 'D:\\cygwin64\\home\\user\\tianqijun\\data\\Stanford Cars\\test.txt'
# 	data_transforms = {
# 		'train': transforms.Compose([
# 			transforms.Resize(size[0]),
# 			transforms.RandomResizedCrop(size[1]),
# 			transforms.RandomHorizontalFlip(),
# 			transforms.ToTensor(),
# 			transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# 		]),
# 		'val': transforms.Compose([
# 			transforms.Resize(size[1]),
# 			transforms.CenterCrop(size[1]),
# 			transforms.ToTensor(),
# 			transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# 		]),
# 	}
#
# 	train_datas = read_image(file=train_lable, root=train_root, transform=data_transforms['train'])
# 	train_loader = torch.utils.data.DataLoader(dataset=train_datas, batch_size=batch_size, shuffle=True, num_workers=8)
#
# 	val_datas = read_image(file=val_lable, root=val_root, transform=data_transforms['val'])
# 	val_loader = torch.utils.data.DataLoader(dataset=val_datas, batch_size=batch_size, shuffle=False, num_workers=8)
# 	return train_loader, val_loader


