import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms, datasets


import matplotlib.pyplot as plt
import time
import argparse


cuda = torch.cuda.is_available()
num_workers = 8 if cuda else 0 
device = torch.device("cuda" if cuda else "cpu")

class CenterLoss(nn.Module):
	"""
	Args:
		num_classes (int): number of classes.
		feat_dim (int): feature dimension.
	"""
	def __init__(self, num_classes, feat_dim, device=torch.device('cpu')):
		super(CenterLoss, self).__init__()
		self.num_classes = num_classes
		self.feat_dim = feat_dim
		self.device = device
		
		self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

	def forward(self, x, labels):
		"""
		Args:
			x: feature matrix with shape (batch_size, feat_dim).
			labels: ground truth labels with shape (batch_size).
		"""
		batch_size = x.size(0)
		distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
				  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
		distmat.addmm_(1, -2, x, self.centers.t())

		classes = torch.arange(self.num_classes).long().to(self.device)
		labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
		mask = labels.eq(classes.expand(batch_size, self.num_classes))

		dist = []
		for i in range(batch_size):
			value = distmat[i][mask[i]]
			value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
			dist.append(value)
		dist = torch.cat(dist)
		loss = dist.mean()

		return loss

class VeriDataset(Dataset):
	def __init__(self, file_list):
		self.file_list = file_list


	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, index):
		l_img = Image.open('../validation_verification/' + self.file_list[index][0])
		r_img = Image.open('../validation_verification/' + self.file_list[index][1])


		l_img = transforms.ToTensor()(l_img)
		r_img = transforms.ToTensor()(r_img)
		label = self.file_list[index][2]
	
		return l_img, r_img, int(label)#.long()


class TestVeriDataset(Dataset):
	def __init__(self, file_list):
		self.file_list = file_list


	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, index):
		l_img = Image.open('../test_verification/' + self.file_list[index][0])
		r_img = Image.open('../test_verification/' + self.file_list[index][1])

		l_img = transforms.ToTensor()(l_img)
		r_img = transforms.ToTensor()(r_img)
		
	
		return l_img, r_img, self.file_list[index]

class face_classifier(nn.Module):
	def __init__(self, feat_dim=10):
		super(face_classifier, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, 5, stride=1, padding=2)
		self.conv1_bn = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 192, 5, stride=1, padding=2)
		self.conv2_bn = nn.BatchNorm2d(192)
		self.conv3 = nn.Conv2d(192, 384, 3, stride=1, padding=1)
		self.conv3_bn = nn.BatchNorm2d(384)
		self.conv4 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
		self.conv4_bn = nn.BatchNorm2d(256)
		self.conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
		self.conv5_bn = nn.BatchNorm2d(256)
		self.pool = nn.MaxPool2d(3, stride=2, padding=1) 
		self.fc1 = nn.Linear(4096, 4096)
		self.fc1_bn = nn.BatchNorm1d(4096)
		self.fc2 = nn.Linear(4096, 4096)
		self.fc2_bn = nn.BatchNorm1d(4096)
		self.fc3 = nn.Linear(4096, 2300)
		self.linear_closs = nn.Linear(4096, 1024, bias=False)

		
		# For creating the embedding to be passed into the Center Loss criterion
		#self.linear_closs = nn.Linear(4096, feat_dim, bias=False)
		self.relu_closs = nn.ReLU(inplace=True)
		

	def forward(self, x):
		x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
		x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
		x = F.relu(self.conv3_bn(self.conv3(x)))
		x = F.relu(self.conv4_bn(self.conv4(x)))
		x = self.pool(F.relu(self.conv5_bn(self.conv5(x))))
		#print("Debug:", x.shape)
		#print("Debug:", x.view(-1, 4096).shape)
		x = x.view(-1, 4096)
		
		x = F.relu(self.fc1_bn(self.fc1(x)))
		x = F.relu(self.fc2_bn(self.fc2(x)))
		label_output = self.fc3(x)

		
		# Create the feature embedding for the Center Loss
		closs_output = self.linear_closs(x)
		closs_output = self.relu_closs(closs_output)

		return closs_output, label_output

from PIL import Image

class Test_Class_Dataset(Dataset):
	def __init__(self, X,transform=None):
		self.transform = transform
		self.sum = [0]
		img_list = []
		self.Y = []
		count = 0
		for img_name in X:
			#if count < 6:
			img = Image.open('../test_classification/medium/' + img_name)
			
			img = np.transpose(np.array(img.getdata()).reshape(32,32,3), (2, 0, 1))
			img_list.append(img/256)
			self.Y.append(img_name)
			
		print(len(img_list))
		self.X = np.asarray(img_list)
		self.Y = np.asarray(self.Y)
		
	def __len__(self):
		return len(self.X)

	def __getitem__(self,index):
		X = self.X[index]
		return torch.Tensor(X).float(),self.Y[index]


def train_epoch(model, train_loader, criterion, optimizer , mode, criterion_closs, optimizer_closs):
	model.train()

	running_loss = 0.0
	running_closs = 0.0
	running_lloss = 0.0


	start_time = time.time()

	for batch_idx, (data, target) in enumerate(train_loader, 0):
		optimizer.zero_grad() 
		#if mode == 'v':
		optimizer_closs.zero_grad()
		
		data = data.to(device)
		target = target.to(device)

		# forward + backward + optimize
		feature, outputs = model(data)
		l_loss = criterion(outputs, target)
		#if mode == 'v':
		c_loss = criterion_closs(feature, target.long())
		closs_weight = 1
		loss = l_loss + closs_weight * c_loss

		loss.backward()
		optimizer.step()

		# by doing so, weight_cent would not impact on the learning of centers

		for param in criterion_closs.parameters():
			param.grad.data *= (1. / closs_weight)
		optimizer_closs.step()

		running_loss += loss.item()

		
		running_lloss += l_loss.item()
		running_closs += c_loss.item()

		if batch_idx % 2000 == 0:
			print("CheckLoss:", running_loss)
			
		torch.cuda.empty_cache()
		del target
		del feature
		del loss
	end_time = time.time()
	
	running_loss /= len(train_loader)
	#if mode == 'c':
	print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
	#elif mode == 'v':
	print('Training Loss: ', running_loss, 'L_Loss: ', running_lloss,'C_Loss: ', running_closs, 'Time: ',end_time - start_time, 's')
	return running_loss

def test_model(model, test_loader, criterion, criterion_closs):
	with torch.no_grad():
		model.eval()

		running_loss = 0.0
		running_closs = 0.0
		running_lloss = 0.0
		total_predictions = 0.0
		correct_predictions = 0.0

		for batch_idx, (data, target) in enumerate(test_loader):   
			data = data.to(device)
			target = target.to(device)
			

			feat, outputs = model(data)

			_, predicted = torch.max(F.softmax(outputs, dim=1), 1)
			total_predictions += target.size(0)
			correct_predictions += (predicted == target).sum().item()
			l_loss = criterion(outputs, target.long())
			c_loss = criterion_closs(feat, target.long())
			loss = l_loss + 1 * c_loss
			running_closs += c_loss.item()
			running_lloss += l_loss.item()
			running_loss += loss.item()
			

		running_loss /= len(test_loader)
		running_closs /= len(test_loader)
		running_lloss /= len(test_loader)
		acc = (correct_predictions/total_predictions)*100.0
		
		print('Testing Loss: ', running_loss)
		print('Testing L_Loss: ', running_lloss)
		print('Testing C_Loss: ', running_closs)
		print('Testing Accuracy: ', acc, '%')
		return running_loss, acc

def test_model_kaggle(model, testK_loader, criterion,idx_map):
	with torch.no_grad():
		model.eval()

		
		result ={'Id':[],'Category':[]}
		for batch_idx, (data,name) in enumerate(testK_loader):   
			data = data.to(device)
			outputs = model(data)
			_, predicted = torch.max(outputs.data, 1)
			result['Id'].append(str(5000+batch_idx) + '.jpg')
			result['Category'].append(idx_map[predicted.cpu().numpy()[0]])

			if batch_idx % 10000 == 0:
				print(batch_idx)
				
		return result




def learning_rate_decay(optim,epoch,lr):
	lr = lr * (0.1 ** (epoch//30))
	for param_group in optim.param_groups:
		param_group['lr'] = lr


def main(args):
	device = torch.device("cuda" if cuda else "cpu")
	if not args.test:

		train_data = datasets.ImageFolder(root = '../train_data/medium/',transform=transforms.ToTensor())

		valid_data = datasets.ImageFolder(root = '../validation_classification/medium/',transform=transforms.ToTensor())

		train_loader_args = dict(shuffle=True, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda\
							else dict(shuffle=True, batch_size=64)
		train_loader = DataLoader(train_data, **train_loader_args) 
		print("Training Data Size: ", len(train_data))
		print("Validation Data Size: ", len(valid_data))

		valid_loader_args = dict(shuffle=True, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda\
							else dict(shuffle=True, batch_size=64)
		valid_loader = DataLoader(valid_data,  **valid_loader_args) 

		model = face_classifier()

		start_epoch	= 0
		
		closs_weight = 0.5
		lr_cent = 0.5
		feat_dim = 4096
		lr = 1e-2
		criterion = nn.CrossEntropyLoss()
		criterion_closs = CenterLoss(2300, 1024, device)
		print(type(criterion_closs))
		optimizer = optim.Adam(model.parameters())
		optimizer_closs = torch.optim.Adam(criterion_closs.parameters())
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
		scheduler_closs = optim.lr_scheduler.ReduceLROnPlateau(optimizer_closs, 'min')

		print(device)
		model = model.to(device)
		print(model)


		n_epochs = args.n_epoch
		Train_loss = []
		Test_loss = []
		Test_acc = []

		for i in range(start_epoch, n_epochs):
			print("Epochs: ", i)
			learning_rate_decay(optimizer, i, lr)
			train_loss = train_epoch(model, train_loader, criterion, optimizer, 'v',criterion_closs, optimizer_closs)
			test_loss, test_acc = test_model(model, valid_loader, criterion, criterion_closs)
			Train_loss.append(train_loss)
			Test_loss.append(test_loss)
			Test_acc.append(test_acc)
			print('='*20)
			scheduler.step(test_loss)
			scheduler_closs.step(test_loss)
			torch.save({
					'epoch': n_epochs + 1, 
					'state_dict': model.state_dict(),
					'optimizer' : optimizer.state_dict(),
				}, '%s/0506 model_epoch_%d.pth' % ('../checkpoints', i))




	else:
		
		if args.mode == "classification":

			x = []
			with open('../test_order_classification.txt','r') as test_list:
				for img in test_list:
					x.append(img.replace('\n',''))
					
			Test = Test_Class_Dataset(x)

			idx_map = sorted([str(i) for i in range(2300)])
			model = face_classifier()
			criterion = nn.CrossEntropyLoss()
			optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
			

			checkpoint = torch.load('../checkpoints/0213 model_epoch_12.pth')
			start_epoch = checkpoint['epoch']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			model = model.to(device)

			testK_loader_args = dict(shuffle=False, batch_size=1, num_workers=0, pin_memory=False) if cuda\
								else dict(shuffle=False, batch_size=1)
			testK_loader = DataLoader(Test, **testK_loader_args)
			r = test_model_kaggle(model, testK_loader, criterion,idx_map)
			import pandas as pd
			df = pd.DataFrame(r)

			df.to_csv('../result/result0212.csv', index=False)
		elif args.mode == "verification":
			veri_data = []
			with open('../validation_trials_verification.txt','r') as veri_list:
				for veri_item in veri_list:
					veri_data.append(veri_item.replace('\n','').split(' '))

			
			veri_dataset = VeriDataset(veri_data)
			veri_loader_args = dict(shuffle=False, batch_size=1, num_workers=1, pin_memory=True) if cuda\
						else dict(shuffle=False, batch_size=1)
			veri_loader = DataLoader(veri_dataset, **veri_loader_args)

			model = face_classifier()


			checkpoint = torch.load('../checkpoints/0429 model_epoch_29.pth')
			start_epoch = checkpoint['epoch']
			model.load_state_dict(checkpoint['state_dict'])
			
			closs_weight = 0.5
			lr_cent = 0.5
			feat_dim = 10
			lr = 1e-2
			criterion = nn.CrossEntropyLoss()
			criterion_closs = CenterLoss(2300, feat_dim, device)
			optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)#optim.Adam(model.parameters(),lr=0.01, momentum=0.9)#
			optimizer.load_state_dict(checkpoint['optimizer'])
			optimizer_closs = torch.optim.SGD(criterion_closs.parameters(), lr=lr_cent)
			scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
			print(device)
			model = model.to(device)
			model.eval()
			test_loss = []
			accuracy = 0
			total = 0
			result = []
			for batch_num, (l_img,r_img, labels) in enumerate(veri_loader):
				l_img, r_img, labels = l_img.to(device), r_img.to(device), labels.to(device)

				l_feature, l_outputs = model(l_img)
				r_feature, r_outputs = model(r_img)
				#print(l_outputs.data,r_outputs.data)
				cos = nn.CosineSimilarity(dim=1, eps=1e-6)
				output = cos(l_outputs, r_outputs)
				#print(output.size(), output.data)
				result.append(output.data, labels)
				
				del l_img
				del r_img
				del labels
			df = pd.DataFrame(result)

			df.to_csv('./result_veri.csv', index=False)

		elif args.mode == "test_veri":
			test_data = []
			with open('../test_trials_verification_student.txt','r') as test_list:
				for test_item in test_list:
					#print(test_item.replace('\n','').split(' '))
					test_data.append(test_item.replace('\n','').split(' '))


			
			test_dataset = TestVeriDataset(test_data)
			test_loader_args = dict(shuffle=False, batch_size=1, num_workers=1, pin_memory=True) if cuda\
						else dict(shuffle=False, batch_size=1)
			test_loader = DataLoader(test_dataset, **test_loader_args)

			model = face_classifier()


			checkpoint = torch.load('../checkpoints/0427 model_epoch_39.pth')
			start_epoch = checkpoint['epoch']
			
			model.load_state_dict(checkpoint['state_dict'])
			
			closs_weight = 1
			lr_cent = 0.5
			feat_dim = 10
			lr = 1e-2
			criterion = nn.CrossEntropyLoss()
			criterion_closs = CenterLoss(2300, feat_dim, device)
			optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
			optimizer.load_state_dict(checkpoint['optimizer'])
			optimizer_closs = torch.optim.SGD(criterion_closs.parameters(), lr=lr_cent)
			scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
			print(device)
			model = model.to(device)
			model.eval()
			test_loss = []
			accuracy = 0
			total = 0
			result = []
			import pandas as pd
			start_time = time.time()
			for batch_num, (l_img,r_img, labels) in enumerate(test_loader):
				l_img, r_img= l_img.to(device), r_img.to(device)
				l_feature, l_outputs = model(l_img)
				r_feature, r_outputs = model(r_img)
				#print(l_outputs.data,r_outputs.data)
				cos = nn.CosineSimilarity(dim=1, eps=1e-6)
				output = cos(l_outputs, r_outputs)
				#print(labels[0][0] + '\n' + labels[1][0] )
				#print([labels[0] + '\n' + labels[1] , output.data.cpu().numpy()[0]] )
				result.append([labels[0][0] + '\n' + labels[1][0] , output.data.cpu().numpy()[0]])
				del l_img
				del r_img
				del labels

				if(batch_num % 10000 == 0):
					end_time = time.time()
					print(batch_num, 'Time: ',end_time - start_time, 's')
					start_time = end_time
			df = pd.DataFrame(result,columns = ['trial','score'])

			df.to_csv('./result_veri.csv', index=False)

			


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--test", default=False, action='store_true', help="")
	parser.add_argument("--mode", default="classification", help="")
	parser.add_argument("--n_epoch", type=int, default=30, help="")
	args = parser.parse_args()
	main(args)
