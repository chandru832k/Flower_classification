
from google.colab import drive
drive.mount('/content/gdrive')


DIR_PATH = '/content/gdrive/MyDrive/Kaggle/flowers/flowers'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

transformations = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
}

learning_rate = 0.001
batch_size = 8
num_epochs = 50
num_classes = 5


device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(device)

total_dataset = torchvision.datasets.ImageFolder(DIR_PATH,transform=transformations['train'])

len(total_dataset),total_dataset[0][0].shape,total_dataset.class_to_idx

SPLIT_SIZE = 0.8
tot_len = len(total_dataset)

train_size = int(SPLIT_SIZE * tot_len)
val_size = tot_len - train_size

print(f'Training set size = {train_size} \nValidation set size = {val_size}')

train_dataset, val_dataset =  torch.utils.data.random_split(total_dataset,[train_size,val_size])

prediction_dataset=total_dataset[0][0]


print(prediction_dataset)

len(train_dataset),len(val_dataset)

# dataloaders
train_loader = DataLoader(dataset=train_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4)

val_loader = DataLoader(dataset=val_dataset,
                       batch_size=1,
                       shuffle=True,
                       num_workers=4)
prediction_loader=DataLoader(dataset=prediction_dataset,batch_size=1,shuffle=False,num_workers=4)

examples = iter(train_loader)
samples,labels = examples.next()
print(samples.shape,labels.shape)
len(train_loader),len(val_loader)

class ConvNet(nn.Module):
    def __init__(self,model,num_classes):
        super(ConvNet,self).__init__()
        self.base_model = nn.Sequential(*list(model.children())[:-1])
        self.linear1 = nn.Linear(in_features=2048,out_features=512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=512,out_features=num_classes)
    
    def forward(self,x):
        x = self.base_model(x)
        x = torch.flatten(x,1)
        lin = self.linear1(x)
        x = self.relu(lin)
        out = self.linear2(x)
        return lin, out

model = torchvision.models.resnet50(pretrained=True) 
model = ConvNet(model,num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)

print(model)

n_iters = len(train_loader)

for epoch in range(num_epochs):
    model.train()
    for ii,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        _,outputs = model(images)
        loss = criterion(outputs,labels)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (ii+1)%108 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{ii+1}/{n_iters}], Loss = {loss.item():.6f}')
            
    print('----------------------------------------')

def eval_model_extract_features(features,true_labels,model,dataloader,phase):

    with torch.no_grad():
        # for entire dataset
        n_correct = 0
        n_samples = 0

        model.eval()

        for images,labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            true_labels.append(labels)
            
            ftrs,outputs = model(images)
            features.append(ftrs)

            _,preds = torch.max(outputs,1)
            n_samples += labels.size(0)
            n_correct += (preds == labels).sum().item()
                
        accuracy = n_correct/float(n_samples)

        print(f'Accuracy of model on {phase} set = {(100.0 * accuracy):.4f} %')

    return features,true_labels

MODEL_PATH = '/content/gdrive/MyDrive/Kaggle/resnet50_TL_model_94(2)%acc.pth'
model = torchvision.models.resnet50(pretrained=True)
model = ConvNet(model,num_classes)
model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH))

features = []
true_labels = []

train_loader = DataLoader(dataset=train_dataset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=4)

features,true_labels = eval_model_extract_features(features,true_labels,model,dataloader=train_loader,phase='training')

print(len(features),len(true_labels))

features,true_labels = eval_model_extract_features(features,true_labels,model,dataloader=val_loader,phase='validation')

print(len(features),len(true_labels))

ftrs = features.copy() 
lbls = true_labels.copy()

for i in range(len(ftrs)):
    ftrs[i]=ftrs[i].cpu().numpy()

ftrs[0].shape

for i in range(len(lbls)):
    lbls[i]=lbls[i].cpu().numpy()

lbls[0].shape

type(ftrs),type(lbls)

ftrs = np.array(ftrs)
lbls = np.array(lbls)

ftrs.shape,lbls.shape

n_samples = ftrs.shape[0]*ftrs.shape[1]
n_features = ftrs.shape[2]
ftrs = ftrs.reshape(n_samples,n_features)

print(ftrs.shape)

n_lbls = lbls.shape[0]
lbls = lbls.reshape(n_lbls)

print(lbls.shape)

ftrs_df = pd.DataFrame(ftrs)
ftrs_df.to_csv('/content/gdrive/MyDrive/Kaggle/resnet50_FC_features_512.csv',index=False)


ftrs_df = pd.read_csv('/content/gdrive/MyDrive/Kaggle/resnet50_FC_features_512.csv')
ftrs_df

ftrs_df['label'] = lbls

ftrs_df.head()

ftrs_df.to_csv('/content/gdrive/MyDrive/Kaggle/resnet50_FC_512_features_with_labels1.csv',index=False)

print('feature set saved successfully !')

MODEL_PATH = '/content/gdrive/MyDrive/Kaggle/resnet50_TL_model_94(2)%acc.pth'
torch.save(model.state_dict(),MODEL_PATH)

model = torchvision.models.resnet50(pretrained=True)
model = ConvNet(model,num_classes)
model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH))

from PIL import Image
import numpy as np

h = 224
w = 224

pic = Image.open("/content/tulip.jpeg")
pix = transformations['train'](pic)
print(pix.size())

with torch.no_grad():
  model.eval()
  images = pix.to(device).unsqueeze(0)
  ftrs,outputs = model(images)
  _,preds = torch.max(outputs,1)
  print(preds)
  print(outputs)

