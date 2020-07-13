import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.model_zoo as model_zoo
from data import ShipDataset
from torch.optim import lr_scheduler
import time
from shutil import copyfile
from tqdm import tqdm
import pandas as pd

from torch.optim import lr_scheduler
from torch.autograd import Variable

pretrained = True
num_classes = 26
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-241335ed.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-6f0f7f60.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-4c113574.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-17b70270.pth',   
    'inceptionv3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
model = models.resnet50(pretrained)
model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], progress=True))

for param in model.parameters():
    param.requires_grad = False   
out_size = model.fc.in_features
model.fc = nn.Linear(out_size, num_classes)
model.load_state_dict(torch.load('models/model.pth'))

transformtest = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

#train_data = df.sample(frac=0.7)
#valid_data = df[~df['file'].isin(train_data['file'])]

batch_size = 64
testset = ShipDataset(csv_file='dataset_train.csv', transform=transformtest)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=2)
        
def accuracy_perclass(net, test, num_classes = 26, cuda=True):
    net.eval()
    correct = 0
    total = 0
    loss = 0
    TP = [0]*num_classes
    Total= [0]*num_classes
    with torch.no_grad():
        for data in tqdm(test):
            images, labels = data['image'], data['label'].view(-1)
            if cuda:
                net = net.cuda()
                images = images.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.LongTensor)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            for ind in range(labels.size(0)):
                for i in range(num_classes):
                    if i == labels[ind]:
                        TP[i] += int(labels[ind] == predicted[ind])
                        Total[i] += 1
            #print(predicted, labels, TP, Total)
                        
    for i in range(num_classes):
        if Total[i]!=0:
            TP[i] /= Total[i]
    
    net.train()
    return TP
 
accuracy_perclass(model, testloader)

