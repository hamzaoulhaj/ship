import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

names = ['Container Ship',
            'Bulk Carrier',
            'Passengers Ship',
            'Ro-ro/passenger Ship',
            'Ro-ro Cargo',
            'Tug',
            'Vehicles Carrier',
            'Reefer',
            'Yacht',
            'Sailing Vessel',
            'Heavy Load Carrier',
            'Wood Chips Carrier',
            'Patrol Vessel',
            'Platform',
            'Standby Safety Vessel',
            'Combat Vessel',
            'Icebreaker',
            'Replenishment Vessel',
            'Tankers',
            'Fishing Vessels',
            'Supply Vessels',
            'Carrier/Floating',
            'Dredgers']

num_classes = 23
def classifier(out_size, num_classes):
    classifier = nn.Sequential(nn.Linear(out_size, int(out_size/2)), 		          nn.BatchNorm1d(int(out_size/2)) , nn.ReLU(),
                       nn.Dropout(p=0.1), nn.Linear(int(out_size/2), num_classes))
    return(classifier)

def Resnet34(num_classes):
    net = torchvision.models.resnet34(pretrained=True, progress=True)
    out_size = net.fc.in_features
    net.fc = classifier(out_size, num_classes)
    return(net)
    
transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                        ])
                        

resol = (256,256)
Reduction = True
eps = 1e-7
PATH = "Shipspotting Varna, Bulgaria - Summer 2019.mp4"
cuda = True

model = Resnet34(num_classes).eval()
model.load_state_dict(torch.load('resnet.pth'))
if cuda:
    model = model.cuda()

video = cv2.VideoCapture(PATH)

#Determine fps
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver) < 3:
    fps = int(video.get(cv2.cv.CV_CAP_PROP_FPS))
else:
    fps = int(video.get(cv2.CAP_PROP_FPS))

r_, frame = video.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame, resol)
image = transform(frame).view(1,3,256,256)
softner = nn.Softmax()
if cuda:
    image = image.cuda()
prediction = model(image)
f = softner(prediction[0]).tolist()
p = softner(prediction[0]).tolist()
n = num_classes
ro = [0]*n

fr = 0

while(True):
    fr+=1

    r_, frame = video.read()
    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resol)
    except:
        print("Finished video")
        break
    image = transform(frame).view(1,3,256,256)
    if cuda:
        image = image.cuda()
    prediction = model(image)
    p = softner(prediction[0]).tolist()

    for i in range(n):
        s = 0
        for j in range(n):
	        s += p[j]*f[j]
        ro[i] = s / (p[i]*f[i] + eps)
    for i in range(n):
        f[i] = 1 / (ro[i] + eps)

    m = 0
    for i in range(n):
        if f[i]>=f[m]:
	        m = i
    
    print(names[m], "minute: ", fr/(60*fps))
    frame = None
    image = None

video.release()
cv2.destroyAllWindows()

