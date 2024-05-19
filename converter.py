import torch
import torchvision
from torchvision import models
import torch.nn as nn
from torch2trt import torch2trt

CATEGORIES = ['apex']
device = torch.device('cuda')

#Load the ResNet model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 2 * len(CATEGORIES))
model = model.to(device).eval().half()

#Path
model_path = r'C:\Users\Ilker\Desktop\trained_model.pth'

model.load_state_dict(torch.load(model_path))

data = torch.zeros((1, 3, 224, 224)).cuda().half()

model_trt = torch2trt(model, [data], fp16_mode=True)

#Path where the TensorRT model will be saved
trt_model_path = r'C:\Users\Ilker\Desktop\road_following_modelresnet18_trt.pth'

torch.save(model_trt.state_dict(), trt_model_path)
