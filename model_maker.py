import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.optim as optim


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir) if
                            filename.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        filename = os.path.basename(img_path)
        x, y = map(float, filename.split('_')[:2])  #Extracting x, y from filename
        label = torch.tensor([x, y], dtype=torch.float32) / 224.0
        if self.transform:
            image = self.transform(image)
        return image, label


# Setup transforms and dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#Specifying dataset
dataset_path = r'C:\Users\Ilker\Desktop\mixed_data'
dataset = CustomDataset(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Continue with model setup and training as before
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


def train_model(model, dataloader, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}')


train_model(model, dataloader, 10) #10 Epochs

# Save the trained model
torch.save(model.state_dict(), 'C:\\Users\\Ilker\\Desktop\\trained_model.pth')
