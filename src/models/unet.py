import os
import torch
import monai
from monai.transforms import (
    Compose,
    LoadImage,
    Resize,
    ScaleIntensity,
    EnsureType,
)
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss

# Define your dataset directory and classes
dataset_dir = "../../dataset_2d"
classes = ["proclined", "retroclined", "normal"]

# Load the dataset
images = []
labels = []

for i, cls in enumerate(classes):
    cls_path = os.path.join(dataset_dir, cls)
    for img_name in os.listdir(cls_path):
        if img_name.lower().endswith('.png'):  # Check if the file is a .png file
            img_path = os.path.join(cls_path, img_name)
            images.append(img_path)
            labels.append(i)  # Assuming class indices as labels

# Preprocessing transformations
transforms = Compose([
    LoadImage(image_only=True),
    Resize(spatial_size=(256, 256)),
    ScaleIntensity(),
    EnsureType(),
])

# Prepare dataset and dataloader
data_dicts = [{"image": img, "label": label} for img, label in zip(images, labels)]
dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=1.0, num_workers=4)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Create UNet model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    dimensions=2,
    in_channels=1,
    out_channels=len(classes),
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_data in dataloader:
        inputs, targets = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"epoch {epoch + 1}/{num_epochs}, average loss: {epoch_loss / len(dataloader)}")

# Add your validation loop here with DiceMetric for performance evaluation
