import os
import numpy as np
import torchinfo
from torchinfo import summary

## Progress bar
from tqdm import tqdm
import argparse

## PyTorch
import torch
import torch.utils.data as data
from data import CustomImageDataset, DiscretizeTransform

# Torchvision

from model import ImageFlow

parser = argparse.ArgumentParser()

parser.add_argument("-real", "--real_data_dir", default='/workspace/real images/', help = "Path to Real Images Data Directory")
parser.add_argument("-batch", "--Batch_size", default=64, help = "Batch Size")
parser.add_argument("-path", "--save_flow_path", default='/workspace/flow.pth', help= "Path to save the trained Flow model")
parser.add_argument("-load", "--load", default=False, help = "Check and Load Pre-trained Weights if available")

args = parser.parse_args()



# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fetching the device that will be used throughout this notebook
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)


transform = DiscretizeTransform()

image_folder = str(args.real_data_dir)

dataset = CustomImageDataset(image_folder=image_folder, transform=transform)

print(len(dataset))

trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])


# Data loaders
train_loader = data.DataLoader(trainset, batch_size=int(args.Batch_size), shuffle=True, drop_last=False, pin_memory=True, num_workers=8)
val_loader = data.DataLoader(valset, batch_size=int(args.Batch_size)*2, shuffle=False, drop_last=False, num_workers=4)




#Define the Normalizing Flow
n_flow = ImageFlow(
    depth_vq=4, 
    depth_coupling =8)

n_flow.to(device)
print(summary(n_flow, (1, 3, 256, 256), depth = 10))

counter = 0
epoch = 0
# Create optimizer and scheduler
optimizer = torch.optim.AdamW(n_flow.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

# Check if pretrained n_flow exists

if args.load:
    if os.path.isfile(str(args.save_flow_path)):
        print("Found pretrained n_flow, loading...")
        n_flow.load_state_dict(torch.load(str(args.save_flow_path)))
    else:
        print("No saved weights found, training from scratch")

print("Start training")
# Training loop
best_val_loss = float('inf')
while counter < 10:
    n_flow.train()
    train_losses = []
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        imgs = batch
        
        imgs = imgs.to(device)
        optimizer.zero_grad()
        loss = -n_flow(imgs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(n_flow.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())
    train_loss = np.mean(train_losses)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
    
    epoch += 1

    if (epoch+1) % 5 == 0:
        # Evaluate on validation set
        n_flow.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch
                imgs = imgs.to(device)
                loss = -n_flow(imgs)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        print(f"Validation Loss: {val_loss:.4f}")
        counter += 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(n_flow.state_dict(), str(args.save_flow_path))
            print("saving_weights")
            counter = 0


# After training, run the model over the entire real dataset
n_flow.load_state_dict(torch.load(str(args.save_flow_path)))

data_loader = data.DataLoader(dataset, batch_size=int(args.Batch_size), shuffle=False, drop_last=False, pin_memory=True, num_workers=8)
n_flow.eval()
log_likelihood = 0
with torch.no_grad():
    for batch in data_loader:
        imgs = batch
        imgs = imgs.to(device)
        log_likelihood += n_flow(imgs)

    log_likelihood = log_likelihood / len(data_loader)

print(f"Average Log Likelihood of Real Images: {log_likelihood}")












