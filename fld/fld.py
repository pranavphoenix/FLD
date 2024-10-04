import os
import argparse

## PyTorch
import torch
import torch.utils.data as data

# Torchvision

from model import ImageFlow
from data import CustomImageDataset, DiscretizeTransform

parser = argparse.ArgumentParser()

parser.add_argument("-real", "--real_data_dir", default='/workspace/real_images/', help = "Path to Real Images Data Directory")
parser.add_argument("-gen", "--gen_data_dir", default='/workspace/gen_images/', help = "Path to Real Images Data Directory")
parser.add_argument("-batch", "--Batch_size", default=64, help = "Batch Size")
parser.add_argument("-path", "--save_flow_path", default='/workspace/flow.pth', help= "Path to save the trained Flow model")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)

n_flow = ImageFlow(
    depth_vq=4, 
    depth_coupling =8)

n_flow.to(device)

if os.path.isfile(str(args.save_flow_path)):
    print("Found pretrained n_flow, loading...")
    n_flow.load_state_dict(torch.load(str(args.save_flow_path)))
else:
    print("No saved weights found, use train.py first to train the flow on real images")
    exit()


n_flow.eval()

real_image_folder = str(args.real_data_dir)
gen_image_folder = str(args.gen_data_dir)

transform = DiscretizeTransform()

real_dataset = CustomImageDataset(image_folder=real_image_folder, transform=transform)
gen_dataset = CustomImageDataset(image_folder=gen_image_folder, transform=transform)

print(f"No. of Images in Real Data set: {len(real_dataset)}")
print(f"No. of Images in Generated Data set: {len(gen_dataset)}")


real_data_loader = data.DataLoader(real_dataset, batch_size=int(args.Batch_size), shuffle=False, drop_last=False, pin_memory=True, num_workers=8)
gen_data_loader = data.DataLoader(gen_dataset, batch_size=int(args.Batch_size), shuffle=False, drop_last=False, pin_memory=True, num_workers=8)


real_log_likelihood = 0
with torch.no_grad():
    for batch in real_data_loader:
        imgs = batch
        imgs = imgs.to(device)
        real_log_likelihood += n_flow(imgs)

    real_log_likelihood = real_log_likelihood / len(real_data_loader)

print(f"Average Log Likelihood of Real Images: {real_log_likelihood}")

gen_log_likelihood = 0
with torch.no_grad():
    for batch in gen_data_loader:
        imgs = batch
        imgs = imgs.to(device)
        gen_log_likelihood += n_flow(imgs)

    gen_log_likelihood = gen_log_likelihood / len(gen_data_loader)

print(f"Average Log Likelihood of Generated Images: {gen_log_likelihood}")

FLD = real_log_likelihood / gen_log_likelihood

print(f"FLD: {FLD}")