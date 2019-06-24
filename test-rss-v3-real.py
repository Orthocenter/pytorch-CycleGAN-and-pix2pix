import glob

files = glob.glob("/mnt/data/yanzi/xiaomi_vacuum_as_data_collector/data_parsed_all/*.pickle")
files_png = glob.glob("/mnt/data/yanzi/xiaomi_vacuum_as_data_collector/data_parsed_all/*.png")

def file_info(fname):
    parts = fname.split('/')[-1].split('_')
    date, time, dev, mac, _, pkt_type, _ = parts
    return date, time, dev, mac, pkt_type

import pickle
import numpy as np
import os
import sys
sys.path.append("/home/gomezp/advloc")

from models import create_model
from options.test_options import TestOptions
import torch
from data.rss_dataset import RSSDataset

# hack to avoid passing command-line arguments
sys.argv = ["--seed=666", "--gpu_ids=1",
            "--dataroot=''", "--name=rss_v3_8",
            "--model=rssmap2rssmap", "--input_nc=1",
            "--output_nc=1", "--norm=batch",
            "--dataset_mode=rss", "--num_threads=0",
            "--batch_size=1", "--netG=unet_64",
            "--verbose", "--no_flip", "--serial_batches"]
opt = TestOptions().parse()  # get test options
opt.display_id = -1 # do not use visdom; we will save plots ourselves

test_epoches = [200]
dataset = RSSDataset(opt)
model = create_model(opt)
opt.epoch = test_epoches[0]
opt.load_iter = 0
model.setup(opt)

def test_single(A_path):
    with open(A_path, 'rb') as f:
        data_A = pickle.load(f)
    if data_A[1] is None:
        return
    
    # Extract ground truth from file
    date, time, dev, mac, pkt_type = file_info(A_path)
    map_A, x, y = data_A[0], *data_A[1]
    loc = np.array([x, y])
    
    # Prepare image as model input
    map_A = torch.tensor(dataset.normalize_data(map_A), dtype=torch.float32)
    map_A = map_A.view((1, map_A.size()[0], -1)).unsqueeze(0)

    tx_loc_np = (loc - 32.) / 64.
    # `normalize_loc` is not correct here: input coords are i [0,64] range, not [-5,5]
    #tx_loc_np = dataset.normalize_loc(loc)
    tx_loc = torch.tensor(tx_loc_np).float()
    tx_pwr_np = dataset.normalize_data(np.array([-34])) # hack; figure this out!
    tx_pwr = torch.tensor(tx_pwr_np).float()
    tx_loc_pwr = torch.cat((tx_loc, tx_pwr))

    # Pack input data
    data = {'A': map_A, 'B': map_A, 'A_paths': A_path, 'B_paths': A_path, 'tx_loc_pwr': tx_loc_pwr}
    
    realA = map_A.squeeze().numpy()
    
    # Forward and extract visuals
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    visuals_np = {}
    for v in visuals:
        t = visuals[v]
        visuals_np[v] = t.cpu().float().squeeze().numpy()
    
    # Extract transmitter location from latent space
    task_tx_loc = model.latent[:2].cpu().float().squeeze().numpy()
    
    #return realA, realB, visuals_np, tx_loc_np, tx_pwr_np, task_tx_loc
    return realA, visuals_np, loc, task_tx_loc


def l2(x):
    from math import sqrt
    return sqrt(sum(pow(x, 2)))

diffs = []
for idx, fname in enumerate(files):
    res = test_single(fname)
    if res is None:
        continue
        
    _, _, loc, task_loc = res
    loc = (loc - 32.) / 64.
    diffs.append(l2(loc - task_loc))

diffs = np.array(diffs)
print("mean, min, max")
print(diffs.mean(), diffs.min(), diffs.max())