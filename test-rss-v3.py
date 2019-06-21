import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# sys.path.append('../')

from models import create_model
from options.test_options import TestOptions

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
import numpy as np
import math
import glob

import pickle
import torch

from data.rss_dataset import RSSDataset

opt = TestOptions().parse()  # get test options
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # do not use visdom; we will save plots ourselves

# Choose model and data paths
###############################
dir_a = "/mnt/data/yanzi/input_synthetic_train_gamma_2.0"
dir_b = "/mnt/data/yanzi/input_real_emu_train_gamma_5.0_noise_10dBvar_same_loc_as_synthetic"
###############################

test_epoches = range(30,200,10)
dataset = RSSDataset(opt) # we just need the functionality in RSSDataset
model = create_model(opt) # Create a model given opt.model and other options

"""
Given synthetic RSS map path `A_path` and
real RSS map path `B_path`, compute generated
fake RSS map G(A) and transmitter location using
the v3 model.
"""
def test_single(A_path, B_path):
    with open(A_path, 'rb') as f:
        data_A = pickle.load(f)
    with open(B_path, 'rb') as f:
        data_B = pickle.load(f)
    
    # Prepare images as model input
    data_A = dataset.transform(data_A).unsqueeze(0)
    data_B = dataset.transform(data_B).unsqueeze(0)
    
    # Extract ground truth from filenames
    tx_loc_np = dataset.get_loc_from_path(A_path)
    tx_loc = torch.tensor(tx_loc_np).float()
    tx_pwr_np = dataset.get_pwr_from_path(A_path)
    tx_pwr = torch.tensor(tx_pwr_np).float()
    tx_loc_pwr = torch.cat((tx_loc, tx_pwr))

    # Pack input data
    data = {'A': data_A, 'B': data_B, 'A_paths': A_path, 'B_paths': B_path, 'tx_loc_pwr': tx_loc_pwr}
    
    realA = data_A.squeeze().numpy()
    realB = data_B.squeeze().numpy()
    
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
    """
    This is old: used for previous model on separate task network
    with torch.no_grad():
        _, task_tx_loc = model.netG(data_A)
        task_tx_loc = task_tx_loc.cpu().float().squeeze().numpy()
    """
    
    return realA, realB, visuals_np, tx_loc_np, tx_pwr_np, task_tx_loc


###############################
# Helper functions

def l1(x):
    return np.sum(np.abs(x))

def l2(x):
    return math.sqrt(np.sum(x**2))

def scale_rss(x):
    return x
    
def denorm_rss(x):
    return (x + 1) / 2 * (dataset.max_rss - dataset.min_rss) + dataset.min_rss
###############################

# Source and target domain gammas
gamma_a = "2.0"
gamma_b = "5.0"

# Bookkeeping, TODO: clean up and summarize
d3 = []
d4 = []
d5 = []

p1 = []
p2 = []
p3 = []

sim1 = []
sim2 = []
epoches = []

# Collect all source domain images
paths = glob.glob(dir_a + "/*.pickle")
paths = sorted(paths)

for epoch in test_epoches:
    # Prepare to load model at `epoch`
    epoches.append(epoch)
    opt.epoch = "%d" % epoch
    opt.load_iter = 0

    # Regular setup for each epoch: load and print networks, create schedulers
    model.setup(opt) 

    # Ground truth location distance
    dis_gt_loc = []
    # Ground truth power distance
    dis_gt_pwr = []
    
    sim_realA = []
    sim_fakeB = []
    
    # Test every single image in source domain A
    for path_a in paths:
        tmp = path_a.split("/")[-1].split('_')
        x = float(tmp[1])
        y = float(tmp[2])
        
        if x < -3.2 or x > 3.2 or y < -3.2 or y > 3.2:
            continue
        
        pwr = float(tmp[-1].split('.')[0])

        # Load corresponding image in target domain B with same location and power
        test_file = "img_%.2f_%.2f_"  % (x, y) + "%s_" + "%d.pickle" % pwr
        path_b = os.path.join(dir_b, test_file % gamma_b)
        
        # Compute!
        realA, realB, visuals, txloc, txpwr, task_txloc = test_single(path_a, path_b)

        # Calculate performance metrics
        ###############################
        fakeB = visuals['fake_B']

        # Transmitter location L2 distance
        dis_gt_loc.append(l2(txloc - task_txloc))
        # TODO: train power prediction as well

        scaled_realA = scale_rss(denorm_rss(realA))
        scaled_fakeB = scale_rss(denorm_rss(fakeB))
        scaled_realB = scale_rss(denorm_rss(realB))

        # Per-pixel RSS map L1 distances
        sim_realA.append(l1(scaled_realA - scaled_realB) / realA.size)
        sim_fakeB.append(l1(scaled_fakeB - scaled_realB) / realA.size)
        ###############################
    
    #dis_gt_taskA = np.array(dis_gt_taskA) * 5
    #dis_gt_taskB = np.array(dis_gt_taskB) * 5
    dis_gt_loc = np.array(dis_gt_loc) * 5 # TODO: why x5?
    #dis_gt_taskrB = np.array(dis_gt_taskrB) * 5
    #pwr_gt_taskA = denorm_rss(np.array(pwr_gt_taskA))
    #pwr_gt_taskB = denorm_rss(np.array(pwr_gt_taskB))
    #pwr_gt_taskrB = denorm_rss(np.array(pwr_gt_taskrB))
    sim_realA = np.array(sim_realA)
    sim_fakeB = np.array(sim_fakeB)
    
    # Print localization statistics
    print('[epoch %d] dis_gt_loc: %.2f' % \
          (epoch, dis_gt_loc.mean()))

    # Print per-pixel RSS map difference statistics
    print('           sim_realA: %.2f, sim_fakeB: %.2f' % (sim_realA.mean(), sim_fakeB.mean()))
    
    # Collect epoch statistics
    d4.append((dis_gt_loc.mean(), dis_gt_loc.std(), dis_gt_loc.min(), dis_gt_loc.max()))
    #d5.append((dis_gt_taskrB.mean(), dis_gt_taskrB.min(), dis_gt_taskrB.max()))
    """p1.append((pwr_gt_taskA.mean(), pwr_gt_taskA.min(), pwr_gt_taskA.max()))
    p2.append((pwr_gt_taskB.mean(), pwr_gt_taskB.min(), pwr_gt_taskB.max()))
    p3.append((pwr_gt_taskrB.mean(), pwr_gt_taskrB.min(), pwr_gt_taskrB.max()))"""
    sim1.append((sim_realA.mean(), sim_realA.min(), sim_realA.max()))
    sim2.append((sim_fakeB.mean(), sim_fakeB.std(), sim_fakeB.min(), sim_fakeB.max()))


# Plot some figures
import time
timestamp = time.time()
out_dir = "./results/{}_out_{}".format(opt.name, timestamp)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
test_epoches = list(test_epoches)

# Localization error
###############################
plt.figure()
plt.title("Localization error (mean)")
plt.xlabel("Epoch")
plt.ylabel("Distance (L2)")

d4 = np.array(d4)
loc_errs = d4[:,0]
loc_std = d4[:,1]
plt.errorbar(test_epoches, loc_errs, loc_std, marker="s", linestyle="--", capsize=5, color="blue")
plt.savefig("{}/loc_errs.pdf".format(out_dir))

# Per-pixel RSS error
###############################
plt.figure()
plt.title("Per-pixel RSS error (mean)")
plt.xlabel("Epoch")
plt.ylabel("Distance (L1)")

sim2 = np.array(sim2)
rss_errs = sim2[:,0]
rss_std = sim2[:,1]
plt.errorbar(test_epoches, rss_errs, rss_std, marker="s", linestyle="--", capsize=5, color="red")
plt.savefig("{}/rss_errs.pdf".format(out_dir))