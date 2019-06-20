import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
# sys.path.append('../')

from models import create_model
from options.test_options import TestOptions

opt = TestOptions().parse([])  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

# parameters for this experiments
###############################
opt.name = "rss_v3_7"

dir_a = "/mnt/data/yanzi/input_synthetic_train_gamma_2.0"
dir_b = "/mnt/data/yanzi/input_real_emu_train_gamma_5.0_noise_10dBvar_same_loc_as_synthetic"

test_epoches = range(200,205,5)

###############################

opt.model = "rssmap2rssmap"
opt.input_nc = 1
opt.output_nc = 1
opt.norm = "batch"
opt.netG = "unet_64"

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
import numpy as np

import pickle
import torch

from data.rss_dataset import RSSDataset

opt.dataroot = ''
dataset = RSSDataset(opt)

def test_single(A_path, B_path):
    # open source A and target B
    with open(A_path, 'rb') as f:
        data_A = pickle.load(f)
    with open(B_path, 'rb') as f:
        data_B = pickle.load(f)
    
    data_A = dataset.transform(data_A).unsqueeze(0)
    data_B = dataset.transform(data_B).unsqueeze(0)
    
    # extract ground truth from filenames
    tx_loc_np = dataset.get_loc_from_path(A_path)
    tx_loc = torch.tensor(tx_loc_np).float()
    tx_pwr_np = dataset.get_pwr_from_path(A_path)
    tx_pwr = torch.tensor(tx_pwr_np).float()

    tx_loc_pwr = torch.cat((tx_loc, tx_pwr))

    data = {'A': data_A, 'B': data_B, 'A_paths': A_path, 'B_paths': B_path, 'tx_loc_pwr': tx_loc}
    
    realA = data_A.squeeze().numpy()
    realB = data_B.squeeze().numpy()
    
    # compute v3 visual output
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    visuals_np = {}
    for v in visuals:
        t = visuals[v]
        visuals_np[v] = t.cpu().float().squeeze().numpy()
    
    # extract v3 task output
    with torch.no_grad():
        _, task_tx_loc = model.netG(data_A)
        task_tx_loc = task_tx_loc.cpu().float().squeeze().numpy()
    
    return realA, realB, visuals_np, tx_loc_np, tx_pwr_np, task_tx_loc

# tx location l2 diff, rss map l1 diff
gamma_a = "2.0"
gamma_b = "5.0"

import math
import glob

def l1(x):
    return np.sum(np.abs(x))

def l2(x):
    return math.sqrt(np.sum(x**2))

def scale_rss(x):
    return x
    
def denorm_rss(x):
    return (x + 1) / 2 * (dataset.max_rss - dataset.min_rss) + dataset.min_rss

d3 = []
d4 = []
d5 = []

p1 = []
p2 = []
p3 = []

sim1 = []
sim2 = []
epoches = []

paths = glob.glob(dir_a + "/*.pickle")
paths = sorted(paths)

for epoch in test_epoches:
    epoches.append(epoch)
    
    opt.epoch = "%d" % epoch
    opt.load_iter = 0

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    dis_gt_loc = []
    dis_gt_taskrB = []
    
    pwr_gt_latent = []
    pwr_gt_taskrB = []
    
    sim_realA = []
    sim_fakeB = []
    
    for path_a in paths:
        tmp = path_a.split("/")[-1].split('_')
        x = float(tmp[1])
        y = float(tmp[2])
        
        if x < -3.2 or x > 3.2 or y < -3.2 or y > 3.2:
            continue
        
        pwr = float(tmp[-1].split('.')[0])

        test_file = "img_%.2f_%.2f_"  % (x, y) + "%s_" + "%d.pickle" % pwr
        path_b = os.path.join(dir_b, test_file % gamma_b)
        
        realA, realB, visuals, txloc, txpwr, task_tx_loc = test_single(path_a, path_b)

        fakeB = visuals['fake_B']
        dis_gt_loc.append(l2(txloc - task_tx_loc[:2]))
        #dis_gt_taskrB.append(l2(txloc - task_rB[:2]))
        # not training to compute power for now
        #pwr_gt_taskA.append(abs(txpwr - task_A[2:]))
        #pwr_gt_taskB.append(abs(txpwr - task_B[2:]))
        #pwr_gt_taskrB.append(abs(txpwr - task_rB[2:]))

        scaled_realA = scale_rss(denorm_rss(realA))
        scaled_fakeB = scale_rss(denorm_rss(fakeB))
        scaled_realB = scale_rss(denorm_rss(realB))
        sim_realA.append(l1(scaled_realA - scaled_realB) / realA.size)
        sim_fakeB.append(l1(scaled_fakeB - scaled_realB) / realA.size)
    
    #dis_gt_taskA = np.array(dis_gt_taskA) * 5
    #dis_gt_taskB = np.array(dis_gt_taskB) * 5
    dis_gt_latent = np.array(dis_gt_loc) * 5
    #dis_gt_taskrB = np.array(dis_gt_taskrB) * 5
    #pwr_gt_taskA = denorm_rss(np.array(pwr_gt_taskA))
    #pwr_gt_taskB = denorm_rss(np.array(pwr_gt_taskB))
    #pwr_gt_taskrB = denorm_rss(np.array(pwr_gt_taskrB))
    sim_realA = np.array(sim_realA)
    sim_fakeB = np.array(sim_fakeB)
    
    """print('[epoch %d] dis_gt_latent: %.2f, dis_gt_taskrB: %.2f' % \
          (epoch, dis_gt_latent.mean(), dis_gt_taskrB.mean()))"""
    print('[epoch %d] dis_gt_latent: %.2f' % \
          (epoch, dis_gt_latent.mean()))

    #print('           pwr_gt_taskA: %.2f, pwr_gt_taskB: %.2f, pwr_gt_taskrB: %.2f' % \
    #      (pwr_gt_taskA.mean(), pwr_gt_taskB.mean(), pwr_gt_taskrB.mean()))
    print('           sim_realA: %.2f, sim_fakeB: %.2f' % (sim_realA.mean(), sim_fakeB.mean()))
    
    d4.append((dis_gt_latent.mean(), dis_gt_latent.min(), dis_gt_latent.max()))
    #d5.append((dis_gt_taskrB.mean(), dis_gt_taskrB.min(), dis_gt_taskrB.max()))

    """p1.append((pwr_gt_taskA.mean(), pwr_gt_taskA.min(), pwr_gt_taskA.max()))
    p2.append((pwr_gt_taskB.mean(), pwr_gt_taskB.min(), pwr_gt_taskB.max()))
    p3.append((pwr_gt_taskrB.mean(), pwr_gt_taskrB.min(), pwr_gt_taskrB.max()))"""
    
    sim1.append((sim_realA.mean(), sim_realA.min(), sim_realA.max()))
    sim2.append((sim_fakeB.mean(), sim_fakeB.min(), sim_fakeB.max()))
