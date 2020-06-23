import pdb
import numpy as np
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

NUM_CLASSES = 65
RESTORE_MODE = False  # if True, it will load saved model from OUT_PATH and continue to train
START_ITER = 0 # starting iteration 
OUTPUT_PATH = 'con_out/' # output path where result (.e.g drawing images, cost, chart) will be stored
# MODE = 'wgan-gp'
DIM = 64 # Model dimensionality
CRITIC_ITERS = 5 # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 65# Batch size. Must be a multiple of N_GPUS
END_ITER = 100000 # How many iterations to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 64*64*3 # Number of pixels in each iamge
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 1. # How to scale generator's ACGAN loss relative to WGAN loss

OUTPUT_PATH = 'con_out/'

aG = torch.load(OUTPUT_PATH + "generator.pt")

def gen_rand_noise_with_label(label=None):
    if label is None:
        label = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
    #attach label into noise
    BATCH_SIZE = label.shape[0]
    noise = np.random.normal(0, 1, (BATCH_SIZE, 128))
    prefix = np.zeros((BATCH_SIZE, NUM_CLASSES))
    prefix[np.arange(BATCH_SIZE), label] = 1
    noise[np.arange(BATCH_SIZE), :NUM_CLASSES] = prefix[np.arange(BATCH_SIZE)]

    noise = torch.from_numpy(noise).float()
    noise = noise.to(device)

    return noise, label

def generate_image(netG, noise=None):
    if noise is None:
        rand_label = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
        noise = gen_rand_noise_with_label(rand_label)
    with torch.no_grad():
        noisev = noise
    samples = netG(noisev)
    # samples = samples.view(BATCH_SIZE, 3, DIM, DIM)

    samples = samples * 0.5 + 0.5

    return samples

lab = np.arange(0, NUM_CLASSES)
lab = np.repeat(lab, 50, axis=0) # to generate 50 samples per class

noise_sample, _ = gen_rand_noise_with_label(lab)
lab = torch.from_numpy(lab).float().unsqueeze(1)
image_vectors = generate_image(aG, noise_sample)
final_file = torch.cat((image_vectors.detach().cpu(), lab),1)
np.save("gen_vect_lab.npy", final_file)
pdb.set_trace()

