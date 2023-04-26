import argparse
import os
import random as rn

import cv2
import numpy as np
import tensorflow as tf

import wandb

seed = 0
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(seed)
rn.seed(seed)
tf.random.set_seed(seed)

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

import pandas as pd
from lpips import LPIPS, im2tensor
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from impl.architectures import FSRCNN, SRCNN
from impl.datagen import SRDataGenerator
from impl.metrics import PSNR

if __name__ == '__main__':

    # Argument parsing

    parser = argparse.ArgumentParser(description="SRCNN/FSRCNN model training.")
    parser.add_argument('run_name', help="Name for the run recording.")
    parser.add_argument('--fsrcnn', action='store_true', help="Train FSRCNN instead.")
    parser.add_argument('--data_dir', default="./data", help="Source data directory.")
    parser.add_argument('--dataset_name', default="", help="Dataset name (of .csv file).")
    parser.add_argument('--save_path', default="./models", help="Where to save models/logs.")
    parser.add_argument('--test_pic', default=None, help="Picture used to test the model progression.")
    args = parser.parse_args()

    # W & B initialization

    if args.fsrcnn:
        wandb.init(project = "FSRCNN", name = args.run_name)
    else:
        wandb.init(project = "SRCNN", name = args.run_name)

    # Define paths

    data_dir = args.data_dir
    patch_dir = data_dir + '/patches'
    save_path = args.save_path
    model_path = save_path + '/%s.h5' % args.run_name
    
    # Training control parameters

    epochs = 1000
    patience = 10 # Stop after 10 epochs without improvement
    reduce_after = 5 # Reduce LR after 5 epochs without improvement
    batch_size = 30

    # Load data

    if args.dataset_name != "":
        train_csv = args.dataset_name + '_train.csv'
        validation_csv = args.dataset_name + '_validation.csv'
    else:
        train_csv = 'train.csv'
        validation_csv = 'validation.csv'

    train_df = pd.read_csv(data_dir + '/' + train_csv)
    validation_df = pd.read_csv(data_dir + '/' + validation_csv)

    if args.test_pic != None:
        test_pic = cv2.imread(args.test_pic)
    
    # Data generators

    border = None
    if args.fsrcnn == False: # Crop pictures for SRCNN
        border = 8

    train_gen = SRDataGenerator(train_df, patch_dir, batch_size, True, border)
    val_gen = SRDataGenerator(validation_df, patch_dir, batch_size, False, border)

    # Define models

    if args.fsrcnn:
        model = FSRCNN()
    else:
        model = SRCNN()

    opt = Adam(learning_rate = 2e-4)

    # LPIPS metric (for monitoring)

    lpips_model = LPIPS(net = 'alex').cuda()
    lpips_alex = lambda x, y: lpips_model.forward(
        im2tensor(x, factor = 0.5).cuda(),
        im2tensor(y, factor = 0.5).cuda()).item()

    #========== TRAIN LOOP ==========

    best_psnr = 0 # Initial best PSNR found
    no_improvement = 0 # Nº of epochs without improvement

    for i in range(epochs):
        
        #========== TRAIN ==========

        train_loss = train_psnr = train_lpips = 0

        for j in range(train_gen.__len__()): # Full epoch

            lq, hq = train_gen.__getitem__(j)

            # Forward pass and train metrics computation

            with tf.GradientTape(persistent = True) as tape:
                sr = model(lq)

                losses = {}
                losses['MSE'] = MeanSquaredError()(hq, sr) # Content loss
                loss = tf.add_n([loss for loss in losses.values()])

                psnr = PSNR(hq, sr) # PSNR
                lpips = np.mean([lpips_alex(a, b) for a, b in zip(hq, sr.numpy())]) # LPIPS

            # Backpropagation

            gradients = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(gradients, model.trainable_variables))

            # Accumulate loss (monitoring)

            train_loss = (train_loss * j + losses['MSE']) / (j + 1)
            train_psnr = (train_psnr * j + psnr) / (j + 1)
            train_lpips = (train_lpips * j + lpips) / (j + 1)

        #========== VALIDATION ==========

        val_loss = val_psnr = val_lpips = 0

        # Compute validation metrics

        for j in range(val_gen.__len__()):

            lq, hq = val_gen.__getitem__(j)
            sr = model(lq)

            mse = MeanSquaredError()(hq, sr)
            psnr = PSNR(hq, sr)
            lpips = np.mean([lpips_alex(a, b) for a, b in zip(hq, sr.numpy())])

            val_loss = (val_loss * j + mse) / (j + 1)
            val_psnr = (val_psnr * j + psnr) / (j + 1)
            val_lpips = (val_lpips * j + lpips) / (j + 1)

        #========== W & B log ==========

        # W & B monitor

        wandb.log({
            "Learning rate": opt.learning_rate,
            "Training MSE": train_loss,
            "Training PSNR": train_psnr,
            "Training LPIPS": train_lpips,
            "Validation MSE": val_loss,
            "Validation PSNR": val_psnr,
            "Validation LPIPS": val_lpips
        })

        # Update picture

        if args.test_pic != None:

            sr_test = model(np.array([test_pic / 255.0]))[0] * 255.0
            sr_test = np.clip(sr_test, 0, 255)

            wandb.log({
                "Progress": wandb.Image(cv2.cvtColor(sr_test, cv2.COLOR_BGR2RGB)) 
            })

        #========== TRAIN CONTROL ==========

        # Decrease LR or stop

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            model.save(model_path)
            no_improvement = 0

        else:
            no_improvement = no_improvement + 1

            if no_improvement == reduce_after:
                opt.learning_rate = opt.learning_rate / 2
            elif no_improvement == patience:
                break
