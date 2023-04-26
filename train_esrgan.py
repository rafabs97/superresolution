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
tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=6144)])

import pandas as pd
from lpips import LPIPS, im2tensor
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from impl.architectures import ESRGAN_disc, ESRGAN_gen
from impl.datagen import SRDataGenerator
from impl.metrics import PSNR, VGG_loss, discriminator_loss, generator_loss

if __name__ == '__main__':

    # Argument parsing

    parser = argparse.ArgumentParser(description="ESRGAN-based model training.")
    parser.add_argument('run_name', help="Name for the run recording.")
    parser.add_argument('--data_dir', default="./data", help="Source data directory.")
    parser.add_argument('--dataset_name', default="", help="Dataset name (of .csv file).")
    parser.add_argument('--save_path', default="./models", help="Where to save models/logs.")
    parser.add_argument('--test_pic', default=None, help="Picture used to test the model progression.")
    args = parser.parse_args()

    # W & B initialization

    wandb.init(project = "ESRGAN", name = args.run_name)

    # Define paths

    data_dir = args.data_dir
    patch_dir = data_dir + '/patches'
    save_path = args.save_path

    psnr_gen_path = save_path + '/%s_psnr.h5' % args.run_name
    gen_path = save_path + '/%s_gen.h5' % args.run_name
    disc_path = save_path + '/%s_disc.h5' % args.run_name

    #========== PSNR TRAIN ==========

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

    train_gen = SRDataGenerator(train_df, patch_dir, batch_size, True)
    val_gen = SRDataGenerator(validation_df, patch_dir, batch_size, False)

    # Define models

    generator = ESRGAN_gen()
    discriminator = ESRGAN_disc(64)

    gen_opt = Adam(learning_rate = 2e-4)

    # LPIPS metric (for monitoring)

    lpips_model = LPIPS(net = 'alex').cuda()
    lpips_alex = lambda x, y: lpips_model.forward(
        im2tensor(x, factor = 0.5).cuda(),
        im2tensor(y, factor = 0.5).cuda()).item()

    #========== PSNR TRAIN LOOP ==========

    best_loss = float('inf') # Initial best loss found
    no_improvement = 0 # Nº of epochs without improvement

    for i in range(epochs):

        #========== TRAIN ==========
        
        train_loss = train_psnr = train_lpips = 0

        for j in range(train_gen.__len__()): # Full epoch

            lq, hq = train_gen.__getitem__(j)

            # Forsard pass and train metrics computation

            with tf.GradientTape(persistent = True) as tape:
                sr = generator(lq)

                gen_losses = {}
                gen_losses['MSE'] = MeanSquaredError()(hq, sr) # Content/L1 loss
                gen_loss = tf.add_n([loss for loss in gen_losses.values()])

                psnr = PSNR(hq, sr)
                lpips = np.mean([lpips_alex(a, b) for a, b in zip(hq, sr.numpy())]) # LPIPS

            # Backpropagation

            gen_gradients = tape.gradient(gen_loss, generator.trainable_variables)
            gen_opt.apply_gradients(zip(gen_gradients, generator.trainable_variables))

            # Accumulate loss

            train_loss = (train_loss * j + gen_losses['MSE']) / (j + 1)
            train_psnr = (train_psnr * j + psnr) / (j + 1)
            train_lpips = (train_lpips * j + lpips) / (j + 1)
        
        #========== VALIDATION ==========

        val_loss = val_psnr = val_lpips = 0

        # Compute validation metrics

        for j in range(val_gen.__len__()):

            lq, hq = val_gen.__getitem__(j)
            sr = generator(lq)

            mse = MeanSquaredError()(hq, sr)
            psnr = PSNR(hq, sr)
            lpips = np.mean([lpips_alex(a, b) for a, b in zip(hq, sr.numpy())])

            val_loss = (val_loss * j + mse) / (j + 1)
            val_psnr = (val_psnr * j + psnr) / (j + 1)
            val_lpips = (val_lpips * j + lpips) / (j + 1)

        #========== W & B log ==========

        # W & B monitor

        wandb.log({
            "Learning rate (pre-training)": gen_opt.learning_rate,
            "Training MSE (pre-training)": train_loss,
            "Training PSNR (pre-training)": train_psnr,
            "Training LPIPS (pre-training)": train_lpips,
            "Validation MSE (pre-training)": val_loss,
            "Validation PSNR (pre-training)": val_psnr,
            "Validation LPIPS (pre-training)": val_lpips
        })

        # Update picture

        if args.test_pic != None:

            sr_test = generator(np.array([test_pic / 255.0]))[0] * 255.0
            sr_test = np.clip(sr_test, 0, 255)

            wandb.log({
                "Progress": wandb.Image(cv2.cvtColor(sr_test, cv2.COLOR_BGR2RGB)) 
            })

        #========== TRAIN CONTROL ==========

        # Decrease LR or stop

        if val_loss < best_loss:
            best_loss = val_loss
            generator.save(psnr_gen_path)
            no_improvement = 0

        else:
            no_improvement = no_improvement + 1

            if no_improvement == reduce_after:
                gen_opt.learning_rate = gen_opt.learning_rate / 2
            elif no_improvement == patience:
                break

    #========== ESRGAN TRAIN ==========

    # Restore best weights (PSNR)

    del generator
    generator = load_model(psnr_gen_path)

    # Training control parameters

    steps = 1000000
    batch_size = 15

    # Data generators

    train_gen = SRDataGenerator(train_df, patch_dir, batch_size, True)

    # Optimizers

    gen_opt = Adam(learning_rate = 1e-4)
    disc_opt = Adam(learning_rate = 1e-4)

    # VGG feature extractor

    vgg = VGG19(input_shape = (None, None, 3), include_top = False)
    vgg.layers[20].activation = None
    
    get_features = Model(vgg.input, vgg.layers[20].output)
    get_features.trainable = False

    #========== TRAIN LOOP ==========

    best_lpips = float('inf') # Initial best LPIPS found

    n = 0 # Steps counter
    train_loss_gen = {'VGG': 0, 'RaGAN': 0, 'MSE': 0, 'Total': 0}
    train_loss_disc = 0
    train_psnr = train_lpips = 0

    no_improvement = 0 # Nº of steps (x10000) without improvement

    for i in range(steps):

        lq, hq = train_gen.__getitem__(i % train_gen.__len__())

        # Forward pass and train metrics computation

        with tf.GradientTape(persistent = True) as tape:
            sr = generator(lq)

            probs_real = discriminator(hq)
            probs_sr = discriminator(sr)

            gen_losses = {}
            gen_losses['VGG'] = VGG_loss(hq, sr, get_features) # Perceptual loss
            gen_losses['RaGAN'] = 5e-3 * generator_loss(probs_real, probs_sr) # RaGAN loss
            gen_losses['MSE'] = 1e-2 * MeanSquaredError()(hq, sr) # Content/L1 loss

            disc_losses = {}
            disc_losses['RaGAN'] = discriminator_loss(probs_real, probs_sr) # RaGAN loss

            gen_loss = tf.add_n([loss for loss in gen_losses.values()])
            disc_loss = tf.add_n([loss for loss in disc_losses.values()])

            psnr = PSNR(hq, sr) # PSNR
            lpips = np.mean([lpips_alex(a, b) for a, b in zip(hq, sr.numpy())]) # LPIPS

        # Backpropagation

        gen_gradients = tape.gradient(gen_loss, generator.trainable_variables)
        gen_opt.apply_gradients(zip(gen_gradients, generator.trainable_variables))

        if i % 2 == 1:
            disc_gradients = tape.gradient(disc_loss, discriminator.trainable_variables)
            disc_opt.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

        # Accumulate loss (monitoring)

        train_loss_disc = (train_loss_disc * n + disc_losses['RaGAN']) / (n + 1)

        train_loss_gen['VGG'] = (train_loss_gen['VGG'] * n + gen_losses['VGG']) / (n + 1)
        train_loss_gen['RaGAN'] = (train_loss_gen['RaGAN'] * n + gen_losses['RaGAN'] / 5e-3) / (n + 1)
        train_loss_gen['MSE'] = (train_loss_gen['MSE'] * n + gen_losses['MSE'] / 1e-2) / (n + 1)
        train_loss_gen['Total'] = (train_loss_gen['Total'] * n + gen_loss) / (n + 1)

        train_psnr = (train_psnr * n + psnr) / (n + 1)
        train_lpips = (train_lpips * n + lpips) / (n + 1)

        n = n + 1

        if n == 10000: # Each 10000 steps

            val_loss_gen = {'VGG': 0, 'RaGAN': 0, 'MSE': 0, 'Total': 0}
            val_loss_disc = 0
            val_psnr = val_lpips = 0

            # Compute validation metrics

            for j in range(val_gen.__len__()):

                lq, hq = val_gen.__getitem__(j)

                sr = generator(lq)
                probs_real = discriminator(hq)
                probs_sr = discriminator(sr)

                gen_loss_vgg = VGG_loss(hq, sr, get_features)
                gen_loss_ragan = generator_loss(probs_real, probs_sr)
                gen_loss_mse =  MeanSquaredError()(hq, sr)
                gen_loss = gen_loss_vgg + gen_loss_ragan * 5e-3 + gen_loss_mse * 1e-2

                psnr = PSNR(hq, sr) # PSNR
                lpips = np.mean([lpips_alex(a, b) for a, b in zip(hq, sr.numpy())]) # LPIPS

                val_loss_disc = (val_loss_disc * j + discriminator_loss(probs_real, probs_sr)) / (j + 1)
                val_loss_gen['VGG'] = (val_loss_gen['VGG'] * j + gen_loss_vgg) / (j + 1)
                val_loss_gen['RaGAN'] = (val_loss_gen['RaGAN'] * j + gen_loss_ragan) / (j + 1)
                val_loss_gen['MSE'] = (val_loss_gen['MSE'] * j + gen_loss_mse) / (j + 1)
                val_loss_gen['Total'] = (val_loss_gen['Total'] * j + gen_loss) / (j + 1)

                val_psnr = (val_psnr * j + psnr) / (j + 1)
                val_lpips = (val_lpips * j + lpips) / (j + 1)

            #========== W & B log ==========

            # W & B monitor
        
            wandb.log({
                "Generator learning rate": gen_opt.learning_rate,
                "Discriminator learning rate": disc_opt.learning_rate,
                "Training VGG loss": train_loss_gen['VGG'],
                "Training RaGAN loss (gen.)": train_loss_gen['RaGAN'],
                "Training MSE": train_loss_gen['MSE'],
                "Training loss (gen.)": train_loss_gen['Total'],
                "Training loss (disc.)": train_loss_disc,
                "Training PSNR": train_psnr,
                "Training LPIPS": train_lpips,
                "Validation VGG loss": val_loss_gen['VGG'],
                "Validation RaGAN loss (gen.)": val_loss_gen['RaGAN'],
                "Validation MSE": val_loss_gen['MSE'],
                "Validation loss (gen.)": val_loss_gen['Total'],
                "Validation loss (disc.)": val_loss_disc,
                "Validation PSNR": val_psnr,
                "Validation LPIPS": val_lpips
            })

            # Update picture

            if args.test_pic != None:

                sr_test = generator(np.array([test_pic / 255.0]))[0] * 255.0
                sr_test = np.clip(sr_test, 0, 255)

                wandb.log({
                    "Progress": wandb.Image(cv2.cvtColor(sr_test, cv2.COLOR_BGR2RGB)) 
                })

            # Reset accumulated train metrics

            n = 0
            train_loss_gen = {'VGG': 0, 'RaGAN': 0, 'MSE': 0, 'Total': 0}
            train_loss_disc = 0
            train_psnr = train_lpips = 0

            # Save weights

            if val_lpips < best_lpips:
                best_lpips = val_lpips
                discriminator.save(disc_path)
                generator.save(gen_path)
                no_improvement = 0

            else:
                no_improvement = no_improvement + 1

                if no_improvement == reduce_after:
                    gen_opt.learning_rate = gen_opt.learning_rate / 2
                    disc_opt.learning_rate = disc_opt.learning_rate / 2

                elif no_improvement == patience:
                    break
