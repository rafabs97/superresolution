import argparse
import pandas as pd
import tqdm

import numpy as np
import cv2
import tensorflow as tf

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
#tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

from tensorflow.keras.models import load_model

from lpips import LPIPS, im2tensor
from impl.metrics import PSNR
from impl.datagen import SRDataGenerator

from math import sqrt

if __name__ == '__main__':

    # Argument parsing

    parser = argparse.ArgumentParser(description="Model testing (on dataset).")
    parser.add_argument('model_path', help="Path to the model.")
    parser.add_argument('--data_dir', default="./data", help="Source data directory.")
    parser.add_argument('--dataset_name', default="", help="Dataset name (of .csv file).")
    args = parser.parse_args()

    # Define paths

    data_dir = args.data_dir
    patch_dir = data_dir + '/patches'

    # Control parameters

    batch_size = 30
    #border = 8
    border = None

    # LPIPS metric (for monitoring)

    lpips_model = LPIPS(net = 'alex').cuda()
    lpips_alex = lambda x, y: lpips_model.forward(
        im2tensor(x, factor = 0.5).cuda(),
        im2tensor(y, factor = 0.5).cuda()).item()

    # Load model

    model = load_model(args.model_path)

    # Load data

    if args.dataset_name != "":
        test_csv = args.dataset_name + '_test.csv'
    else:
        test_csv = 'test.csv'

    test_df = pd.read_csv(data_dir + '/' + test_csv)
    #test_gen = SRDataGenerator(test_df, patch_dir, batch_size, False, border)

    # Per-picture

    print("Per patch metrics:")

    psnr = []
    lpips = []

    for _, row in tqdm.tqdm(test_df.iterrows(), total = len(test_df.index)):

        lq = cv2.imread(patch_dir + '/' + row['name'] + '_lq.png')
        sr = np.clip(model(np.array([lq / 255.0])), 0, 255)[0]

        if border != None:
            hq = cv2.imread(patch_dir + '/' + row['name'] + '_hq.png')[border:-border,border:-border] / 255.0
        else:
            hq = cv2.imread(patch_dir + '/' + row['name'] + '_hq.png') / 255.0

        psnr.append(PSNR(hq, sr)) # PSNR
        lpips.append(lpips_alex(hq, sr)) # LPIPS

    print("%s on %s:\n" % (args.model_path, test_csv))
    print("Check: %d\n" % sum([val < 0.0 for val in lpips]))
    print("PSNR: %2e +- %.2e\n" % (np.mean(psnr), 1.96 * np.std(psnr) / sqrt(len(test_df.index))))
    print("LPIPS: %e +- %.2e\n\n" % (np.mean(lpips), 1.96 * np.std(lpips) / sqrt(len(test_df.index))))

    # Per batch

    '''
    print("Per batch metrics:")

    psnr = []
    lpips = []

    for j in tqdm.tqdm(range(test_gen.__len__())):

        lq, hq = test_gen.__getitem__(j)

        sr = model(lq)

        psnr.append(PSNR(hq, sr)) # PSNR
        lpips.append(np.mean([lpips_alex(a, b) for a, b in zip(hq, sr.numpy())])) # LPIPS

    print("PSNR: %.2f +- %.2f" % (np.mean(psnr), np.std(psnr)))
    print("LPIPS: %.2f +- %.2f" % (np.mean(lpips), np.std(lpips)))
    '''