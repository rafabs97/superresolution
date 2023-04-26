import argparse
import os
from random import randint

import cv2
import pandas as pd
from skimage.metrics import structural_similarity

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get patches from pictures.')
    parser.add_argument('data_dir', default = './', help = 'Directory containing training data.')
    args = parser.parse_args()

    source_dir = args.data_dir + '/original'
    source_csv = source_dir + '/selected.csv'
    target_dir = args.data_dir + '/patches'

    if os.path.isdir(target_dir) == False:
        os.mkdir(target_dir)

    hq_patch_size = 64
    num_patches = 50
    frame_df = pd.read_csv(source_csv)

    count = 1
    patch_corr = []

    for _, row in frame_df.iterrows():

        print('Getting patches from pic ' + str(count) + '/' + str(len(frame_df)) + '...', end = '\r', flush = True)

        hq = cv2.imread(source_dir + '/' + row['game'] + '/hq/' + row['hq'])
        lq = cv2.imread(source_dir + '/' + row['game'] + '/lq/' + row['lq'])

        for i in range(num_patches):

            ssim = 0
            pic_name = row['game'] + '_' + row['hq'].split('.')[0] + '_' + str(i)

            while ssim < 0.8:
                x = randint(0, hq.shape[1] - 1)
                while (x + hq_patch_size) > hq.shape[1]:
                    x = randint(0, hq.shape[1] - 1)

                y = randint(0, hq.shape[0] - 1)
                while (y + hq_patch_size) > hq.shape[0]:
                    y = randint(0, hq.shape[0] - 1)

                hq_crop = hq[y : y + hq_patch_size, x : x + hq_patch_size, :]
                lq_crop = lq[int(y / 2) : int((y + hq_patch_size) / 2), int(x / 2) : int((x + hq_patch_size) / 2), :]

                hq_crop_scaled = cv2.resize(hq_crop, (int(hq_patch_size / 2), int(hq_patch_size / 2)), interpolation=cv2.INTER_AREA)
                ssim = structural_similarity(hq_crop_scaled, lq_crop, channel_axis=-1)
            
            cv2.imwrite(target_dir + '/' + pic_name + '_hq.png', hq_crop)
            cv2.imwrite(target_dir + '/' + pic_name + '_lq.png', lq_crop)
            patch_corr.append([row['game'], pic_name, ssim])

        count = count + 1

    patch_df = pd.DataFrame(patch_corr, columns = ['game', 'name', 'ssim'])
    patch_df.to_csv(args.data_dir + '/patches.csv', index = False)
