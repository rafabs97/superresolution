import os

import pandas as pd
import cv2
import tqdm

data_dir = 'data_ds'
data_csv = data_dir + '/patches.csv'
patch_dir = data_dir + '/patches'

data_df = pd.read_csv(data_csv)

for _, row in tqdm.tqdm(data_df.iterrows(), total = len(data_df.index)):
    hq_path = patch_dir + '/' + row['name'] + '_hq.png'
    lq_path = patch_dir + '/' + row['name'] + '_lq.png'

    os.remove(lq_path)

    pic_hq = cv2.imread(hq_path)
    width = int(pic_hq.shape[1] / 2.0)
    height = int(pic_hq.shape[0] / 2.0)

    cv2.imwrite(lq_path, cv2.resize(pic_hq, (width, height), interpolation = cv2.INTER_CUBIC))