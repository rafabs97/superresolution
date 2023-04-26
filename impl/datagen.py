# -*- coding: utf-8 -*-

import random
from math import ceil

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm


def random_transform(pic_a, pic_b):

    if random.randint(0, 1) == 0: # Flip vertically
        pic_a, pic_b = cv2.flip(pic_a, 0), cv2.flip(pic_b, 0)

    if random.randint(0, 1) == 0: # Flip horizontally
        pic_a, pic_b = cv2.flip(pic_a, 1), cv2.flip(pic_b, 1)

    rot = random.randint(0, 3) # Rotate
    if rot != 3: 
        pic_a, pic_b = cv2.rotate(pic_a, rot), cv2.rotate(pic_b, rot)
        
    return pic_a, pic_b

class SRDataGenerator(Sequence):

    def __init__(self, img_dataframe, source_dir, batch_size, augment = False, border = None):

        self.batch_size = batch_size
        self.augment = augment

        self.pics = {}
        self.common_iterator = 0

        print('Loading data...')
        for _, row in tqdm(img_dataframe.iterrows(), total=img_dataframe.shape[0]):
            if (row['game']) not in self.pics:
                self.pics[row['game']] = {}
                self.pics[row['game']]['hq'] = []
                self.pics[row['game']]['lq'] = []

            pic_hq = cv2.imread(source_dir + '/' + row['name'] + '_hq.png')
            pic_lq = cv2.imread(source_dir + '/' + row['name'] + '_lq.png')

            if border != None:
                pic_hq = pic_hq[border:-border,border:-border]

            self.pics[row['game']]['hq'].append(pic_hq)
            self.pics[row['game']]['lq'].append(pic_lq)

        assert (self.batch_size % len(self.pics.keys()) == 0)
        self.pics_per_game = int(self.batch_size / len(self.pics.keys()))
        self.length = ceil(max([len(self.pics[game]['hq']) for game in self.pics.keys()]) / self.pics_per_game)

    def __data_generation(self):

        hq = []
        lq = []

        for i in range(self.pics_per_game):
            for game in self.pics.keys():

                pic_hq = self.pics[game]['hq'][self.common_iterator % len(self.pics[game]['hq'])]
                pic_lq = self.pics[game]['lq'][self.common_iterator % len(self.pics[game]['lq'])]

                if self.augment == True: 
                    pic_hq, pic_lq = random_transform(pic_hq, pic_lq)

                pic_hq = pic_hq / 255.0
                pic_lq = pic_lq / 255.0

                hq.append(pic_hq)
                lq.append(pic_lq)

            self.common_iterator = self.common_iterator + 1

        return np.array(lq), np.array(hq)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.common_iterator = index * self.pics_per_game
        lq, hq = self.__data_generation()
        return lq, hq

    def on_epoch_end(self):
        self.common_iterator = 0
