from argparse import ArgumentParser
from math import ceil, floor
from os.path import basename
from time import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

from impl.architectures import FSRCNN, SRCNN, ESRGAN_gen

#gpu = tf.config.experimental.list_physical_devices('GPU')[0]
#tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

@tf.function
def get_sr(input):
    return model(input)

if __name__ == '__main__':
    parser = ArgumentParser(description='Demo program.')
    parser.add_argument('model_path', help="Path to the model.")
    parser.add_argument('pic_path', help="Path to the input picture.")
    parser.add_argument('--tile_size', type=int, default=32, help='')
    parser.add_argument('--discard', type=int, default=2, help='N° pixels to discard from border (at output).')
    parser.add_argument('--batch_size', type=int, default=16, help='N° of patches to process at a time.')
    parser.add_argument('--time', action='store_true', help='Measure time per frame.')
    args = parser.parse_args()

    if args.discard % 2 != 0:
        print('Please choose an even number of pixels to discard at output.')
        exit()

    pic_lq = cv2.imread(args.pic_path)

    model_family = basename(args.model_path).split('_')[0]
    
    if model_family == 'srcnn':
        model = SRCNN()
    elif model_family == 'fsrcnn':
        model = FSRCNN()
    else:
        model = ESRGAN_gen()

    model.load_weights(args.model_path)

    total_discard = args.discard
    if model_family == 'srcnn':
        total_discard = total_discard + 8

    out_size = 2 * args.tile_size - 2 * total_discard

    h_pad = w_pad = int(total_discard / 2)

    if pic_lq.shape[0] % (out_size / 2) != 0:
        h_pad = h_pad + ((out_size / 2) - (pic_lq.shape[0] % (out_size / 2))) / 2
    if pic_lq.shape[1] % (out_size / 2) != 0:
        w_pad = w_pad + ((out_size / 2) - (pic_lq.shape[1] % (out_size / 2))) / 2

    pic_lq = cv2.copyMakeBorder(pic_lq, floor(h_pad), ceil(h_pad), floor(w_pad), ceil(w_pad),
                                borderType=cv2.BORDER_REFLECT)
    
    tiles = []
    for i in range(0, pic_lq.shape[0] - args.tile_size + 1, int(out_size / 2)):
        for j in range(0, pic_lq.shape[1] - args.tile_size + 1, int(out_size / 2)):
            patch = pic_lq[i:i+args.tile_size,j:j+args.tile_size]
            tiles.append(patch / 255.0)

    tiles = np.array(tiles)
    tiles_sr = np.empty((len(tiles), (args.tile_size - total_discard) * 2, (args.tile_size - total_discard) * 2, 3), dtype=np.float32)

    print(f'Padded shape: {pic_lq.shape}')
    print(f'Num tiles: {len(tiles)}')
    print(f'Input stride: {out_size/2}')

    total_time = 0

    iter = 1
    if args.time:
        iter = 101

    for i in range(0, iter):
        start = time()

        for j in range(0, len(tiles), args.batch_size):
            if args.discard != 0:
                tiles_sr[j:j+args.batch_size] = get_sr(tiles[j:j+args.batch_size])[:,args.discard:-args.discard,args.discard:-args.discard,:]
            else:
                tiles_sr[j:j+args.batch_size] = get_sr(tiles[j:j+args.batch_size])

        if i != 0:
            total_time = total_time + time() - start

    if args.time:
        print(f'Mean time: {total_time / 100}')

    tiles_sr = np.clip(tiles_sr * 255.0, 0, 255)

    pic_sr = np.zeros((2 * (pic_lq.shape[0] - total_discard), 2 * (pic_lq.shape[1] - total_discard), 3), dtype=np.uint8)

    tile_iter = 0
    for i in range(0, pic_sr.shape[0] - out_size + 1, out_size):
        for j in range(0, pic_sr.shape[1] - out_size + 1, out_size):
            pic_sr[i:i+out_size, j:j+out_size] = tiles_sr[tile_iter]
            tile_iter = tile_iter + 1

    if h_pad > total_discard:
        pic_sr = pic_sr[2 * floor(h_pad) - total_discard:-(2 * ceil(h_pad) - total_discard), :]

    if w_pad > total_discard:
        pic_sr = pic_sr[:, 2 * floor(w_pad) - total_discard:-(2 * ceil(w_pad) - total_discard)]

    cv2.imwrite('output.png', pic_sr)