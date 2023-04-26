# -*- coding: utf-8 -*-

from cv2 import imread
from numpy import square, subtract

def mse_pics(a_path, b):
    a = imread(a_path)
    return square(subtract(a, b)).mean()

if __name__ == '__main__':

    from cv2 import resize, INTER_AREA

    import argparse
    import os
    from itertools import repeat
    from math import ceil, floor
    from multiprocessing import Pool, cpu_count

    import pandas as pd
    
    parser = argparse.ArgumentParser(description='Get patches from pictures.')
    parser.add_argument('data_dir', default = './', help = 'Directory containing training data.')
    args = parser.parse_args()

    source_dir = args.data_dir + '/original'

    check_num = 120 # Roughly 2 seconds
    frame_corr = []

    mp_pool = Pool(cpu_count())

    for game in os.listdir(source_dir):
        hq_pics = [hq for hq in os.listdir(source_dir + '/' + game + '/hq') if hq.endswith('.png')]
        lq_pics = [lq for lq in os.listdir(source_dir + '/' + game + '/lq') if lq.endswith('.png')]

        ratio = (len(lq_pics) - 1) / (len(hq_pics) - 1)

        for i in range(len(hq_pics)):
            print(game + ': ' + str(i + 1) + '/' + str(len(hq_pics)), end = '\r', flush = True)

            hq = imread(source_dir + '/' + game + '/hq/' + hq_pics[i])
            hq_scaled = resize(hq, (1280, 720), interpolation = INTER_AREA)

            min_pos = floor(max((i * ratio) - check_num, 0))
            max_pos = ceil(min((i * ratio) + check_num, len(lq_pics)))
            to_check = [source_dir + '/' + game + '/lq/' + lq_pics[j] for j in range(min_pos, max_pos)]

            mses = mp_pool.starmap(mse_pics, zip(to_check, repeat(hq_scaled)))

            best_match = floor(max((i * ratio) - check_num, 0)) + mses.index(min(mses))
            frame_corr.append([game, hq_pics[i], lq_pics[best_match], min(mses)])

    mp_pool.close()
    mp_pool.join()

    frame_df = pd.DataFrame(frame_corr, columns = ['game', 'hq', 'lq', 'mse'])

    # Remove repeated
    frame_df.sort_values(by = ['game', 'lq', 'mse'], inplace = True)
    frame_df.drop_duplicates(subset = ['game', 'lq'], inplace = True)
    frame_df.to_csv(source_dir + '/frames.csv', index = False)

    frames_per_game = 1000
    selected_frames = []

    # Sample frames
    for game in frame_df['game'].unique():
        all_frames = frame_df[frame_df['game'] == game].sort_values(by = ['hq']).reset_index()
        ratio = (len(all_frames) - 1) / (frames_per_game - 1)

        for i in range(0, frames_per_game):
            row = all_frames.iloc[int(round(i * ratio))]
            selected_frames.append([row['game'], row['hq'], row['lq'], row['mse']])

    selected_df = pd.DataFrame(selected_frames, columns = ['game', 'hq', 'lq', 'mse'])
    selected_df.to_csv(source_dir + '/selected.csv', index = False)
