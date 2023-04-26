import argparse
import os
import shutil

import cv2
import numpy as np
from scipy.ndimage import sobel

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Split video in frames.')
    parser.add_argument('video_path', help = 'Path to the target video.')
    parser.add_argument('--min_score', type = float, default = 0.25, help = 'Minimum score for Sobel filtering.')
    parser.add_argument('--max_score', type = float, default = 0.75, help = 'Maximum score for Sobel filtering.')
    args = parser.parse_args()

    target_dir = os.path.splitext(args.video_path)[0]

    if os.path.isdir(target_dir) == True:
        shutil.rmtree(target_dir, ignore_errors = True)    
    os.mkdir(target_dir)

    cap = cv2.VideoCapture(args.video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    i = j = 0

    while True:
        ok, frame = cap.read()

        if ok:
            sobel_pic = sobel(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            border_score = np.mean(sobel_pic / 255.0)

            if (border_score > args.min_score) and (border_score < args.max_score):
                name = target_dir + '/%06d' % i + '.png'
                cv2.imwrite(name, frame)
                i = i + 1
        else:
            break

        j = j + 1
        print(str(j + 1) + '/' + str(length), end = '\r', flush = True)

    cap.release()
