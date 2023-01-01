import os
import numpy as np
import cv2
from pathlib import Path


def read_frame(path, norm_val=None, rotate=None):
    if norm_val == (2 ** 16 - 1):
        frame = cv2.imread(path, -1)
        if rotate is not None:
            frame = cv2.rotate(frame, rotate)
        frame = frame / norm_val
        frame = frame[..., ::-1]
    else:
        frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if rotate is not None:
            frame = cv2.rotate(frame, rotate)
        frame = frame / 255.

    return np.expand_dims(frame, axis=0)


def refine_image(img, val=16):
    shape = img.shape
    if len(shape) == 4:
        _, h, w, _ = shape[:]
        return img[:, 0: h - h % val, 0: w - w % val, :]
    elif len(shape) == 3:
        h, w = shape[:2]
        return img[0: h - h % val, 0: w - w % val, :]
    elif len(shape) == 2:
        h, w = shape[:2]
        return img[0: h - h % val, 0: w - w % val]


def load_file_list(root_path, child_path=None, is_flatten=False):
    folder_paths = []
    filenames_pure = []
    filenames_structured = []
    num_files = 0
    for root, dirnames, filenames in os.walk(root_path):
        print('Root:', root, ', dirnames:', dirnames, ', filenames', filenames)
        if len(dirnames) != 0:
            if dirnames[0][0] == '@':
                del (dirnames[0])

        if len(dirnames) == 0:
            if root == '.':
                continue
            if child_path is not None and child_path != Path(root).name:
                continue
            folder_paths.append(root)
            filenames_pure = []
            for i in np.arange(len(filenames)):
                if filenames[i][0] != '.' and filenames[i] != 'Thumbs.db':
                    filenames_pure.append(os.path.join(root, filenames[i]))
            # print('filenames_pure:', filenames_pure)
            filenames_structured.append(np.array(sorted(filenames_pure), dtype='str'))
            num_files += len(filenames_pure)

    folder_paths = np.array(folder_paths)
    filenames_structured = np.array(filenames_structured, dtype=object)

    sort_idx = np.argsort(folder_paths)
    folder_paths = folder_paths[sort_idx]
    filenames_structured = filenames_structured[sort_idx]

    if is_flatten:
        if len(filenames_structured) > 1:
            filenames_structured = np.concatenate(filenames_structured).ravel()
        else:
            filenames_structured = filenames_structured.flatten()

    return folder_paths, filenames_structured, num_files
