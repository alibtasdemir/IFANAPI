import torch
from easydict import EasyDict as edict
import cv2
import os
import time
import numpy as np

from models import create_model
from ckpt_manager import CKPT_Manager

from utils import toYellow, toGreen


# Define settings for the IFAN model
def createConfig():
    cfg = edict()
    cfg.mode = "IFAN"
    cfg.is_train = False
    cfg.is_verbose = False
    cfg.save_sample = False

    # Change this to False if you use cpu
    cfg.cuda = True
    if cfg.cuda:
        cfg.device = 'cuda'
    else:
        cfg.device = 'cpu'

    cfg.dist = False

    cfg.trainer = 'trainer'
    cfg.network = 'IFAN'

    # Network parameters like channel size, input bits etc.
    cfg.ks = 3
    cfg.ch = 32
    cfg.res_num = 2
    cfg.in_bit = 8
    cfg.norm_val = (2 ** cfg.in_bit - 1)

    cfg.Fs = 3  # filter size
    cfg.N = 17
    cfg.refine_val = 8

    # weight init
    cfg.wiF = 1.5

    return cfg


def runTest():
    # To show the results on the screen. True for debugging purposes.
    show_on_screen = False
    # Define input/output folders
    input_offset = 'input_images/'
    results_dir = 'result_images/'

    config = createConfig()

    # Create model with the setting in config
    model = create_model(config)
    # Model to evaluation mode
    network = model.get_network().to().eval()

    # Restore the saved weights to the model
    ckpt_manager = CKPT_Manager(root_dir='', model_name="IFAN", cuda=config.cuda)
    load_state, ckpt_name = ckpt_manager.load_ckpt(network, abs_name='./ckpt/{}.pytorch'.format("IFAN"))
    print('\nLoading checkpoint \'{}\' on model \'{}\': {}\n'.format(ckpt_name, config.mode, load_state))

    from data_loader.utils import load_file_list, refine_image, read_frame
    # load inputs
    print(toGreen('Reading Input(s)...'))
    _, input_path_list, _ = load_file_list(input_offset, None, is_flatten=True)

    print(toYellow('\n====== DEMO START ======'))
    max_side = 1920
    runtime = 0
    for i, input_path in enumerate(input_path_list):
        C_cpu = read_frame(input_path, config.norm_val, None)
        # resize image if max side exceeds 1920 (due to GPU mem)
        b, h, w, c = C_cpu.shape
        if max(h, w) > max_side:
            scale_ratio = max_side / max(h, w)
            C_cpu = np.expand_dims(
                cv2.resize(C_cpu[0], dsize=(int(w * scale_ratio), int(h * scale_ratio)), interpolation=cv2.INTER_AREA),
                0
            )

        C = torch.FloatTensor(refine_image(C_cpu, config.refine_val).transpose(0, 3, 1, 2).copy())
        if config.cuda:
            C = C.cuda()

        # running network
        init_time = time.time()
        with torch.no_grad():
            out = network(C, is_train=False)
        itr_time = time.time() - init_time
        print(
            toGreen(
                '\n[EVAL {}][{:02}/{:02}] {}'.format(
                    config.mode, i + 1, len(input_path_list), os.path.basename(input_path)
                )
            )
        )
        savepath = os.path.join(results_dir, os.path.basename(input_path))
        print(savepath)
        # display
        input_cpu = C_cpu[0]
        output = out['result']
        output_cpu = output.cpu().numpy()[0].transpose(1, 2, 0)

        before_im = np.flip(input_cpu * 255., 2).astype(np.uint8)
        after_im = np.flip(output_cpu * 255., 2).astype(np.uint8)
        print(toYellow("\tin {0:.03f} seconds".format(itr_time)))
        runtime = itr_time
        if show_on_screen:
            print('Input')
            imbefore = ResizeWithAspectRatio(before_im, width=800)
            imafter = ResizeWithAspectRatio(after_im, width=800)
            cv2.imshow("", imbefore)
            cv2.waitKey(0)
            print('\nOutput')
            cv2.imshow("", imafter)
            cv2.waitKey(0)
        cv2.imwrite(savepath, after_im)
    return round(runtime, 3)


# Resize the image with aspect ratio. (Due to memory limitations)
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


if __name__ == '__main__':
    runTest()
