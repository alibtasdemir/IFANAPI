import os
import torch


class CKPT_Manager:
    def __init__(self, root_dir, model_name, cuda, max_files_to_keep=10, is_descending=False):
        self.root_dir = root_dir
        self.root_dir_ckpt = os.path.join(root_dir, 'ckpt')
        self.root_dir_state = os.path.join(root_dir, 'state')
        self.cuda = cuda

        self.model_name = model_name
        self.max_files = max_files_to_keep

        self.ckpt_list = os.path.join(self.root_dir, 'checkpoints.txt')
        self.is_descending = is_descending

    def load_ckpt(self, network, by_score=True, name=None, abs_name=None, epoch=None):
        # get ckpt path
        if name is None and abs_name is None and epoch is None:
            try:
                with open(self.ckpt_list, 'r') as file:
                    lines = file.read().splitlines()
                    file.close()
            except:
                print('ckpt_list does not exists')
                return

            if by_score:
                file_name = lines[0].split(' ')[0]
            else:
                file_name = lines[-1].split(' ')[0]

            file_path = os.path.join(self.root_dir_ckpt, file_name)
        else:
            if name is not None:
                file_name = name
                file_path = os.path.join(self.root_dir_ckpt, file_name)
            if abs_name is not None:
                file_name = os.path.basename(abs_name)
                file_path = abs_name
            if epoch is not None:
                file_name = '{}_{:05d}.pytorch'.format(self.model_name, epoch)
                file_path = os.path.join(self.root_dir_ckpt, file_name)

        if self.cuda is False:
            state_dict = torch.load(file_path, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.split('.', 1)[-1]
                new_state_dict[k] = v
            return network.load_state_dict(new_state_dict, strict=False), os.path.basename(file_name)

        else:
            device_id = torch.cuda.current_device()
            return network.load_state_dict(
                torch.load(file_path, map_location="cuda:{}".format(device_id) if self.cuda else "cpu"), strict=False
                ), os.path.basename(file_name)
