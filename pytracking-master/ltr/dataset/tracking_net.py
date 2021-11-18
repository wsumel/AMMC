import torch
import os
import os.path
import numpy as np
import pandas
import random
from tqdm import tqdm
from collections import OrderedDict
from ltr.data.image_loader import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from ltr.admin.environment import env_settings
from .generate_vedio import Motion

def list_sequences(root, set_ids):
    """ Lists all the videos in the input set_ids. Returns a list of tuples (set_id, video_name)

    args:
        root: Root directory to TrackingNet
        set_ids: Sets (0-11) which are to be used

    returns:
        list - list of tuples (set_id, video_name) containing the set_id and video_name for each sequence
    """
    sequence_list = []

    for s in set_ids:
        anno_dir = os.path.join(root, "TRAIN_" + str(s), "anno")

        sequences_cur_set = [(s, os.path.splitext(f)[0]) for f in os.listdir(anno_dir) if f.endswith('.txt')]
        sequence_list += sequences_cur_set

    return sequence_list


class TrackingNet(BaseVideoDataset):
    """ TrackingNet dataset.

    Publication:
        TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild.
        Matthias Mueller,Adel Bibi, Silvio Giancola, Salman Al-Subaihi and Bernard Ghanem
        ECCV, 2018
        https://ivul.kaust.edu.sa/Documents/Publications/2018/TrackingNet%20A%20Large%20Scale%20Dataset%20and%20Benchmark%20for%20Object%20Tracking%20in%20the%20Wild.pdf

    Download the dataset using the toolkit https://github.com/SilvioGiancola/TrackingNet-devkit.
    """
    def __init__(self, root=None, image_loader=jpeg4py_loader, set_ids=None, data_fraction=None):
        """
        args:
            root        - The path to the TrackingNet folder, containing the training sets.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            set_ids (None) - List containing the ids of the TrackingNet sets to be used for training. If None, all the
                            sets (0 - 11) will be used.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().trackingnet_dir if root is None else root
        super().__init__('TrackingNet', root, image_loader)

        if set_ids is None:
            set_ids = [i for i in range(12)]

        self.set_ids = set_ids

        # Keep a list of all videos. Sequence list is a list of tuples (set_id, video_name) containing the set_id and
        # video_name for each sequence
        self.sequence_list = list_sequences(self.root, self.set_ids)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        #21 个类别
        self.seq_to_class_map, self.seq_per_class = self._load_class_info()

        # we do not have the class_lists for the tracking net
        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

        self.sample_dict = self.build_sample_dict()

    def build_sample_dict(self):
        sample_frame_number = env_settings().sample_frame_number
        sampled_dict = {}
        # for seq_id in tqdm(range(len(self.sequence_list))):
        for seq_id in range(len(self.sequence_list)):
            seq_info_dict = self.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']
            valid_ids = [i for i in range(len(visible)) if visible[i]]
            if len(valid_ids) != 0:
                if len(valid_ids) >= sample_frame_number:
                    sampled_frames = list(np.random.choice(valid_ids,size=sample_frame_number,replace=False))
                else:
                    sampled_frames = list(np.random.choice(valid_ids,size=sample_frame_number - len(valid_ids),replace=True))
                    sampled_frames  = sampled_frames + valid_ids      
                sampled_dict[seq_id] = sampled_frames
            else:
                sampled_dict[seq_id] = []
        print("trackingnet[%d] sample %d frames per sequence done!"%(len(self.sequence_list),sample_frame_number))
        return sampled_dict

    def _load_class_info(self):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        class_map_path = os.path.join(ltr_path, 'data_specs', 'trackingnet_classmap.txt')

        with open(class_map_path, 'r') as f:
            seq_to_class_map = {seq_class.split('\t')[0]: seq_class.rstrip().split('\t')[1] for seq_class in f}

        seq_per_class = {}
        for i, seq in enumerate(self.sequence_list):
            class_name = seq_to_class_map.get(seq[1], 'Unknown')
            if class_name not in seq_per_class:
                seq_per_class[class_name] = [i]
            else:
                seq_per_class[class_name].append(i)

        return seq_to_class_map, seq_per_class

    def get_name(self):
        return 'trackingnet'

    def has_class_info(self):
        return True

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_id):
        set_id = self.sequence_list[seq_id][0]
        vid_name = self.sequence_list[seq_id][1]
        bb_anno_file = os.path.join(self.root, "TRAIN_" + str(set_id), "anno", vid_name + ".txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        bbox = self._read_bb_anno(seq_id)
        #add here 
        h,w = self._get_frame(seq_id,0).shape[:2]
        # valid1 = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        valid2 = (bbox[:, 2] > 0) & (bbox[:, 3] > 0) & (bbox[:, 0] < w) & (bbox[:, 1] < h) & (bbox[:, 0]+bbox[:, 2]>0) & (bbox[:, 1]+bbox[:, 3]>0)
        # for i,[a,b] in enumerate(zip(valid1,valid2)):
        #     if a!=b:
        #         print(bbox[i])
        visible = valid2.clone().byte()
        return {'bbox': bbox, 'valid': valid2, 'visible': visible}
        
    def _get_sequence_path(self, seq_id):
        set_id = self.sequence_list[seq_id][0]
        vid_name = self.sequence_list[seq_id][1]
        return os.path.join(self.root, "TRAIN_" + str(set_id), "frames", vid_name)
         

    def _get_frame(self, seq_id, frame_id):
        set_id = self.sequence_list[seq_id][0]
        vid_name = self.sequence_list[seq_id][1]
        frame_path = os.path.join(self.root, "TRAIN_" + str(set_id), "frames", vid_name, str(frame_id) + ".jpg")
        return self.image_loader(frame_path)

    def _get_class(self, seq_id):
        seq_name = self.sequence_list[seq_id][1]
        return self.seq_to_class_map[seq_name]

    def get_class_name(self, seq_id):
        obj_class = self._get_class(seq_id)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None,dtype='train'):
        frame_list = [self._get_frame(seq_id, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        obj_class = self._get_class(seq_id)

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})


        return frame_list, anno_frames, object_meta 
        # if 'train' in dtype:
        #     return frame_list, anno_frames, object_meta
        
        # for i in range(len(frame_list)):
        #     if random.uniform(0,1) < 0.2:
        #         img = frame_list[i]
        #         anno = anno_frames['bbox'][i].numpy()
        #         # draw_gt(img,gt=anno,savefig_path='/home/user-njf87/hl/workspace3/figs3/org.jpg')
        #         motion_generator = Motion(np.copy(img),np.copy(anno))
        #         im_new,gt_new = motion_generator()
        #         # draw_gt(im_new,gt=gt_new,savefig_path='/home/user-njf87/hl/workspace3/figs3/trans.jpg')
        #         frame_list[i] = np.copy(im_new)
        #         gt_new = np.copy(gt_new.astype(np.float32))
        #         anno_frames['bbox'][i] = torch.from_numpy(gt_new)
            
        # return frame_list, anno_frames, object_meta
