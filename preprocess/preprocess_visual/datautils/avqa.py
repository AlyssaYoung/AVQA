import os
import json
from datautils import utils
import nltk
import pickle
import numpy as np

QUESTION_CATEGORY_DICT = {'Which':0,'Come From':1,'Happening':2,'Where':3,'Why':4, 'Before Next':5, 'When': 6, 'Used For':7}

def load_video_paths(args):
    ''' Load a list of (path,image_id tuples).'''
    video_paths = []
    video_names = []
    modes = ['train', 'val']
    for mode in modes:
       with open(args.annotation_file.format(mode), 'r') as anno_file:
           instances = json.load(anno_file)
       [video_names.append(instance['video_name']) for instance in instances]
    video_names = set(video_names)

    with open(args.video_name_mapping, 'r') as f:
        video_dict = json.load(f)

    for video_name in video_names:
        video_paths.append(((os.path.join(args.video_path + '{}.mp4'.format(video_name))), video_dict[video_name]))
    return video_paths