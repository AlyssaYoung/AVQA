import argparse, os
import h5py
import skvideo.io
from PIL import Image

import torch
from torch import nn
import torchvision
import random
import numpy as np

from models import resnext
from datautils import utils
from datautils import avqa


def build_resnet():
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    cnn = getattr(torchvision.models, args.model)(pretrained=True)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    model.cuda()
    model.eval()
    return model

def build_resnext():
    model = resnext.resnet101(num_classes=400, shortcut_type='B', cardinality=32,
                              sample_size=112, sample_duration=16,
                              last_fc=False)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    assert os.path.exists('./pretrained/resnext-101-kinetics.pth')
    model_data = torch.load('./pretrained/resnext-101-kinetics.pth', map_location='cpu')
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    return model


def run_batch(cur_batch, model):
    """
    Args:
        cur_batch: treat a video as a batch of images
        model: ResNet model for feature extraction
    Returns:
        ResNet extracted feature.
    """
    mean = np.array(args.mean).reshape(1, 3, 1, 1)
    std = np.array(args.std).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std

    image_batch = torch.FloatTensor(image_batch).cuda()
    with torch.no_grad():
        #image_batch = torch.autograd.Variable(image_batch)

        feats = model(image_batch)
        
        feats = feats.detach().cpu().clone().numpy()

    return feats


def extract_clips_with_consecutive_frames(path, num_clips, num_frames_per_clip):
    """
    Args:
        path: path of a video
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
    Returns:
        A list of raw features of clips.
    """
    valid = True
    clips = list()
    try:
        video_data = skvideo.io.vread(path)
    except:
        print('file {} error'.format(path))
        valid = False
        if args.model in motion_model_list:
            return list(np.zeros(shape=(num_clips, 3, num_frames_per_clip, 112, 112))), valid
        else:
            return list(np.zeros(shape=(num_clips, num_frames_per_clip, 3, 224, 224))), valid
    total_frames = video_data.shape[0]
    img_size = (args.image_height, args.image_width)
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1:num_clips + 1]:
        if num_frames_per_clip % 2 == 0:
            clip_start = int(i) - int(num_frames_per_clip / 2)
            clip_end = int(i) + int(num_frames_per_clip / 2)
        else:
            clip_start = int(i) - int(num_frames_per_clip / 2) - 1
            clip_end = int(i) + int(num_frames_per_clip / 2)
        if clip_start < 0:
            clip_start = 0
        if clip_end > total_frames:
            clip_end = total_frames - 1
        clip = video_data[clip_start:clip_end]
        if clip_start == 0:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_start], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((added_frames, clip), axis=0)
        if clip_end == (total_frames - 1):
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_end], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((clip, added_frames), axis=0)
        new_clip = []
        for j in range(num_frames_per_clip):
            frame_data = clip[j]
            #img = Image.fromarray(frame_data)
            if args.interpolation == 'bicubic':
                img = np.array(Image.fromarray(frame_data).resize(img_size, Image.BICUBIC))
            else:
                img = np.array(Image.fromarray(frame_data).resize(img_size, Image.BILINEAR))
            img = img.transpose(2, 0, 1)[None]
            frame_data = np.array(img)
            new_clip.append(frame_data)
        new_clip = np.asarray(new_clip)  # (num_frames, width, height, channels)
        if args.model in motion_model_list:
            new_clip = np.squeeze(new_clip)
            new_clip = np.transpose(new_clip, axes=(1, 0, 2, 3))
        clips.append(new_clip)
    return clips, valid


def generate_h5(model, video_ids, num_clips, outfile):
    """
    Args:
        model: loaded pretrained model for feature extraction
        video_ids: list of video ids
        num_clips: expected numbers of splitted clips
        outfile: path of output file to be written
    Returns:
        h5 file containing visual features of splitted clips.
    """

    dataset_size = len(video_ids)

    with h5py.File(outfile, 'w') as fd:
        feat_dset = None
        video_ids_dset = None
        i0 = 0
        _t = {'misc': utils.Timer()}
        for i, (video_path, video_id) in enumerate(video_ids):
            _t['misc'].tic()
            clips, valid = extract_clips_with_consecutive_frames(video_path, num_clips=num_clips, num_frames_per_clip=args.num_frames)
            if args.feature_type == 'appearance':
                clip_feat = []
                if valid:
                    for clip_id, clip in enumerate(clips):
                        feats = run_batch(clip, model)  # (16, 2048)
                        feats = feats.squeeze()
                        clip_feat.append(feats)
                else:
                    clip_feat = np.zeros(shape=(num_clips, args.num_frames, 2048))
                clip_feat = np.asarray(clip_feat)  # (8, 16, 2048)
                if feat_dset is None:
                    print(clip_feat.shape)
                    C, F, D = clip_feat.shape
                    feat_dset = fd.create_dataset('resnet_features', (dataset_size, C, F, D),
                                                  dtype=np.float32)
                    video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)
            elif args.feature_type == 'motion':
                clip_torch = torch.FloatTensor(np.asarray(clips)).cuda()
                if valid:
                    with torch.no_grad():
                        clip_feat = model(clip_torch).squeeze()  # (8, 2048)
                        clip_feat = clip_feat.detach().cpu().numpy()
                else:
                    clip_feat = np.zeros(shape=(num_clips, 2048))
                if feat_dset is None:
                    print(valid)
                    print(clip_feat.shape)
                    C, D = clip_feat.shape
                    feat_dset = fd.create_dataset('resnext_features', (dataset_size, C, D),
                                                  dtype=np.float32)
                    video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)

            i1 = i0 + 1
            if clip_feat.shape[0] == 1:
                clip_feat = clip_feat[0]
            try:
                feat_dset[i0:i1] = clip_feat
            except:
                print(feat_dset[i0:i1].shape)
                print(clip_feat.shape)
            video_ids_dset[i0:i1] = video_id
            i0 = i1
            _t['misc'].toc()
            print('{:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                    .format(i1, dataset_size, _t['misc'].average_time,
                            _t['misc'].average_time * (dataset_size - i1) / 3600))


if __name__ == '__main__':

    appearance_model_list = ['resnet50', 'resnet101', 'resnet152']
    motion_model_list = ['resnext101']

    model_list = appearance_model_list + motion_model_list

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='specify which gpu will be used')
    # dataset info
    parser.add_argument('--dataset', default='avqa', type=str)
    parser.add_argument('--question_type', default='none', choices=['frameqa', 'count', 'transition', 'action', 'none'], type=str)
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--video_name_mapping', type=str, required=True)
    parser.add_argument('--annotation_file', type=str, required=True)
    # output
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default="data/feats/{}_{}_{}_feat.h5", type=str)
    # image sizes
    parser.add_argument('--num_clips', default=8, type=int)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--image_height', default=224, type=int)
    parser.add_argument('--image_width', default=224, type=int)
    parser.add_argument('--interpolation', default='bicubic', type=str)

    # network params
    parser.add_argument('--model', default='resnext101', choices=model_list, type=str)
    parser.add_argument('--seed', default='666', type=int, help='random seed')
    args = parser.parse_args()
    if(args.model in appearance_model_list):
        args.feature_type = 'appearance'
    elif(args.model in motion_model_list):
        args.feature_type = 'motion'
    else:
        raise Exception('Feature type not supported!')
    # set gpu
    if args.model != 'resnext101':
        torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Making model ......")    

    if(args.model in ['resnet50', 'resnet101', 'resnet152']):
        args.image_height = 224
        args.image_width = 224
        args.interpolation = 'bilinear'
        args.mean = [0.485, 0.456, 0.406]
        args.std = [0.229, 0.224, 0.224]
        model = build_resnet()
    elif args.model in ['resnext101']:
        args.image_height = 112
        args.image_width = 112
        args.interpolation = 'bilinear'
        args.mean=[114.7748, 107.7354, 99.4750]
        args.std=[1, 1, 1]
        model = build_resnext()

    print("Start extraction ......")

    # annotation files
    # args.annotation_file = '/DATA/DATANAS1/yangpinci/Datasets/VGGSound/VGGSound-QA/annotation/{}_qa.json'
    # args.video_dir = '/DATA/DATANAS1/yangpinci/Datasets/VGGSound/VGGSound-QA/video'
    # args.video_name_mapping = '/DATA/DATANAS1/yangpinci/Datasets/VGGSound/VGGSound-QA/annotation/video_map.json'
    video_paths = avqa.load_video_paths(args)
    random.shuffle(video_paths)

    """
    TODO:
    fix the output h5 file path
    """
    generate_h5(model, video_paths, args.num_clips,
                args.outfile.format(args.dataset, args.feature_type, args.model))
