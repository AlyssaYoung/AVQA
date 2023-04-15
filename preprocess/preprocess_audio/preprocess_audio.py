import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
import h5py
import json
import random

from models import Cnn14, Cnn14_DecisionLevelMax
from pytorch_utils import move_data_to_device
from utils import logger


def get_audio_embedding(ckpt_path, feature_type, audio_path, sample_rate, window_size, 
        hop_size, mel_bins, fmin, fmax, classes_num):
    """Inference audio tagging result of an audio clip.
    """
    # Arugments & parameters
    checkpoint_path = ckpt_path
    audio_path = audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    # Model
    if feature_type == 'vlaudio':
        model = Cnn14(sample_rate=sample_rate, window_size=window_size, 
                    hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                    classes_num=classes_num)
    else:
        model = Cnn14_DecisionLevelMax(sample_rate=sample_rate, window_size=window_size, 
                    hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                    classes_num=classes_num)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    # Print embedding
    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        # print('embedding: {}'.format(embedding.shape))

    return embedding

"""
TODO:
2023-04-15: delete the irrelevant notes
"""

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Example of parser. ')

    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000)
    parser.add_argument('--classes_num', type=int, default=527) 
    parser.add_argument('--clip_size', type=int, default=8)
    parser.add_argument('--feature_type', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--cuda', action='store_true', default=True)
    # output
    # parser.add_argument('--out', type=str, help='output filepath', default="{}_{}_{}_feat.h5")
    parser.add_argument('--outfile', type=str, help='output filepath', required=True)
    parser.add_argument('--logfile', type=str, help='logger filepath', default="log.txt")
    
    args = parser.parse_args()

    with open('video_map.json','r') as f:
        name_id_pairs = json.load(f)

    audio_paths = [(os.path.join(args.audio_path, key+'.wav'), name_id_pairs[key]) for key in name_id_pairs]
    random.shuffle(audio_paths)
    
    dataset_size = len(audio_paths)

    
    # outfile = args.out.format('vggsound-qa', args.feature_type, 'PANNs')
    logger = logger.Logger(args.logfile)
    
    with h5py.File(args.outfile, 'w') as fd:
        feat_dset = None
        video_ids_dset = None
        i0 = 0
        for i, (video_path, video_id) in enumerate(audio_paths):
            audio_path = video_path
            if args.feature_type == 'vlaudio':
                audio_embedding = get_audio_embedding(args.checkpoint_path, args.feature_type, audio_path, args.sample_rate, args.window_size, 
                                                    args.hop_size, args.mel_bins, args.fmin, args.fmax, args.classes_num)
                audio_embedding = np.asarray(audio_embedding)
                # videowise_audio_embedding = videowise_audio_embedding[np.newaxis,:]
                # print(audio_embedding.shape)
                dim = audio_embedding.shape[0]

                with torch.no_grad():
                    if feat_dset is None:
                        feat_dset = fd.create_dataset('vlaudio_features', (dataset_size, dim),
                                                  dtype=np.float32)
                        video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)
            else:
                audio_embedding = get_audio_embedding(args.checkpoint_path, args.feature_type, audio_path, args.sample_rate, args.window_size, 
                                                    args.hop_size, args.mel_bins, args.fmin, args.fmax, args.classes_num)
                audio_embedding = np.asarray(audio_embedding)
                # sample N clips as clipwise embedding
                # (?, 2048) to (8, 2048)
                # print(audio_embedding.shape)
                dim0, dim1 = audio_embedding.shape
                if dim0 < args.clip_size:
                    print(audio_path, video_id, dim0)
                    logger.write('\t%s %d %d' % (audio_path, video_id, dim0))
                    ratio = args.clip_size // dim0 
                    # upsampled_embedding = audio_embedding[:, None, :].repeat(1, ratio, 1)
                    # upsampled_embedding = upsampled_embedding.reshape(dim0 * ratio, dim1)
                    upsampled_embedding = np.repeat(audio_embedding, ratio, axis=0)
                    print(upsampled_embedding.shape)
                    assert upsampled_embedding.shape[0] == dim0 * ratio
                    # pad = upsampled_embedding[-1 :, :].repeat(args.clip_size - dim0, 1)
                    pad = np.repeat(upsampled_embedding[-1:, :], args.clip_size - upsampled_embedding.shape[0], axis=0)
                    upsampled_embedding = np.concatenate((upsampled_embedding, pad), axis=0)
                    print('upsampled_embedding shape:', upsampled_embedding.shape)
                    assert upsampled_embedding.shape[0] == args.clip_size
                    audio_embedding = upsampled_embedding

                else:
                    sampled_indices = np.linspace(0, dim0-1, num=args.clip_size, endpoint=True, retstep=False, dtype=np.int)
                    # print(sampled_indices)
                    sampled_embedding = audio_embedding[sampled_indices]
                    # print(sampled_embedding.shape)
                    audio_embedding = sampled_embedding

                with torch.no_grad():
                    if feat_dset is None:
                        feat_dset = fd.create_dataset('claudio_features', (dataset_size, args.clip_size, dim1),
                                                  dtype=np.float32)
                        video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)

            i1 = i0 + 1

            try:
                feat_dset[i0:i1] = audio_embedding
            except:
                print('except information:',feat_dset[i0:i1].shape)

            video_ids_dset[i0:i1] = video_id
            i0 = i1
            
            print('{:d}/{:d}'.format(i1, dataset_size))
