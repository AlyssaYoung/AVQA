# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import numpy as np
import json
import pickle
import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab


class AVQADataset(Dataset):

    def __init__(self, answers, ans_candidates, ans_candidates_len, questions, questions_len, video_ids, q_ids,
                 app_feature_h5, app_feat_id_to_index, motion_feature_h5, motion_feat_id_to_index, vlaudio_feature_h5, 
                 vlaudio_feat_id_to_index, question_category, useAudio=True, ablation='none'):
        # convert data to tensor
        self.useAudio = useAudio
        self.ablation = ablation
        self.all_answers = answers
        self.question_category = question_category
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.all_q_ids = q_ids
        self.app_feature_h5 = app_feature_h5
        self.motion_feature_h5 = motion_feature_h5
        self.vlaudio_feature_h5 = vlaudio_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.motion_feat_id_to_index = motion_feat_id_to_index
        self.vlaudio_feat_id_to_index = vlaudio_feat_id_to_index

        self.f_app = h5py.File(self.app_feature_h5, 'r')
        self.f_motion = h5py.File(self.motion_feature_h5, 'r')
        self.f_vlaudio = h5py.File(self.vlaudio_feature_h5, 'r')
        self.app_feature = self.f_app['resnet_features']
        self.motion_feature = self.f_motion['resnext_features']
        self.vlaudio_feature = self.f_vlaudio['vlaudio_features']

        if not np.any(ans_candidates):
            self.question_type = 'openended'
        else:
            self.question_type = 'mulchoices'
            self.all_ans_candidates = torch.LongTensor(np.asarray(ans_candidates))
            self.all_ans_candidates_len = torch.LongTensor(np.asarray(ans_candidates_len))

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        ans_candidates = torch.zeros(4)
        ans_candidates_len = torch.zeros(4)
        if self.question_type == 'mulchoices':
            ans_candidates = self.all_ans_candidates[index]
            ans_candidates_len = self.all_ans_candidates_len[index]
        question = self.all_questions[index]
        question_category = self.question_category[index] if self.question_category is not None else None
        question_len = self.all_questions_len[index]
        video_idx = self.all_video_ids[index].item()
        question_idx = self.all_q_ids[index]
        app_index = self.app_feat_id_to_index[str(video_idx)]
        motion_index = self.motion_feat_id_to_index[str(video_idx)]
        vlaudio_index = self.vlaudio_feat_id_to_index[str(video_idx)]

        appearance_feat = self.app_feature[app_index]  # (8, 16, 2048)
        motion_feat = self.motion_feature[motion_index]  # (8, 2048)
        vlaudio_feat = self.vlaudio_feature[vlaudio_index] # (2048,)

        appearance_feat = torch.from_numpy(appearance_feat)
        motion_feat = torch.from_numpy(motion_feat)
        vl_audio_feat = torch.from_numpy(vlaudio_feat)

        if self.ablation == 'ques':
            appearance_feat = torch.zeros_like(appearance_feat)
            motion_feat = torch.zeros_like(motion_feat)
            vl_audio_feat = torch.zeros_like(vl_audio_feat)
            
        elif self.ablation == 'app':
            question = torch.zeros_like(question)
            motion_feat = torch.zeros_like(motion_feat)
            vl_audio_feat = torch.zeros_like(vl_audio_feat)
            
        elif self.ablation == 'mot':
            question = torch.zeros_like(question)
            appearance_feat = torch.zeros_like(appearance_feat)
            vl_audio_feat = torch.zeros_like(vl_audio_feat)
            
        elif self.ablation == 'app_mot':
            question = torch.zeros_like(question)
            vl_audio_feat = torch.zeros_like(vl_audio_feat)
            
        elif self.ablation == 'aud':
            question = torch.zeros_like(question)
            appearance_feat = torch.zeros_like(appearance_feat)
            motion_feat = torch.zeros_like(motion_feat)
            
        elif self.ablation == 'app_ques':
            motion_feat = torch.zeros_like(motion_feat)
            vl_audio_feat = torch.zeros_like(vl_audio_feat)
            
        elif self.ablation == 'mot_ques':
            appearance_feat = torch.zeros_like(appearance_feat)
            vl_audio_feat = torch.zeros_like(vl_audio_feat)
            
        elif self.ablation == 'app_mot_ques':
            vl_audio_feat = torch.zeros_like(vl_audio_feat)
            
        elif self.ablation == 'aud_ques':
            appearance_feat = torch.zeros_like(appearance_feat)
            motion_feat = torch.zeros_like(motion_feat)
            
        
        if self.useAudio:
            return (
                video_idx, question_idx, question_category, answer, ans_candidates, ans_candidates_len, appearance_feat, 
                motion_feat, vl_audio_feat, question, question_len)
        else:
            return (
                video_idx, question_idx, question_category, answer, ans_candidates, ans_candidates_len, appearance_feat, motion_feat, question,
                question_len)

    def __len__(self):
        return len(self.all_questions)


class AVQADataLoader(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        question_type = kwargs.pop('question_type')
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            question_category = obj['question_category']
            questions_len = obj['questions_len']
            video_ids = obj['video_ids']
            q_ids = obj['question_id']
            answers = obj['answers']
            glove_matrix = obj['glove']
            
            ans_candidates = np.zeros(4)
            ans_candidates_len = np.zeros(4)
            ans_candidates = obj['ans_candidates']
            ans_candidates_len = obj['ans_candidates_len']

        if 'train_num' in kwargs:
            trained_num = kwargs.pop('train_num')
            if trained_num > 0:
                questions = questions[:trained_num]
                question_category = question_category[:trained_num]
                questions_len = questions_len[:trained_num]
                video_ids = video_ids[:trained_num]
                q_ids = q_ids[:trained_num]
                answers = answers[:trained_num]
                ans_candidates = ans_candidates[:trained_num]
                ans_candidates_len = ans_candidates_len[:trained_num]
        if 'val_num' in kwargs:
            val_num = kwargs.pop('val_num')
            if val_num > 0:
                questions = questions[:val_num]
                question_category = question_category[:val_num]
                questions_len = questions_len[:val_num]
                video_ids = video_ids[:val_num]
                q_ids = q_ids[:val_num]
                answers = answers[:val_num]
                ans_candidates = ans_candidates[:val_num]
                ans_candidates_len = ans_candidates_len[:val_num]
        if 'test_num' in kwargs:
            test_num = kwargs.pop('test_num')
            if test_num > 0:
                questions = questions[:test_num]
                question_category = question_category[:test_num]
                questions_len = questions_len[:test_num]
                video_ids = video_ids[:test_num]
                q_ids = q_ids[:test_num]
                answers = answers[:test_num]
                ans_candidates = ans_candidates[:test_num]
                ans_candidates_len = ans_candidates_len[:test_num]

        print('loading appearance feature from %s' % (kwargs['appearance_feat']))
        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]
        
        print('loading motion feature from %s' % (kwargs['motion_feat']))
        with h5py.File(kwargs['motion_feat'], 'r') as motion_features_file:
            motion_video_ids = motion_features_file['ids'][()]
        
        print('loading video level audio feature from %s' % (kwargs['vl_audio_feat']))
        with h5py.File(kwargs['vl_audio_feat'], 'r') as vlaudio_features_file:
            vlaudio_video_ids = vlaudio_features_file['ids'][()]

        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}
        motion_feat_id_to_index = {str(id): i for i, id in enumerate(motion_video_ids)}
        vlaudio_feat_id_to_index = {str(id): i for i, id in enumerate(vlaudio_video_ids)}

        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.motion_feature_h5 = kwargs.pop('motion_feat')
        self.vlaudio_feature_h5 = kwargs.pop('vl_audio_feat')
        self.useAudio = kwargs.pop('useAudio')
        self.ablation = kwargs.pop('ablation')
        self.dataset = AVQADataset(answers, ans_candidates, ans_candidates_len, questions, questions_len,
                                      video_ids, q_ids,
                                      self.app_feature_h5, app_feat_id_to_index, self.motion_feature_h5,
                                      motion_feat_id_to_index, self.vlaudio_feature_h5, vlaudio_feat_id_to_index, 
                                      question_category, self.useAudio, self.ablation)

        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
