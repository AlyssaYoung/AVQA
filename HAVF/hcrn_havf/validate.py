import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import os, sys
import json
import pickle
from termcolor import colored

from DataLoader import AVQADataLoader
from utils.utils import todevice

import model.HCRN as HCRN

from configs.config import cfg, cfg_from_file

QUESTION_CATEGORY = {0:'Which',1:'Come From',2:'Happening',3:'Where',4:'Why', 5:'Before Next', 6:'When', 7:'Used For'}

def validate(cfg, model, data, device, write_preds=False):
    model.eval()
    print('validating...')
    total_acc, count = 0.0, 0
    which_acc, comefrom_acc, happening_acc, where_acc, why_acc, beforenext_acc, when_acc, usedfor_acc = 0.,0.,0.,0.,0.,0.,0.,0.
    which_count, comefrom_count, happening_count, where_count, why_count, beforenext_count, when_count, usedfor_count = 0,0,0,0,0,0,0,0
    all_preds = []
    gts = []
    v_ids = []
    q_ids = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            # video_ids, question_ids, answers, *batch_input = [todevice(x, device) for x in batch]
            video_ids, question_ids, question_categories, answers, *batch_input = [todevice(x, device) for x in batch]
            
            if cfg.train.batch_size == 1:
                answers = answers.to(device)
            else:
                answers = answers.to(device).squeeze()
            batch_size = answers.size(0)
            logits = model(*batch_input).to(device)

            which_idx = []
            comefrom_idx = []
            happening_idx = []
            where_idx = []
            why_idx = []
            beforenext_idx = []
            when_idx = []
            usedfor_idx = []
            for i, category in enumerate(question_categories):
                category = int(category.cpu())
                if QUESTION_CATEGORY[category] == 'Which':
                    which_idx.append(i)
                elif QUESTION_CATEGORY[category] == 'Come From':
                    comefrom_idx.append(i)
                elif QUESTION_CATEGORY[category] == 'Happening':
                    happening_idx.append(i)
                elif QUESTION_CATEGORY[category] == 'Where':
                    where_idx.append(i)
                elif QUESTION_CATEGORY[category] == 'Why':
                    why_idx.append(i)
                elif QUESTION_CATEGORY[category] == 'Before Next':
                    beforenext_idx.append(i)
                elif QUESTION_CATEGORY[category] == 'When':
                    when_idx.append(i)
                elif QUESTION_CATEGORY[category] == 'Used For':
                    usedfor_idx.append(i)
                else:
                    raise ValueError('unseen value in question categories?')
        
            preds = torch.argmax(logits.view(batch_size, cfg.dataset.ans_count), dim=1)
            agreeings = (preds == answers)

            if write_preds:
                preds = logits.argmax(1)
                answer_vocab = data.vocab['question_answer_idx_to_token']
                for predict in preds:
                    all_preds.append(predict.item())
                    
                for gt in answers:
                    gts.append(gt.item())
                    
                for id in video_ids:
                    v_ids.append(id.cpu().numpy())
                for ques_id in question_ids:
                    q_ids.append(ques_id.cpu().numpy())

            total_acc += agreeings.float().sum().item()
            count += answers.size(0)

            which_acc += agreeings.float()[which_idx].sum().item() if which_idx != [] else 0
            comefrom_acc += agreeings.float()[comefrom_idx].sum().item() if comefrom_idx != [] else 0
            happening_acc += agreeings.float()[happening_idx].sum().item() if happening_idx != [] else 0
            where_acc += agreeings.float()[where_idx].sum().item() if where_idx != [] else 0
            why_acc += agreeings.float()[why_idx].sum().item() if why_idx != [] else 0
            beforenext_acc += agreeings.float()[beforenext_idx].sum().item() if beforenext_idx != [] else 0
            when_acc += agreeings.float()[when_idx].sum().item() if when_idx != [] else 0
            usedfor_acc += agreeings.float()[usedfor_idx].sum().item() if usedfor_idx != [] else 0
            which_count += len(which_idx)
            comefrom_count += len(comefrom_idx)
            happening_count += len(happening_idx)
            where_count += len(where_idx)
            why_count += len(why_idx)
            beforenext_count += len(beforenext_idx)
            when_count += len(when_idx)
            usedfor_count += len(usedfor_idx)


        acc = total_acc / count
        which_acc = which_acc / which_count
        comefrom_acc = comefrom_acc / comefrom_count
        happening_acc = happening_acc / happening_count
        where_acc = where_acc / where_count
        why_acc = why_acc / why_count
        beforenext_acc = beforenext_acc / beforenext_count
        when_acc = when_acc / when_count
        usedfor_acc = usedfor_acc / usedfor_count
        
    if not write_preds:
        return acc, which_acc, comefrom_acc, happening_acc, where_acc, why_acc, beforenext_acc, when_acc, usedfor_acc
    else:
        return acc, all_preds, gts, v_ids, q_ids, which_acc, comefrom_acc, happening_acc, where_acc, why_acc, beforenext_acc, when_acc, usedfor_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='avqa.yml', type=str)
    parser.add_argument('--app_feat', default='resnet101', type=str)
    parser.add_argument('--motion_feat', default='resnext101', type=str)
    parser.add_argument('--audio_feat', default='PANNs', type=str)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model.pt')
    assert os.path.exists(ckpt)
    # load pretrained model
    loaded = torch.load(ckpt, map_location='cpu')
    model_kwargs = loaded['model_kwargs']

    # cfg.dataset.question_type = 'none'
    cfg.dataset.appearance_feat = '{}_appearance_{}_feat.h5'
    cfg.dataset.motion_feat = '{}_motion_{}_feat.h5'
    cfg.dataset.vl_audio_feat = '{}_vlaudio_{}_feat.h5'
    cfg.dataset.vocab_json = '{}_vocab.json'
    cfg.dataset.test_question_pt = '{}_val_questions.pt'

    cfg.dataset.test_question_pt = os.path.join(cfg.dataset.data_dir,
                                                cfg.dataset.test_question_pt.format(cfg.dataset.name))
    cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))

    cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name, args.app_feat))
    cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name, args.motion_feat))
    cfg.dataset.vl_audio_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.vl_audio_feat.format(cfg.dataset.name, args.audio_feat))


    test_loader_kwargs = {
        'question_type': cfg.dataset.question_type,
        'question_pt': cfg.dataset.test_question_pt,
        'vocab_json': cfg.dataset.vocab_json,
        'appearance_feat': cfg.dataset.appearance_feat,
        'motion_feat': cfg.dataset.motion_feat,
        'vl_audio_feat': cfg.dataset.vl_audio_feat,
        'test_num': cfg.test.test_num,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': False
    }
    test_loader = AVQADataLoader(**test_loader_kwargs)
    model_kwargs.update({'vocab': test_loader.vocab})
    model = HCRN.HCRNNetwork(**model_kwargs).to(device)
    model.load_state_dict(loaded['state_dict'])

    if cfg.test.write_preds:
        acc, preds, gts, v_ids, q_ids = validate(cfg, model, test_loader, device)

        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()

        # write predictions for visualization purposes
        output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            assert os.path.isdir(output_dir)
        preds_file = os.path.join(output_dir, "test_preds.json")
        
        # Find groundtruth questions and corresponding answer candidates
        vocab = test_loader.vocab['question_answer_idx_to_token']
        dict = {}
        with open(cfg.dataset.test_question_pt, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            org_v_ids = obj['video_ids']
            org_v_names = obj['video_names']
            org_q_ids = obj['question_id']
            ans_candidates = obj['ans_candidates']

        for idx in range(len(org_q_ids)):
            dict[str(org_q_ids[idx])] = [org_v_names[idx], org_v_ids[idx], org_q_ids[idx], questions[idx], ans_candidates[idx]]

        instances = []
        for video_id, q_id, answer, pred in zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds):
            if answer != pred:
                instances.append({'question_id': dict[str(q_id)][2], 'video_name': dict[str(q_id)][0],
                                'answer': answer, 'prediction': pred})
        # write preditions to json file
        with open(preds_file, 'w') as f:
            json.dump(instances, f)
        sys.stdout.write('Display 10 samples...\n')
        # Display 10 samples
        # for idx in range(10):
        #     print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
        #     cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
        #     print('Question: ' + ' '.join(cur_question) + '?')
        #     all_answer_cands = dict[str(q_ids[idx].item())][2]
        #     for cand_id in range(len(all_answer_cands)):
        #         cur_answer_cands = [vocab[word.item()] for word in all_answer_cands[cand_id] if word
        #                             != 0]
        #         print('({}): '.format(cand_id) + ' '.join(cur_answer_cands))
        #     print('Prediction: {}'.format(preds[idx]))
        #     print('Groundtruth: {}'.format(gts[idx]))
    else:
        acc = validate(cfg, model, test_loader, device, cfg.test.write_preds)
        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()
