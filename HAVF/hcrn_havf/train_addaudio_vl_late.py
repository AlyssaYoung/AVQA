from dbm import whichdb
import os, sys

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import numpy as np
import argparse
import time
import logging
from termcolor import colored

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from data.dataloader.DataLoader_addaudio import VideoQADataLoader
from utils.utils import todevice
from validate_addaudio_vl import validate

import model.HCRN_addaudio_vl as HCRN
import model.HCRN_addaudio_vl_multilevel_late as HCRN_multilevel


from configs.config import cfg, cfg_from_file

class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, model3=None):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
    
    def forward(self, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, vl_audio_feat, cl_audio_feat, question,
                question_len):
        result1 = self.model1(ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, vl_audio_feat, cl_audio_feat, question,question_len)
        result2 = self.model2(ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, vl_audio_feat, cl_audio_feat, question,question_len)
        if self.model3 is not None:
            result3 = self.model3(ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, vl_audio_feat, cl_audio_feat, question,question_len)
            out = (result1 + result2 + result3) / 3
        else:
            out = (result1 + result2) / 2

        return out

def train(cfg):
    logging.info("Create train_loader and val_loader.........")
    train_loader_kwargs = {
        'question_type': cfg.dataset.question_type,
        'question_pt': '/DATA/DATANAS1/yangpinci/VGGSound_QA/hcrn-videoqa/data/datasets/vggsound-qa/vggsound-qa_train_questions_addtype.pt',
        'vocab_json': '/DATA/DATANAS1/yangpinci/VGGSound_QA/hcrn-videoqa/data/datasets/vggsound-qa/vggsound-qa_vocab.json',
        'appearance_feat': cfg.dataset.appearance_feat,
        'motion_feat': cfg.dataset.motion_feat,
        'vl_audio_feat': '/DATA/DATANAS1/yangpinci/VGGSound_QA/preprocess_audio/vggsound-qa_vlaudio_PANNs_feat.h5',
        'cl_audio_feat': '/DATA/DATANAS1/yangpinci/VGGSound_QA/preprocess_audio/vggsound-qa_claudio_PANNs_feat.h5',
        'train_num': cfg.train.train_num,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': True,
        'drop_last': True,
        'useAudio': True
    }
    train_loader = VideoQADataLoader(**train_loader_kwargs)
    logging.info("number of train instances: {}".format(len(train_loader.dataset)))
    if cfg.val.flag:
        val_loader_kwargs = {
            'question_type': cfg.dataset.question_type,
            'question_pt': '/DATA/DATANAS1/yangpinci/VGGSound_QA/hcrn-videoqa/data/datasets/vggsound-qa/vggsound-qa_val_questions_addtype.pt',
            'vocab_json': '/DATA/DATANAS1/yangpinci/VGGSound_QA/hcrn-videoqa/data/datasets/vggsound-qa/vggsound-qa_vocab.json',
            'appearance_feat': cfg.dataset.appearance_feat,
            'motion_feat': cfg.dataset.motion_feat,
            'vl_audio_feat': '/DATA/DATANAS1/yangpinci/VGGSound_QA/preprocess_audio/vggsound-qa_vlaudio_PANNs_feat.h5',
            'cl_audio_feat': '/DATA/DATANAS1/yangpinci/VGGSound_QA/preprocess_audio/vggsound-qa_claudio_PANNs_feat.h5',
            'val_num': cfg.val.val_num,
            'batch_size': cfg.train.batch_size,
            'num_workers': cfg.num_workers,
            'shuffle': False,
            'useAudio': True            
        }
        val_loader = VideoQADataLoader(**val_loader_kwargs)
        logging.info("number of val instances: {}".format(len(val_loader.dataset)))

    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {
        'vision_dim': cfg.train.vision_dim,
        'audio_dim' : cfg.train.audio_dim,
        'module_dim': cfg.train.module_dim,
        'word_dim': cfg.train.word_dim,
        'k_max_frame_level': cfg.train.k_max_frame_level,
        'k_max_clip_level': cfg.train.k_max_clip_level,
        'spl_resolution': cfg.train.spl_resolution,
        'vocab': train_loader.vocab,
        'question_type': cfg.dataset.question_type,
        'level': cfg.level
    }
    model_kwargs_tosave = {k: v for k, v in model_kwargs.items() if k != 'vocab'}

    if cfg.ensemble:
        print('test')
    #     model_early = HCRN_multilevel.HCRNNetwork(**model_kwargs).to(device)
    #     ckpt_early = '/DATA/DATANAS1/yangpinci/VGGSound_QA/hcrn-block/exp3_hcrn_addaudio_vl_multilevel_earlyconcat/ckpt/model.pt'
    #     ckpt_early = torch.load(ckpt_early, map_location=lambda storage, loc: storage)
    #     model_early.load_state_dict(ckpt_early['state_dict'])
    #     model_middle = 
    #     ckpt_middle = '/DATA/DATANAS1/yangpinci/VGGSound_QA/hcrn-block/exp3_hcrn_addaudio_vl_multilevel/ckpt/model.pt'
    #     model_middle = torch.load(ckpt_middle, map_location=lambda storage, loc: storage)
    #     model_middle.load_state_dict(ckpt_middle['state_dict'])
    #     model_late = 
    #     ckpt_late = '/DATA/DATANAS1/yangpinci/VGGSound_QA/hcrn-block/exp3_hcrn_addaudio_vl_multilevel/late/ckpt/model.pt'
    #     model_late.load_state_dict(ckpt_late['state_dict'])
    #     model = EnsembleModel(model_early, model_middle, model_late)

    #     valid_acc, which_acc, comefrom_acc, happening_acc, where_acc, why_acc, beforenext_acc, when_acc, usedfor_acc = validate(cfg, model, val_loader, device, write_preds=False)

    #     logging.info('~~~~~~ Valid Accuracy: %.4f ~~~~~~~' % valid_acc)
    #     logging.info('~~~~~~ Valid Which Accuracy: %.4f ~~~~~~~' % which_acc)
    #     logging.info('~~~~~~ Valid Come From Accuracy: %.4f ~~~~~~' % comefrom_acc)
    #     logging.info('~~~~~~ Valid Happening Accuracy: %.4f ~~~~~~' % happening_acc)
    #     logging.info('~~~~~~ Valid Where Accuracy: %.4f ~~~~~~' % where_acc)
    #     logging.info('~~~~~~ Valid Why Accuracy: %.4f ~~~~~~' % why_acc)
    #     logging.info('~~~~~~ Valid Before Next Accuracy: %.4f ~~~~~~' % beforenext_acc)
    #     logging.info('~~~~~~ Valid When Accuracy: %.4f ~~~~~~' % when_acc)
    #     logging.info('~~~~~~ Valid Used For Accuracy: %.4f ~~~~~~' % usedfor_acc)
    #     sys.stdout.write(
    #         '~~~~~~ Valid Accuracy: {valid_acc}, Which Accuracy: {which_acc}, Come From Accuracy: {comefrom_acc}, Happening Accuracy: {happening_acc}, Where Accuracy: {where_acc}, Why Accuracy: {why_acc}, Before Next Accuracy: {beforenext_acc}, When Accuracy: {when_acc}, Used For Accuracy: {usedfor_acc} ~~~~~~~\n'.format(
    #             valid_acc=colored("{:.4f}".format(valid_acc), "red", attrs=['bold']),
    #             which_acc=colored("{:.4f}".format(which_acc), "red", attrs=['bold']),
    #             comefrom_acc=colored('{:.4f}'.format(comefrom_acc), "red", attrs=['bold']),
    #             happening_acc=colored('{:.4f}'.format(happening_acc), "red", attrs=['bold']),
    #             where_acc=colored('{:.4f}'.format(where_acc), "red", attrs=['bold']),
    #             why_acc=colored('{:.4f}'.format(why_acc), "red", attrs=['bold']),
    #             beforenext_acc=colored('{:.4f}'.format(beforenext_acc), "red", attrs=['bold']),
    #             when_acc=colored('{:.4f}'.format(when_acc), "red", attrs=['bold']),
    #             usedfor_acc=colored('{:.4f}'.format(usedfor_acc), "red", attrs=['bold'])
    #             ))
    #     sys.stdout.flush()


    else:
        if cfg.model_name == 'multilevel':
            model = HCRN_multilevel.HCRNNetwork(**model_kwargs).to(device)
        else:
            model = HCRN.HCRNNetwork(**model_kwargs).to(device)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info('num of params: {}'.format(pytorch_total_params))
        logging.info(model)

        if cfg.train.glove:
            logging.info('load glove vectors')
            train_loader.glove_matrix = torch.FloatTensor(train_loader.glove_matrix).to(device)
            with torch.no_grad():
                model.linguistic_input_unit.encoder_embed.weight.set_(train_loader.glove_matrix)
                model.linguistic_input_unit.encoder_embed.weight.requires_grad = False
        if torch.cuda.device_count() > 1 and cfg.multi_gpus:
            model = model.cuda()
            logging.info("Using {} GPUs".format(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=None)

        optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad == True], cfg.train.lr)
        
        start_epoch = 0
        if cfg.dataset.question_type == 'count':
            best_val = 100.0
        else:
            best_val = 0.
            best_which = 0.
            best_comefrom = 0.
            best_happening = 0.
            best_where = 0.
            best_why = 0.
            best_beforenext = 0.
            best_when = 0.
            best_usedfor = 0.
        if cfg.train.restore:
            print("Restore checkpoint and optimizer...")
            ckpt = os.path.join(cfg.dataset.save_dir, 'weights', 'model.pt')
            ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
            start_epoch = ckpt['epoch'] + 1
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
        if cfg.dataset.question_type in ['frameqa', 'none']:
            criterion = nn.CrossEntropyLoss().to(device)
        elif cfg.dataset.question_type == 'count':
            criterion = nn.MSELoss().to(device)
        logging.info("Start training........")
        for epoch in range(start_epoch, cfg.train.max_epochs):
            logging.info('>>>>>> epoch {epoch} <<<<<<'.format(epoch=colored("{}".format(epoch), "green", attrs=["bold"])))
            model.train()
            total_acc, count = 0, 0
            batch_mse_sum = 0.0
            total_loss, avg_loss = 0.0, 0.0
            avg_loss = 0
            train_accuracy = 0
            for i, batch in enumerate(iter(train_loader)):
                progress = epoch + i / len(train_loader)
                _, _, question_categories, answers, *batch_input = [todevice(x, device) for x in batch]
                answers = answers.cuda().squeeze()
                batch_size = answers.size(0)
                optimizer.zero_grad()
                logits = model(*batch_input)
                if cfg.dataset.question_type in ['action', 'transition']:
                    batch_agg = np.concatenate(np.tile(np.arange(batch_size).reshape([batch_size, 1]),
                                                    [1, cfg.dataset.ans_count])) * cfg.dataset.ans_count  # [0, 0, 0, 0, 0, 5, 5, 5, 5, 1, ...]
                    answers_agg = tile(answers, 0, cfg.dataset.ans_count)
                    loss = torch.max(torch.tensor(0.0).cuda(),
                                    1.0 + logits - logits[answers_agg + torch.from_numpy(batch_agg).cuda()])
                    loss = loss.sum() / torch.tensor(1.0*batch_size).cuda()
                    loss.backward()

                    total_loss += loss.detach()
                    avg_loss = total_loss / (i + 1)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                    optimizer.step()
                    preds = torch.argmax(logits.view(batch_size, cfg.dataset.ans_count), dim=1)
                    aggreeings = (preds == answers)
                elif cfg.dataset.question_type == 'count':
                    answers = answers.unsqueeze(-1)
                    loss = criterion(logits, answers.float())
                    loss.backward()
                    total_loss += loss.detach()
                    avg_loss = total_loss / (i + 1)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                    optimizer.step()
                    preds = (logits + 0.5).long().clamp(min=1, max=10)
                    batch_mse = (preds - answers) ** 2
                else:
                    loss = criterion(logits, answers)
                    loss.backward()
                    total_loss += loss.detach()
                    avg_loss = total_loss / (i + 1)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                    optimizer.step()
                    aggreeings = batch_accuracy(logits, answers)

                if cfg.dataset.question_type == 'count':
                    batch_avg_mse = batch_mse.sum().item() / answers.size(0)
                    batch_mse_sum += batch_mse.sum().item()
                    count += answers.size(0)
                    avg_mse = batch_mse_sum / count
                    sys.stdout.write(
                        "\rProgress = {progress}   ce_loss = {ce_loss}   avg_loss = {avg_loss}    train_mse = {train_mse}    avg_mse = {avg_mse}    exp: {exp_name}".format(
                            progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                            ce_loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                            avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                            train_mse=colored("{:.4f}".format(batch_avg_mse), "blue",
                                            attrs=['bold']),
                            avg_mse=colored("{:.4f}".format(avg_mse), "red", attrs=['bold']),
                            exp_name=cfg.exp_name))
                    sys.stdout.flush()
                else:
                    total_acc += aggreeings.sum().item()
                    count += answers.size(0)
                    train_accuracy = total_acc / count
                    sys.stdout.write(
                        "\rProgress = {progress}  ce_loss = {ce_loss}  avg_loss = {avg_loss}   train_acc = {train_acc}   avg_acc = {avg_acc}   exp: {exp_name}".format(
                            progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                            ce_loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                            avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                            train_acc=colored("{:.4f}".format(aggreeings.float().mean().cpu().numpy()), "blue",
                                            attrs=['bold']),
                            avg_acc=colored("{:.4f}".format(train_accuracy), "red", attrs=['bold']),
                            exp_name=cfg.exp_name))
                    sys.stdout.flush()
            sys.stdout.write("\n")
            
            if cfg.dataset.question_type == 'count':
                if (epoch + 1) % 5 == 0:
                    optimizer = step_decay(cfg, optimizer)
            else:
                if (epoch + 1) % 10 == 0:
                    optimizer = step_decay(cfg, optimizer)
            
            #scheduler.step()
            sys.stdout.flush()
            logging.info("Epoch = %s   avg_loss = %.3f    avg_acc = %.3f" % (epoch, avg_loss, train_accuracy))

            if cfg.val.flag:
                output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                else:
                    assert os.path.isdir(output_dir)
                valid_acc, which_acc, comefrom_acc, happening_acc, where_acc, why_acc, beforenext_acc, when_acc, usedfor_acc = validate(cfg, model, val_loader, device, write_preds=False)
                if (valid_acc > best_val and cfg.dataset.question_type != 'count') or (valid_acc < best_val and cfg.dataset.question_type == 'count'):
                    best_val = valid_acc
                    best_which = which_acc
                    best_comefrom = comefrom_acc
                    best_happening = happening_acc
                    best_where = where_acc
                    best_why = why_acc
                    best_beforenext = beforenext_acc
                    best_when = when_acc
                    best_usedfor = usedfor_acc

                    # Save best model
                    ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    else:
                        assert os.path.isdir(ckpt_dir)
                    save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, os.path.join(ckpt_dir, 'model.pt'))
                    sys.stdout.write('\n >>>>>> save to %s <<<<<< \n' % (ckpt_dir))
                    sys.stdout.flush()

                logging.info('~~~~~~ Valid Accuracy: %.4f ~~~~~~~' % valid_acc)
                logging.info('~~~~~~ Valid Which Accuracy: %.4f ~~~~~~~' % which_acc)
                logging.info('~~~~~~ Valid Come From Accuracy: %.4f ~~~~~~' % comefrom_acc)
                logging.info('~~~~~~ Valid Happening Accuracy: %.4f ~~~~~~' % happening_acc)
                logging.info('~~~~~~ Valid Where Accuracy: %.4f ~~~~~~' % where_acc)
                logging.info('~~~~~~ Valid Why Accuracy: %.4f ~~~~~~' % why_acc)
                logging.info('~~~~~~ Valid Before Next Accuracy: %.4f ~~~~~~' % beforenext_acc)
                logging.info('~~~~~~ Valid When Accuracy: %.4f ~~~~~~' % when_acc)
                logging.info('~~~~~~ Valid Used For Accuracy: %.4f ~~~~~~' % usedfor_acc)
                # sys.stdout.write('~~~~~~ Valid Accuracy: {valid_acc} ~~~~~~~\n'.format(
                #     valid_acc=colored("{:.4f}".format(valid_acc), "red", attrs=['bold'])))
                sys.stdout.write(
                    '~~~~~~ Valid Accuracy: {valid_acc}, Which Accuracy: {which_acc}, Come From Accuracy: {comefrom_acc}, Happening Accuracy: {happening_acc}, Where Accuracy: {where_acc}, Why Accuracy: {why_acc}, Before Next Accuracy: {beforenext_acc}, When Accuracy: {when_acc}, Used For Accuracy: {usedfor_acc} ~~~~~~~\n'.format(
                        valid_acc=colored("{:.4f}".format(valid_acc), "red", attrs=['bold']),
                        which_acc=colored("{:.4f}".format(which_acc), "red", attrs=['bold']),
                        comefrom_acc=colored('{:.4f}'.format(comefrom_acc), "red", attrs=['bold']),
                        happening_acc=colored('{:.4f}'.format(happening_acc), "red", attrs=['bold']),
                        where_acc=colored('{:.4f}'.format(where_acc), "red", attrs=['bold']),
                        why_acc=colored('{:.4f}'.format(why_acc), "red", attrs=['bold']),
                        beforenext_acc=colored('{:.4f}'.format(beforenext_acc), "red", attrs=['bold']),
                        when_acc=colored('{:.4f}'.format(when_acc), "red", attrs=['bold']),
                        usedfor_acc=colored('{:.4f}'.format(usedfor_acc), "red", attrs=['bold'])
                        ))
                sys.stdout.flush()

# Credit https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


def step_decay(cfg, optimizer):
    # compute the new learning rate based on decay rate
    cfg.train.lr *= 0.5
    logging.info("Reduced learning rate to {}".format(cfg.train.lr))
    sys.stdout.flush()
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.train.lr

    return optimizer


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    predicted = predicted.detach().argmax(1)
    agreeing = (predicted == true)
    return agreeing


def save_checkpoint(epoch, model, optimizer, model_kwargs, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_kwargs': model_kwargs,
    }
    time.sleep(10)
    torch.save(state, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/msvd_qa.yml', type=str)
    parser.add_argument('--app_feat', default='resnet101', type=str)
    parser.add_argument('--motion_feat', default='resnext101', type=str)
    parser.add_argument('--cl_audio_feat', default='PANNs', type=str)
    parser.add_argument('--vl_audio_feat', default='PANNs', type=str)
    parser.add_argument('--ensemble', default=False, type=bool)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    assert cfg.dataset.name in ['tgif-qa', 'msrvtt-qa', 'msvd-qa', 'vggsound-qa']
    assert cfg.dataset.question_type in ['frameqa', 'count', 'transition', 'action', 'none']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)
    # check if k_max is set correctly
    assert cfg.train.k_max_frame_level <= 16
    assert cfg.train.k_max_clip_level <= 8


    if not cfg.multi_gpus:
        torch.cuda.set_device(cfg.gpu_id)
    # make logging.info display into both shell and file
    if not os.path.exists(cfg.dataset.save_dir):
        os.makedirs(cfg.dataset.save_dir)
    else:
        assert os.path.isdir(cfg.dataset.save_dir)

    fileHandler = logging.FileHandler(os.path.join(cfg.dataset.save_dir, 'stdout.log'), 'w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(cfg).items():
        logging.info(k + ':' + str(v))
    # concat absolute path of input files

    if cfg.dataset.name == 'tgif-qa':
        cfg.dataset.train_question_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_pt.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.val_question_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_pt.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name, cfg.dataset.question_type))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name, cfg.dataset.question_type))
    else:
        #cfg.dataset.question_type = 'none'
        cfg.dataset.appearance_feat = '{}_appearance_{}_feat.h5'
        cfg.dataset.motion_feat = '{}_motion_{}_feat.h5'
        cfg.dataset.vl_audio_feat = '{}_vlaudio_{}_feat.h5'
        cfg.dataset.al_audio_feat = '{}_claudio_{}_feat.h5'
        cfg.dataset.vocab_json = '{}_vocab.json'
        cfg.dataset.train_question_pt = '{}_train_questions.pt'
        cfg.dataset.val_question_pt = '{}_val_questions.pt'
        cfg.dataset.train_question_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_pt.format(cfg.dataset.name))
        cfg.dataset.val_question_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_pt.format(cfg.dataset.name))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name, args.app_feat))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name, args.motion_feat))
        cfg.dataset.vl_audio_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.vl_audio_feat.format(cfg.dataset.name, args.vl_audio_feat))
        # cfg.dataset.cl_audio_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.cl_audio_feat.format(cfg.dataset.name, args.cl_audio_feat))

    # set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    train(cfg)


if __name__ == '__main__':
    main()
