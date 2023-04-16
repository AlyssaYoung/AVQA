import numpy as np
from torch.nn import functional as F

from .utils import *
from .CRN import CRN

class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        visual_feat = self.dropout(visual_feat)
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)
        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1), k=48
        # print('attn shape:', attn.shape)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1), k=48
        # print('attn shape:', attn.shape)

        v_distill = (attn * visual_feat).sum(1) #(bz, 512)
        # print('v_distill shape:', v_distill.shape)

        return v_distill

class InputUnitLinguistic(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len, batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding

class InputUnitAudio(nn.Module):
    def __init__(self, audio_dim, module_dim=512):
        super(InputUnitAudio, self).__init__()
        self.audio_feat_proj = nn.Linear(audio_dim, module_dim)
        self.question_embedding_proj = nn.Linear(module_dim, module_dim)
        self.activation = nn.ELU()
        self.fusion = nn.Linear(module_dim * 2, module_dim)
        self.gate_fusion = nn.Linear(module_dim * 2, module_dim)
    
    def forward(self, vl_audio_feat, question_embedding):
        """
        Args:
            vl_audio_feat: [Tensor] (batch_size, audio_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded audio feature: [Tensor] (batch_size, module_dim)
        """
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        audio_feat_proj = self.audio_feat_proj(vl_audio_feat)
        h_feat = torch.cat((audio_feat_proj, question_embedding_proj), dim=-1)
        h_feat = self.activation(self.fusion(h_feat)) * torch.sigmoid(self.gate_fusion(h_feat))

        return h_feat

class InputUnitVisual(nn.Module):
    def __init__(self, k_max_frame_level, k_max_clip_level, spl_resolution, vision_dim, audio_dim, module_dim=512, level='middle-vcrn', crn_type='concat'):
        super(InputUnitVisual, self).__init__()

        self.crn_type = crn_type
        self.level = level

        self.clip_level_motion_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution)
        self.clip_level_question_cond = CRN(module_dim, k_max_frame_level-2, k_max_frame_level-2, gating=True, spl_resolution=spl_resolution)
        self.video_level_motion_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        self.video_level_audio_cond = CRN(module_dim, k_max_clip_level, k_max_clip_level, gating=False, spl_resolution=spl_resolution)
        self.video_level_question_cond = CRN(module_dim, k_max_clip_level-2, k_max_clip_level-2, gating=True, spl_resolution=spl_resolution)

        self.sequence_encoder = nn.LSTM(vision_dim, module_dim, batch_first=True, bidirectional=False)
        self.clip_level_motion_proj = nn.Linear(vision_dim, module_dim)
        self.video_level_motion_proj = nn.Linear(module_dim, module_dim)
        if self.level == 'early':
            self.appearance_feat_proj = nn.Linear(vision_dim + audio_dim, module_dim)
        else:
            self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)
            self.audio_feat_proj = nn.Linear(audio_dim, module_dim)
            self.clip_level_audio_cond = CRN(module_dim, k_max_frame_level, k_max_frame_level, gating=False, spl_resolution=spl_resolution, fuse_type=self.crn_type)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)
        

        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video_feat, motion_video_feat, vl_audio_feat, question_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
            vl_audio_feat: [Tensor] (batch_size, audio_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = appearance_video_feat.size(0)
        clip_level_crn_outputs = []
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        video_level_audio_feat_proj = self.audio_feat_proj(vl_audio_feat) # (bz, 512)
        
        if self.level in ['middle-vcrn', 'middle-ccrn', 'middle-2crn']:
            audio_embedding_proj = self.audio_feat_proj(vl_audio_feat)

        for i in range(appearance_video_feat.size(1)):
            clip_level_motion = motion_video_feat[:, i, :]  # (bz, 2048)
            clip_level_motion_proj = self.clip_level_motion_proj(clip_level_motion)

            if self.level == 'early':
                vl_audio_feat_repeat = vl_audio_feat.unsqueeze(1)
                vl_audio_feat_repeat = vl_audio_feat_repeat.repeat(1, appearance_video_feat.size(2), 1)
                clip_level_appearance = torch.cat((appearance_video_feat[:, i, :, :], vl_audio_feat_repeat), dim=-1) # (bz, 16, 4096)
            else:
                clip_level_appearance = appearance_video_feat[:, i, :, :]  # (bz, 16, 2048)
            clip_level_appearance_proj = self.appearance_feat_proj(clip_level_appearance)  # (bz, 16, 512)
            # clip level CRNs
            clip_level_crn_motion = self.clip_level_motion_cond(torch.unbind(clip_level_appearance_proj, dim=1),
                                                                clip_level_motion_proj) # [14], (bz, 512)
            # print('clip_level_crn_motion shape:', len(clip_level_crn_motion), clip_level_crn_motion[0].shape) #[12], (bz, 512)
            if self.level in ['middle-ccrn', 'middle-2crn']:
                clip_level_crn_audio = self.clip_level_audio_cond(clip_level_crn_motion, audio_embedding_proj)
                clip_level_crn_question = self.clip_level_question_cond(clip_level_crn_audio, question_embedding_proj)
            else:
                clip_level_crn_question = self.clip_level_question_cond(clip_level_crn_motion, question_embedding_proj)
            # print('clip_level_crn_question shape:', len(clip_level_crn_question), clip_level_crn_question[0].shape)
            clip_level_crn_output = torch.cat(
                [frame_relation.unsqueeze(1) for frame_relation in clip_level_crn_question],
                dim=1)
            clip_level_crn_output = clip_level_crn_output.view(batch_size, -1, self.module_dim) #(bz, 12, 512)
            # print('clip_level_crn_output shape:', clip_level_crn_output.shape)
            clip_level_crn_outputs.append(clip_level_crn_output)
        
        # Encode video level motion
        _, (video_level_motion, _) = self.sequence_encoder(motion_video_feat)
        video_level_motion = video_level_motion.transpose(0, 1)
        video_level_motion_feat_proj = self.video_level_motion_proj(video_level_motion)
        # video level CRNs
        video_level_crn_motion = self.video_level_motion_cond(clip_level_crn_outputs, video_level_motion_feat_proj) # [6], (bz, 12, 512)
        # video_level_crn_question = self.video_level_question_cond(video_level_crn_motion, question_embedding_proj.unsqueeze(1))
        # print('video_level_crn_motion shape:', len(video_level_crn_motion), video_level_crn_motion[0].shape)

        if self.level in ['middle-vcrn', 'middle-2crn']:
            video_level_crn_audio = self.video_level_audio_cond(video_level_crn_motion, video_level_audio_feat_proj.unsqueeze(1)) # [5], (bz, 12, 512)
            # print('video_level_crn_audio shape:', len(video_level_crn_audio), video_level_crn_audio[0].shape)
            video_level_crn_question = self.video_level_question_cond(video_level_crn_audio,
                                                                    question_embedding_proj.unsqueeze(1)) #[4], (bz, 12, 512)
            # print('video_level_crn_question shape:', len(video_level_crn_question), video_level_crn_question[0].shape)
        else:
            video_level_crn_question = self.video_level_question_cond(video_level_crn_motion,
                                                                  question_embedding_proj.unsqueeze(1)) #[4], (bz, 12, 512)
            # print('video_level_crn_question shape:', len(video_level_crn_question), video_level_crn_question[0].shape)

        video_level_crn_output = torch.cat([clip_relation.unsqueeze(1) for clip_relation in video_level_crn_question],
                                           dim=1)
        video_level_crn_output = video_level_crn_output.view(batch_size, -1, self.module_dim) # (bz, 48, 512)
        # print('video_level_crn_output shape:', video_level_crn_output.shape)

        return video_level_crn_output


class OutputUnitOpenEnded(nn.Module):
    def __init__(self, module_dim=512, num_answers=1000):
        super(OutputUnitOpenEnded, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitMultiChoices(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitMultiChoices, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 4, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, q_visual_embedding, ans_candidates_embedding,
                a_visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)
        out = torch.cat([q_visual_embedding, question_embedding, a_visual_embedding,
                         ans_candidates_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitCount(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitCount, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.regression = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.regression(out)

        return out


class HCRNNetwork(nn.Module):
    def __init__(self, vision_dim, audio_dim, module_dim, word_dim, k_max_frame_level, k_max_clip_level, spl_resolution, vocab, question_type, level='middle-vcrn', crn_type='concat'):
        super(HCRNNetwork, self).__init__()

        self.question_type = question_type
        self.level = level
        self.crn_type = crn_type
        self.feature_aggregation = FeatureAggregation(module_dim)

        encoder_vocab_size = len(vocab['question_answer_token_to_idx'])
        self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                            module_dim=module_dim, rnn_dim=module_dim)
        self.audio_input_unit = InputUnitAudio(audio_dim=audio_dim, module_dim=module_dim)
        self.output_unit = OutputUnitMultiChoices(module_dim=module_dim)

       
        self.visual_input_unit = InputUnitVisual(k_max_frame_level=k_max_frame_level, k_max_clip_level=k_max_clip_level, spl_resolution=spl_resolution, vision_dim=vision_dim, audio_dim=audio_dim, module_dim=module_dim, level=level, crn_type=crn_type)
        if self.level == 'late':
            self.av_latefusion = AVLateFusion(module_dim=module_dim)

        init_modules(self.modules(), w_init="xavier_uniform")
        nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)

    def forward(self, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, vl_audio_feat, question,
                question_len):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 4, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 4)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            vl_audio_feat: [Tensor](batch_size, audio_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question.size(0)
        
        question_embedding = self.linguistic_input_unit(question, question_len)
        visual_embedding = self.visual_input_unit(video_appearance_feat, video_motion_feat, vl_audio_feat, question_embedding)
        audio_embedding = self.audio_input_unit(vl_audio_feat, question_embedding)
        if self.level == 'late':
            visual_embedding = self.av_latefusion(audio_embedding, visual_embedding)

        q_visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)
        
        # ans_candidates: (batch_size, num_choices, max_len)
        ans_candidates_agg = ans_candidates.view(-1, ans_candidates.size(2))
        ans_candidates_len_agg = ans_candidates_len.view(-1)

        batch_agg = np.reshape(
            np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, 4]), [-1])

        ans_candidates_embedding = self.linguistic_input_unit(ans_candidates_agg, ans_candidates_len_agg)

        a_visual_embedding = self.feature_aggregation(ans_candidates_embedding, visual_embedding[batch_agg])
        out = self.output_unit(question_embedding[batch_agg], q_visual_embedding[batch_agg],
                                ans_candidates_embedding,
                                a_visual_embedding)


        return out


class AVLateFusion(nn.Module):
    def __init__(self, module_dim=512):
        super(AVLateFusion, self).__init__()
        self.module_dim = module_dim
        self.audio_feat_proj = nn.Linear(module_dim, module_dim)
        self.activation = nn.ELU()
    
    def forward(self, audio_feat, visual_feat):
        """
        Args:
            audio_feat: [Tensor] (batch_size, module_dim)
            visual_feat: [Tensor] (batch_size, k, module_dim)
        return:
            fused visual feature: [Tensor] (batch_size, k, module_dim)
        """
        audio_feat = audio_feat.unsqueeze(1)
        audio_feat_repeat = audio_feat.repeat(1, visual_feat.size(1), 1)
        # print(audio_feat_repeat.shape)
        audio_feat_repeat_proj = self.audio_feat_proj(audio_feat_repeat)
        #print(audio_feat_repeat.shape)

        fused_visual_feat = torch.mul(audio_feat_repeat_proj, visual_feat)
        fused_visual_feat = self.activation(fused_visual_feat)

        return fused_visual_feat
