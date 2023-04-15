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
        video_paths.append(((os.path.join(args.video_dir + '{}.mp4'.format(video_name))), video_dict[video_name]))
    return video_paths

def multichoice_encoding_data(args, vocab, questions, video_names, video_ids, answers, ans_candidates, question_category):
    # Encode all questions
    print('Encoding data')
    print('Data mode:', args.mode)
    questions_encoded = []
    questions_len = []
    question_ids = []
    all_answer_cands_encoded = []
    all_answer_cands_len = []
    video_ids_tbw = []
    video_names_tbw = []
    correct_answers = []
    for idx, question in enumerate(questions):
        question = question.lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        question_encoded = utils.encode(question_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(idx)
        video_names_tbw.append(video_names[idx])
        video_ids_tbw.append(video_ids[idx])

        # grounthtruth  
        answer = int(answers[idx])
        correct_answers.append(answer)
        # answer candidates
        candidates = ans_candidates[idx]
        candidates_encoded = []
        candidates_len = []
        for ans in candidates:
            try:
                ans = ans.lower()
            except:
                if ans is None:
                    ans = 'null'
            ans_tokens = nltk.word_tokenize(ans)
            cand_encoded = utils.encode(ans_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
            candidates_encoded.append(cand_encoded)
            candidates_len.append(len(cand_encoded))
        all_answer_cands_encoded.append(candidates_encoded)
        all_answer_cands_len.append(candidates_len)

    # Pad encoded questions
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_answer_token_to_idx']['<NULL>'])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print('Question shape after encoded:', questions_encoded.shape)

    # Pad encoded answer candidates
    max_answer_cand_length = max(max(len(x) for x in candidate) for candidate in all_answer_cands_encoded)
    for ans_cands in all_answer_cands_encoded:
        for ans in ans_cands:
            while len(ans) < max_answer_cand_length:
                ans.append(vocab['question_answer_token_to_idx']['<NULL>'])
    all_answer_cands_encoded = np.asarray(all_answer_cands_encoded, dtype=np.int32)
    all_answer_cands_len = np.asarray(all_answer_cands_len, dtype=np.int32)
    print('Ansewr shape after encoded:', all_answer_cands_encoded.shape)

    glove_matrix = None
    if args.mode in ['train']:
        token_itow = {i: w for w, i in vocab['question_answer_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print('GloVe matrix shape:', glove_matrix.shape)

    print('Writing', args.output_pt.format(args.dataset, args.dataset, args.mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'video_ids': np.asarray(video_ids_tbw),
        'video_names': np.array(video_names_tbw),
        'ans_candidates': all_answer_cands_encoded,
        'ans_candidates_len': all_answer_cands_len,
        'answers': correct_answers,
        'glove': glove_matrix,
        'question_category': question_category
    }
    with open(args.output_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
        pickle.dump(obj, f)

def process_questions_mulchoices(args):
    print('Loading data')

    # Loading annotation
    with open(args.annotation_file) as f:
        annotation = json.load(f)
    
    questions = []
    answers = []
    video_names = []
    video_ids = []
    ans_candidates = []
    question_category = []
    for anno in annotation:
        questions.append(anno['question_text'])
        answers.append(anno['answer'])
        video_names.append(anno['video_name'])
        video_ids.append(anno['video_id'])
        question_category.append(QUESTION_CATEGORY_DICT[anno['question_type']])
        #ans_candidates.append(anno['multi_choice'])
    
    ans_candidates = np.empty((len(answers), args.ans_count), dtype=object)
    for i, anno in enumerate(annotation):
        for j in range(args.ans_count):
            try:
                ans_candidates[i][j] = anno['multi_choice'][j]
            except:
                ans_candidates[i][j] = 'NULL'
        
    print('number of questions: %s' % len(questions))
    print('number of choice: %s' % ans_candidates.shape[0])
    # Either create the vocab or load it from disk
    if args.mode in ['train']:
        print('Building vocab')

        answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1}
        question_answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for candidates in ans_candidates:
            for ans in candidates:
                try:
                    ans = ans.lower()
                except:
                    print(candidates)
                    quit()
                for token in nltk.word_tokenize(ans):
                    if token not in answer_token_to_idx:
                        answer_token_to_idx[token] = len(answer_token_to_idx)
                    if token not in question_answer_token_to_idx:
                        question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

        question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for i, q in enumerate(questions):
            question = q.lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)

        print('Get question_token_to_idx ', len(question_token_to_idx))
        print('Get question_answer_token_to_idx ', len(question_answer_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            'question_answer_token_to_idx': question_answer_token_to_idx,
        }

        print('Write into %s' % args.vocab_json.format(args.dataset, args.dataset))
        with open(args.vocab_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab, f, indent=4)

        multichoice_encoding_data(args, vocab, questions, video_names, video_ids, answers, ans_candidates, question_category)

    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset, args.dataset, args.method), 'r') as f:
            vocab = json.load(f)
        multichoice_encoding_data(args, vocab, questions, video_names, video_ids, answers, ans_candidates, question_category)
