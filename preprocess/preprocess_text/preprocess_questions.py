import argparse
import numpy as np
import os

from datautils import avqa

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='avqa', type=str)
    parser.add_argument('--answer_top', default=7000, type=int)
    parser.add_argument('--glove_pt', default='../data/glove/glove.840.300d.pkl',
                        help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--annotation_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--method', type=str, default='glove.840.300d')
    parser.add_argument('--mode', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--question_type', choices=['frameqa', 'action', 'transition', 'count', 'none'], default='none')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--ans_count', type=int, default=4)

    args = parser.parse_args()
    np.random.seed(args.seed)

    args.annotation_file = os.path.join(args.annotation_path, '{}_qa.json'.format(args.mode))
    args.output_pt = os.path.join(args.out_path, '{}_{}_questions.pt')
    args.vocab_json = '{}_vocab.json'
    
    avqa.process_questions_mulchoices(args)
