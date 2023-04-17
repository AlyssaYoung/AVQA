python preprocess_questions.py \
    --dataset avqa \
    --glove_pt ../../data/glove/glove.840.300d.pkl \
    --annotation_path ../../data/annotation/ \
    --out_path ../../data/feats/ \
    --mode train
    
python preprocess_questions.py \
    --dataset avqa \
    --annotation_path ../../data/annotation/ \
    --out_path ../../data/feats/ \
    --mode val