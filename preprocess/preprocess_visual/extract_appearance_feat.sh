python preprocess_visual.py \
    --gpu_id 0 \
    --dataset avqa \
    --model resnet101 \
    --feature_type appearance \
    --video_path data/video \
    --video_name_mapping data/video/video_map.json \
    --annotation_path data/annotation \
    --out_path data/feats/avqa_appearance_resnet101_feat.h5