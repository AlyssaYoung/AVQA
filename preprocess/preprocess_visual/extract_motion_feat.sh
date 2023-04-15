python preprocess_visual.py \
    --gpu_id 0 \
    --dataset avqa \
    --model resnext101 \
    --image_height 112 \
    --image_width 112 \
    --feature_type motion \
    --video_path data/video \
    --video_name_mapping data/video/video_map.json \
    --annotation_path data/annotation \
    --out_path data/feats/avqa_motion_resnext101_feat.h5

