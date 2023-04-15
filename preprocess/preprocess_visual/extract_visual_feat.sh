conda create -n preprocess_visual python=3.6
conda activate preprocess_visual
conda install -c conda-forge ffmpeg
conda install -c conda-forge scikit-video
pip install -r requirements.txt

# modify the out_path
python preprocess_visual.py \
    --feature_type="appearance" \
    --video_path="../../data/video" \
    --video_name_mapping="../../data/video/video_map.json" \
    --annotation_file="../../data/annotation/{}_qa.json" \
    --out_path="../../data/feats/avqa_appearance_resnet_feat.h5" \
    --

# download resnext-101-kinetics.pth