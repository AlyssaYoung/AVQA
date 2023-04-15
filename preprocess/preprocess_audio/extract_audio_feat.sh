CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"
wget -O $CHECKPOINT_PATH https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1
CUDA_VISIBLE_DEVICES=0 python preprocess_audio.py \
    --feature_type="vlaudio" \
    --checkpoint_path=$CHECKPOINT_PATH \
    --audio_path= "../../data/audio" \
    --video_name_mapping="../../data/video/video_map.json" \
    --out_path="../../data/feats/avqa_vlaudio_PANNs_feat.h5" \
    --cuda