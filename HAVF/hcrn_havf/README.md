# HCRN+HAVF

#### Install dependencies:
```bash
conda create -n hcrn_videoqa python=3.6
conda activate hcrn_videoqa
conda install -c conda-forge ffmpeg
conda install -c conda-forge scikit-video
pip install -r requirements.txt
```      
#### Training
Choose a suitable config file in `configs/{task}.yml` for one of 4 tasks: `action, transition, count, frameqa` to train the model. For example, to train with action task, run the following command:
```bash
python train.py --cfg configs/tgif_qa_action.yml
```
#### Evaluation
To evaluate the trained model, run the following:
```bash
python validate.py --cfg configs/tgif_qa_action.yml
```
