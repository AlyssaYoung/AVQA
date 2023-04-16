# HCRN+HAVF

#### Install dependencies:
```bash
conda create -n hcrn_havf python=3.6
conda activate hcrn_havf
pip install -r requirements.txt
```      
#### Training
```bash
python train.py --cfg configs/tgif_qa_action.yml
```
#### Evaluation
To evaluate the trained model, run the following:
```bash
python validate.py --cfg configs/tgif_qa_action.yml
```
