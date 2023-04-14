# AVQA
This is the official repo for our ACM Multimedia 2022 paper [AVQA: A Dataset for Audio-Visual Question Answering on Videos](https://dl.acm.org/doi/10.1145/3503161.3548291).

Dataset Website: [https://mn.cs.tsinghua.edu.cn/avqa](https://mn.cs.tsinghua.edu.cn/avqa)

## AVQA Dataset

## Description
### Repo directories
- ./configs: config files;
- ./data: data dictionary;
- ./preprocess: code and scripts for feature extraction;
- ./backbones: six backbone models -- PSAC, HME, LADNet, ACRTransformer, HGA, HCRN;
- ./HAVF: our proposed HAVF model to reproduce the results;
- ./runs: the default output dictionary used to store our trained model and result files;
- ./scripts: all training and evaluating scripts;
- ./requirements: requirement.txt files for each backbone models.

## Usage
### Before we start
1. Clone this repo
    ```
    git clone git@github.com:AlyssaYoung/AVQA.git
    ```
2. Download data
    You can download the raw videos and extract features according to your needs. Besides, you can also directly use the features we provide. More detailed information for downloading data can be found in the #Downloads Section of the [Dataset Website](https://mn.cs.tsinghua.edu.cn/avqa).

3. Data preprocessing
    - Extract audio waveforms.
    - Extract visual frames.

4. Feature extraction
    - Audio feature.
    - Appearance feature.
    - Motion feature.

### Backbone models
#### PSAC
Train with the PSAC baseline.
```
python xxx.py --mode train
```
Train with the PSAC+HAVF model.
```
python xxx.py --mode train
```
Testing.
```
python xxx.py --mode test
```
#### HME
#### LADNet
#### ACRTransformer
#### HGA
#### HCRN

## Citation
If you find our paper or code useful, please cite our paper using the following bibtex:
```
@inproceedings{yang2022avqa,
  title={AVQA: A Dataset for Audio-Visual Question Answering on Videos},
  author={Yang, Pinci and Wang, Xin and Duan, Xuguang and Chen, Hong and Hou, Runze and Jin, Cong and Zhu, Wenwu},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={3480--3491},
  year={2022}
}
```