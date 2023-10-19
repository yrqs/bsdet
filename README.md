# BSDet: Binary Similarity Detector
***

### Quick Start
#### 1. Check Requirements
* Linux with Python >= 3.6
* PyTorch >= 1.6 & torchvision that matches the PyTorch version. 
* CUDA 10.1, 10.2, 11.0 (maybe even higher)
* GCC >= 4.9

#### 2. Build BSDet
* Clone Code
* Create a virtual environment (optional)
  ```
  conda create -n bsdet python=3.7
  source activate bsdet
  ```
* Install PyTorch with CUDA
  ```
  conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
  ```
* Install Detectron2
  ```
  python3 -m pip install detectron2==0.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
  ```
* Install other requirements
  ```
  python3 -m pip install -r requirements.txt
  ```
  
#### 3. Prepare Data
* Refer to [DeFRCN](https://github.com/er-muyue/DeFRCN) and [TFA](https://github.com/ucbdrive/few-shot-object-detection#models)
* The final dataset file structure is as follows:
  ```
  ...
  datasets
    | -- coco
          | -- trainval2014/*.jpg
          | -- val2014/*.jpg
          | -- annotations/*.json
    | -- cocosplit
    | -- VOC2007
    | -- VOC2012
    | -- vocsplit
  defrcn
  tools
  ...
  ```

#### 4. Config Files
***
* BSDet config files:
  ```
  configs 
    | -- coco_CosSimNegHead
    | -- voc_CosSimNegHead
  ```
  
#### 5. Training and Evalution
* VOC
  ```
  bash run_voc.sh
  ```
  * EXP_NAME: save directory
  * CONFIG_NAME: used config directory
  * SPLIT_ID: id of voc split
  * GPUS: number of used GPUs

* COCO
  ```
  bash run_coco.sh
  ```
  * EXP_NAME: save directory
  * CONFIG_NAME: used config directory
  * GPUS: number of used GPUs
  
#### 6. Test
* VOC
  ```
  bash test_voc.sh
  ```
  * CONFIG_NAME: used config directory
  * CHECKPOINT: used checkpoint
  * shot: number of novel class samples
  * SPLIT_ID: id of voc split
  * GPUS: number of used GPUs
  
* COCO
  ```
  bash test_coco.sh
  ```
  * CONFIG_NAME: used config directory
  * CHECKPOINT: used checkpoint
  * shot: number of novel class samples
  * GPUS: number of used GPUs
#### 7. FSOD Test Setting

There are two main types of FSOD test, single test and multiple test. We adopt the single test (such as [MPSR](https://github.com/jiaxi-wu/MPSR)). It is worth noting that DeFRCN has two fine-tuning settings: one predicts only novel classes during fine-tuning (fsod), and the other predicts both base classes and novel classes (gfsod). To remain consistent with other existing works (such as [MPSR](https://github.com/jiaxi-wu/MPSR)), we use the gfsod setting. In addition, we only run novel class samples with seed 0, as it is consistent with the samples used in existing single-test works. The mAP of DeFRCN compared in the paper comes from [DeFRCN](https://github.com/er-muyue/DeFRCN) (the result of seed 0 in the log file), therefore it is not consistent with the mAP reported in the DeFRCN paper. However, this is a fair comparison.


### Acknowledgement
This repo is developed based on [DeFRCN](https://github.com/er-muyue/DeFRCN) and [Detectron2](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Ffacebookresearch%2Fdetectron2). Please check them for more details and features.