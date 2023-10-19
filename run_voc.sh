#!/usr/bin/env bash

EXP_NAME=exp_voc_CosSimNegHead
CONFIG_NAME=voc_CosSimNegHead
SPLIT_ID=1
GPUS=1

SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=checkpoints/MSRA/R-101.pkl                            # <-- change it to you path

# ------------------------------- Base Pre-train ---------------------------------- #
python3 main.py --num-gpus $GPUS --config-file configs/${CONFIG_NAME}/bsdet_det_r101_base${SPLIT_ID}.yaml     \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                   \
           OUTPUT_DIR ${SAVE_DIR}/bsdet_det_r101_base${SPLIT_ID} && sleep 1s;

# ----------------------------- Model Preparation --------------------------------- #
python3 tools/model_surgery.py --dataset voc --method randinit                                \
    --src-path ${SAVE_DIR}/bsdet_det_r101_base${SPLIT_ID}/model_final.pth                    \
    --save-dir ${SAVE_DIR}/bsdet_det_r101_base${SPLIT_ID}
BASE_WEIGHT=${SAVE_DIR}/bsdet_det_r101_base${SPLIT_ID}/model_reset_surgery.pth;

# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
for seed in 0
do
    for shot in 1 2 3 5 10 # if final, 10 -> 1 2 3 5 10
    do
        python3 tools/create_config.py --dataset voc --config_root configs/${CONFIG_NAME}               \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
        CONFIG_PATH=configs/${CONFIG_NAME}/bsdet_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/bsdet_gfsod_r101_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus $GPUS --config-file ${CONFIG_PATH}                            \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth;
    done
done
python3 tools/extract_results.py --res-dir ${SAVE_DIR}/bsdet_gfsod_r101_novel${SPLIT_ID}/tfa-like --shot-list 1 2 3 5 10;  # surmarize all results
