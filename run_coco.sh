#!/usr/bin/env bash

EXPNAME=exp_coco_CosSimNegHead
CONFIG_NAME=coco_CosSimNegHead
GPUS=8

SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN=checkpoints/MSRA/R-101.pkl                            # <-- change it to you path

# ------------------------------- Base Pre-train ---------------------------------- #
python3 main.py --num-gpus $GPUS --config-file configs/${CONFIG_NAME}/bsdet_det_r101_base.yaml     \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                         \
           OUTPUT_DIR ${SAVEDIR}/bsdet_det_r101_base

# ----------------------------- Model Preparation --------------------------------- #
python3 tools/model_surgery.py --dataset coco --method randinit                        \
    --src-path ${SAVEDIR}/bsdet_det_r101_base/model_final.pth                         \
    --save-dir ${SAVEDIR}/bsdet_det_r101_base
BASE_WEIGHT=${SAVEDIR}/bsdet_det_r101_base/model_reset_surgery.pth

# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
for seed in 0
do
    for shot in 1 2 3 5 10 30
    do
        python3 tools/create_config.py --dataset coco14 --config_root configs/${CONFIG_NAME}     \
            --shot ${shot} --seed ${seed} --setting 'gfsod'
        CONFIG_PATH=configs/${CONFIG_NAME}/bsdet_gfsod_r101_novel_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVEDIR}/bsdet_gfsod_r101_novel/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus $GPUS --config-file ${CONFIG_PATH}                      \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done
python3 tools/extract_results.py --res-dir ${SAVEDIR}/bsdet_gfsod_r101_novel/tfa-like --shot-list 1 2 3 5 10 30  # surmarize all results
