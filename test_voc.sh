#!/usr/bin/env bash

CONFIG_NAME=voc_CosSimNegHead
CHECKPOINT=checkpoints/voc/path/to/model_final.pth
shot=1
SPLIT_ID=2
GPUS=2

seed=0
if [ ${shot} -eq 0 ]; then
  CONFIG_PATH=configs/${CONFIG_NAME}/bsdet_det_r101_base${SPLIT_ID}.yaml
else
  python3 tools/create_config.py --dataset voc --config_root configs/${CONFIG_NAME}               \
      --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
  CONFIG_PATH=configs/${CONFIG_NAME}/bsdet_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
fi

python3 main.py --num-gpus $GPUS --config-file ${CONFIG_PATH}     \
    --eval-only \
    --opts MODEL.WEIGHTS ${CHECKPOINT}  \
           OUTPUT_DIR mytest;

rm $CONFIG_PATH;