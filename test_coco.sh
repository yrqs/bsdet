#!/usr/bin/env bash

CONFIG_NAME=coco_CosSimNegHead
CHECKPOINT=checkpoints/coco/path/to/model_final.pth
shot=10
GPUS=1

seed=0
if [ ${shot} -eq 0 ]; then
  CONFIG_PATH=configs/${CONFIG_NAME}/bsdet_det_r101_base.yaml
else
  python3 tools/create_config.py --dataset coco14 --config_root configs/${CONFIG_NAME}               \
      --shot ${shot} --seed ${seed} --setting 'gfsod'
  CONFIG_PATH=configs/${CONFIG_NAME}/bsdet_gfsod_r101_novel_${shot}shot_seed${seed}.yaml
fi

python3 main.py --num-gpus $GPUS --config-file ${CONFIG_PATH}     \
    --eval-only \
    --opts MODEL.WEIGHTS ${CHECKPOINT}  \
           OUTPUT_DIR mytest;

rm $CONFIG_PATH;