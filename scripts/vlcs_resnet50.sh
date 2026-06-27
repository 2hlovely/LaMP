#!/bin/bash

cd ..

DATASET_ROOT=/root/sj-tmp/Data4DG
DATASET=vlcs
NET=resnet50_clip
DATASET_YAML=vlcs_source_free
DEVICE=0
BATCH=128
NUM_WORKERS=4

SEED=1
I2_EPOCHS=100
LOAD_EPOCH=100

if [ ${DATASET} = "vlcs" ]; then
  D1=CALTECH
  D2=LABELME
  D3=PASCAL
  D4=SUN

  DATASET_NAME='VLCS_SF'
  LLM_DESC_PATH="templates/vlcs.json"
fi

UOT_EPSILON=0.005
UOT_TAU=0.001
UOT_ITERS=10
N_CTX=4
PROMPT_DEPTH_TEXT=1

  OUT_DIR=output_NCTX/${DATASET}/${NET}/train
  TEST_OUT_DIR=output_NCTX/${DATASET}/${NET}/test


  CUDA_VISIBLE_DEVICES=$DEVICE \
    python train.py \
    --root ${DATASET_ROOT} \
    --seed ${SEED} \
    --use_cuda True \
    --trainer LAMP \
    --source-domains none --target-domains ${D1} ${D2} ${D3} ${D4} \
    --dataset-config-file configs/datasets/dg/${DATASET_YAML}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir "${OUT_DIR}" \
    --txts_path dassl/txts \
    MODEL.BACKBONE.NAME ${NET} \
    DATALOADER.TRAIN_X.SAMPLER RandomSampler \
    DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
    OPTIM.MAX_EPOCH ${I2_EPOCHS} \
    DATASET.NAME ${DATASET_NAME} \
    OPTIM.LR 0.003 \
    DATASET.ROOT ${DATASET_ROOT} \
    TRAINER.LAMP.LLM_DESC_PATH ${LLM_DESC_PATH} \
    TRAINER.LAMP.N_CTX ${N_CTX} \
    TRAINER.LAMP.LAMBDA_MSE 1000.0 \
    TRAINER.LAMP.PROMPT_DEPTH_TEXT ${PROMPT_DEPTH_TEXT} \
    TRAINER.LAMP.UOT_EPSILON ${UOT_EPSILON} \
    TRAINER.LAMP.UOT_TAU ${UOT_TAU} \
    TRAINER.LAMP.UOT_ITERS ${UOT_ITERS}


  CUDA_VISIBLE_DEVICES=$DEVICE \
    python train.py \
    --root ${DATASET_ROOT} \
    --seed ${SEED} \
    --use_cuda True \
    --trainer LAMP \
    --eval-only \
    --model-dir "${OUT_DIR}" \
    --load-epoch ${LOAD_EPOCH} \
    --source-domains none --target-domains ${D1} ${D2} ${D3} ${D4} \
    --dataset-config-file configs/datasets/dg/${DATASET_YAML}.yaml \
    --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
    --output-dir "${TEST_OUT_DIR}" \
    --txts_path dassl/txts \
    MODEL.BACKBONE.NAME ${NET} \
    DATALOADER.TEST.BATCH_SIZE ${BATCH} \
    DATALOADER.NUM_WORKERS ${NUM_WORKERS} \
    DATASET.NAME ${DATASET_NAME} \
    DATASET.ROOT ${DATASET_ROOT} \
    TRAINER.LAMP.LLM_DESC_PATH ${LLM_DESC_PATH} \
    TRAINER.LAMP.N_CTX ${N_CTX} \
    TRAINER.LAMP.PROMPT_DEPTH_TEXT ${PROMPT_DEPTH_TEXT} \
    TRAINER.LAMP.UOT_EPSILON ${UOT_EPSILON} \
    TRAINER.LAMP.UOT_TAU ${UOT_TAU} \
    TRAINER.LAMP.UOT_ITERS ${UOT_ITERS}
