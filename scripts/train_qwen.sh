#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

doc_path="./data/wuthering_heights.txt"
seq_len=64
batch_size=16
epochs=10
lr=1e-4
save_path="./qwen_model_8_10"

python ./train_qwen.py \
    --doc_path $doc_path \
    --seq_len $seq_len \
    --batch_size $batch_size \
    --epochs $epochs \
    --lr $lr \
    --save_path $save_path
