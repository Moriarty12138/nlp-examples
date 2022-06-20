set -e
#set hyperparameters
BERT_DIR=/path/to/BERT
SQUAD_DIR=./datasets

accu=2

ep=3
eps=1 #1 #e-2
si=1
lr=3

batch_size=24
semi_batch_size=0

seed=1337
torch_seed=${seed}
numpy_seed=${seed}

NAME=BertBaseVAT_lr${lr}eps${eps}si${si}e${ep}_s${seed}b${batch_size}
OUTPUT_DIR=${OUTPUT_BASE_DIR}/${NAME}
mkdir -p $OUTPUT_DIR

python -u main.py \
    --model_name_or_path bertbase \
    --model_type bert \
    --vocab_file $BERT_DIR/vocab.txt \
    --config_file $BERT_DIR/bert_config.json \
    --init_checkpoint $BERT_DIR/pytorch_model.bin \
    --do_train \
    --do_eval \
    --do_predict \
    --doc_stride 128 \
    --train_batch_size ${batch_size} \
    --random_seed $torch_seed \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-5 \
    --ckpt_frequency 2 \
    --schedule slanted_triangular \
    --s_opt1 30 \
    --output_dir $OUTPUT_DIR \
    --gradient_accumulation_steps ${accu} \
    --epsilon ${eps}e-2 \
    --si ${si}e-5 \
    --enable_VAT
