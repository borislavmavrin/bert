#!/bin/bash
EXPERIMENT_DIR=experiments/lr2e5_b32
#rm -rf $EXPERIMENT_DIR

BERT_BASE_DIR=model/bert_base_uncased
DATA_DIR=data
if [ ! -d $BERT_BASE_DIR ]; then
  mkdir -p $BERT_BASE_DIR
  wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip -P $BERT_BASE_DIR
  unzip $BERT_BASE_DIR/uncased_L-12_H-768_A-12.zip -d $BERT_BASE_DIR
fi

if [ ! -f $DATA_DIR/tf_examples.tfrecord ]; then
  python create_pretraining_data.py \
    --input_file=./sample_text.txt \
    --output_file=$DATA_DIR/tf_examples.tfrecord \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5
fi

python run_pretraining.py \
  --input_file=$DATA_DIR/tf_examples.tfrecord \
  --output_dir=$EXPERIMENT_DIR \
  --do_train=True \
  --do_eval=False \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5