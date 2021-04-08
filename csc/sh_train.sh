export BERT_BASE_DIR=/home/azureuser/projects/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12

wget -O ./model/chinese_L-12_H-768_A-12.zip  -c https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip  
cd ./model
unzip chinese_L-12_H-768_A-12.zip
cd ..
export BERT_BASE_DIR=/kaggle/working/bert_csc/csc/model/chinese_L-12_H-768_A-12

python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=10000 \
  --num_warmup_steps=100 \
  --learning_rate=2e-5