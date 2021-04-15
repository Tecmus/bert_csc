# export BERT_BASE_DIR=/home/azureuser/projects/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12
# wget -O ./model/chinese_L-12_H-768_A-12.zip  -c https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip  
# cd ./model
# unzip chinese_L-12_H-768_A-12.zip
# cd ..
# export BERT_BASE_DIR=/kaggle/working/bert_csc/csc/model/chinese_L-12_H-768_A-12
export BERT_BASE_DIR=/root/model/chinese_L-12_H-768_A-12

PYTHONIOENCODING=utf-8 python run_pretraining.py \
  --input_file=./record_data/ocr_test_no_shuf.tfrecord \
  --output_dir=./output_2k \
  --do_predict=True \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=./output_2k/model.ckpt-10000 \
  --train_batch_size=20 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=10000 \
  --num_warmup_steps=100 \
  --learning_rate=2e-5 \
  --top_k=1