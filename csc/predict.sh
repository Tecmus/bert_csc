# export BERT_BASE_DIR=/home/azureuser/projects/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12
# wget -O ./model/chinese_L-12_H-768_A-12.zip  -c https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip  
# cd ./model
# unzip chinese_L-12_H-768_A-12.zip
# cd ..
# export BERT_BASE_DIR=/kaggle/working/bert_csc/csc/model/chinese_L-12_H-768_A-12
export BERT_BASE_DIR=/root/model/chinese_L-12_H-768_A-12

PYTHONIOENCODING=utf-8 python csc_predictor.py \
  --model_dir=./save_model \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128