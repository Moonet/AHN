export GLUE_DIR=./data

PYTHONPATH=../ CUDA_VISIBLE_DEVICES=1,2,3 python ahn_lstm.py \
  --task_name YELP \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/amazon_auto/amau_5core/ \
  --num_head 1 \
  --num_reviews 12 \
  --num_sentence 12 \
  --max_seq_length 24 \
  --train_batch_size 128 \
  --learning_rate 2e-4 \
  --num_train_epochs 6.0 \
  --vocab_file vocab.txt \
  --cache_dir  ./data_cache/amau/amau_5core/ \
  --output_dir /tmp/amau_ahn_c5_output/
