CUDA_VISIBLE_DEVICES=0 python train.py \
  --num_epochs 100 \
  --batch_size 128 \
  --num_bit 48 \
  --num_class 38 \
  --output_dir models/ISDH-48b \
  --tfrecords train-flickr.tfrecords
