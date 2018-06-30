CUDA_VISIBLE_DEVICES=0 python test.py \
  --batch_size  50 \
  --num_bit 48 \
  --num_class 38 \
  --checkpoint models/IDHN-48b \
  --model_name ISDH-48b \
  --img_dir ./data/flickr/images/ \
  --img_file ./data/flickr/test.txt \
  --data_type test_flickr \
  --output_dir ./results/ 
