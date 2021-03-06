# IDHN
Code for the following paper:

Z. Zhang and Q. Zou and Y. Lin and L. Chen and S. Wang, ["Improved Deep Hashing with Soft Pairwise Similarity for Multi-label Image Retrieval"](https://arxiv.org/abs/1803.02987), IEEE Transactions on Multimedia, 2019. 

### Requirements
- [Linux](https://www.ubuntu.com/download)
- [Tensorflow](https://www.tensorflow.org/)
- [NVIDIA GPU + CUDA CuDNN](https://developer.nvidia.com/cudnn)

### Getting Started:
- Prepare the datasets  
Download the [flickr dataset](http://press.liacs.nl/mirflickr/) and put the images into folder  `/data/flickr/images/`
  
- Transform the train images in tfrecord format  
Run `python tf_record.py`, and `train-flickr.tfrecords` will be generated
         
- Prepare the AlexNet weights trained on ImageNet  
Download from [here](ww.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) and put it on current directory
   
- Train:  
Run `sh train-flickr.sh`, and the trained model will be saved in `models/IDHN-48b/`.

- Test:  
Run `sh test-flickr.sh`, and generated hash codes will be saved in `./results/IDHN_48b_test_flickr.txt`.

### Citation
Please cite our paper if you use this code in your own work:
```
@article{zhang2019tmm,
   author = {Zhang, Zheng and Zou, Qin and Lin, Yuewei and Chen, Long and Wang, Song},
   title = {Improved deep hashing with soft pairwise similarity for multi-label image retrieval},
   journal = {IEEE Transactions on Multimedia},
   year = {2019}
}
```
