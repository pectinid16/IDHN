# IDHN

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

### Acknowledgement
- [Tensorflow Implementation of AlexNet](https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html) 
