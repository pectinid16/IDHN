import tensorflow as tf
from PIL import Image
import scipy.io as sio
import numpy as np

#img_dir = './data/flickr/images/'
img_dir = '../flickr-25k/im256/'
label_dir = './data/flickr/trainlabel.mat'
image_size = 227

f1 = open('./data/flickr/train.txt')
imagelists = f1.readlines()
l = len(imagelists)

mean_value = np.array([123, 117, 104]).reshape((1,3))
data = sio.loadmat(label_dir)

train_label = data['trainlabel']
writer = tf.python_io.TFRecordWriter("train-flickr.tfrecords")

for i in np.arange(l):
    img_name = imagelists[i].strip('\n\r')
    img_path = img_dir + img_name
    img = Image.open(img_path)
    img = img.resize((image_size, image_size))
    new_im = img-mean_value
    new_im =new_im.astype(np.int16)
    img_raw = new_im.tobytes()
    
    feature = train_label[i,:]
    print(img_name)
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=feature)),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    writer.write(example.SerializeToString())
writer.close()
