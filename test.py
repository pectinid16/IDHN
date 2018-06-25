import os
import numpy as np
import tensorflow as tf
import Image
import scipy.io as sio
from alexnet import AlexNet
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=50, help="number of images in batch")
parser.add_argument("--num_bit", required=True, type=int, help="number of hash bits")
parser.add_argument("--num_class", type=int, help="number of class")

parser.add_argument("--keep_prob", type=float, default=1, help="dropout rate")
parser.add_argument("--img_size", type=int, default=227, help="image size of input")
parser.add_argument("--checkpoint", required=True, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--model_name", required=True, help="which model to be evaluated")
parser.add_argument("--img_dir", required=True, help="directory of input images")
parser.add_argument("--img_file", required=True, help="test image file")
parser.add_argument("--data_type", required=True)
parser.add_argument("--output_dir", required=True, default="results/", help="where to put output images")
args = parser.parse_args()

skip_layers = ['fc8']
mean_value = np.array([123, 117, 104]).reshape((1, 3))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def toBinaryString(binary_like_values):
    numOfImage, bit_length = binary_like_values.shape
    list_string_binary = []
    for i in range(numOfImage):
        str = ''
        for j in range(bit_length):
            str += '0' if binary_like_values[i][j] <= 0 else '1'
        list_string_binary.append(str)
    return list_string_binary


def evaluate():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image = tf.placeholder(tf.float32, [args.batch_size, args.img_size, args.img_size, 3], name='image')
    model = AlexNet(image, args.keep_prob, args.num_bit, args.num_class, skip_layers)

    D = model.softsign

    f = open(args.img_file)
    lines = f.readlines()
    l = len(lines)

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.global_variables())
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        if args.checkpoint is not None:
            checkpoint = tf.train.latest_checkpoint(args.checkpoint)
	    print('Restoring model from {}'.format(checkpoint))
            saver.restore(sess, checkpoint)
            print('Loading success')

            res = open(args.output_dir+args.model_name+'_'+args.data_type+'.txt', 'w')
            for i in range(int(math.ceil(float(l)/args.batch_size))):
                print i
                data = np.zeros([args.batch_size, args.img_size, args.img_size, 3], np.float32)
                for j in range(args.batch_size):
                    if j + i * args.batch_size < l:
                        img_name = lines[j + i * args.batch_size].strip('\n\r')
                        img_path = args.img_dir + img_name
                        img = Image.open(img_path)
                        img = img.resize((args.img_size, args.img_size))
                        new_im = img - mean_value
                        new_im = new_im.astype(np.int16)
                        data[j, :, :, :] = new_im
                eval_sess = sess.run(D, feed_dict={image: data})
                w_res = toBinaryString(eval_sess)
                for j in range(args.batch_size):
                    res.write(w_res[j] + '\n')
            res.close()


if __name__ == '__main__':
    evaluate()
