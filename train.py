import os
import tensorflow as tf
import reader
import time
from alexnet import AlexNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="number of images in batch")
parser.add_argument("--num_bit", type=int, help="number of hash bits")
parser.add_argument("--num_class", type=int, help="number of object calss")

parser.add_argument("--dropout_rate", type=float, default=0.5, help="dropout rate")
parser.add_argument("--alpha", type=float, default=0.1, help="weight on regularizer term")
parser.add_argument("--belta", type=float, default=5, help="threshold to limit the range of value")
parser.add_argument("--gama", type=float, default=0.1, help="weight on pairwise similar or dissimilar")
parser.add_argument("--img_size", type=int, default=227, help="image size of input")

parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate for adam")
parser.add_argument("--decay_step", type=int, default=500, help="number of steps to dacay lreaning rate")
parser.add_argument("--decay_rate", type=float, default=0.5, help="decaying rate")

parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from")
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps")
parser.add_argument("--tfrecords", required=True, help="training dataset")
args = parser.parse_args()
# List of names of the layer trained from scratch
skip_layers = ['fc8']

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# hash loss function 
def hashing_loss(D, label, alpha, belta, gama, m):
    label_count = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(label), 1)),1)
    norm_label = label/tf.tile(label_count,[1,args.num_class])
    w_label = tf.matmul(norm_label, norm_label, False, True)
    semi_label = tf.where(w_label>0.99, w_label-w_label,w_label)
    p2_distance = tf.matmul(D, D, False, True)
    
    scale_distance = belta * p2_distance / m
    temp = tf.log(1+tf.exp(scale_distance))
    
    loss = tf.where(semi_label<0.01,temp - w_label * scale_distance, gama*m*tf.square((p2_distance+m)/2/m-w_label))
    regularizer = tf.reduce_mean(tf.abs(tf.abs(D) - 1))
    d_loss = tf.reduce_mean(loss) + alpha * regularizer 
    return d_loss,w_label

def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # read data from tfrecord files
    img, label = reader.read_and_decode(args.tfrecords, epochs=args.num_epochs, size=args.img_size)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=args.batch_size, capacity=2000,
                                                    min_after_dequeue=1000)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    # construct network model
    model = AlexNet(img_batch, args.dropout_rate,  args.num_bit, args.num_class, skip_layers)
    
    D = model.softsign
    [d_loss,out]= hashing_loss(D, label_batch, args.alpha, args.belta, args.gama, args.num_bit)
    
    # List of trainable variables of the layers to finetune
    var_list1 = [v for v in tf.trainable_variables() if v.name.split('/')[0] not in skip_layers]
    # List of trainable variables of the layers to train from scratch
    var_list2 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in skip_layers]

    # learning rate 
    learning_rate = tf.train.exponential_decay(args.lr, global_step, args.decay_step,  args.decay_rate, staircase=True)
    opt1 = tf.train.AdamOptimizer(learning_rate * 0.01)
    opt2 = tf.train.AdamOptimizer(learning_rate)

    # apply different grads for two type layers 
    grads = tf.gradients(d_loss, var_list1+var_list2)
    grads1 = grads[:len(var_list1)]
    grads2 = grads[len(var_list1):] 
    train_op1 = opt1.apply_gradients(zip(grads1,var_list1))
    train_op2 = opt2.apply_gradients(zip(grads2,var_list2), global_step=global_step)
    train_op = tf.group(train_op1, train_op2)    

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.global_variables())
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        if args.checkpoint is not None:
            checkpoint = tf.train.latest_checkpoint(args.checkpoint)
            print('Restoring model from {}'.format(checkpoint))
            saver.restore(sess, checkpoint)
        else:
            # Load the pretrained weights into the non-trainable layer
            model.load_initial_weights(sess)  

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        try:
            while not coord.should_stop():
                _, loss_t,dt, step1 = sess.run([train_op, d_loss,out, global_step])
                elapsed_time = time.time() - start_time
                start_time = time.time()

                if step1 % 10 == 0:
                    print("iter: %4d, loss: %.8f, time: %.3f" % (step1, loss_t, elapsed_time))
                if step1 % args.save_freq == 0:
                    saver.save(sess, args.output_dir + '/model.ckpt', global_step=step1)

        except tf.errors.OutOfRangeError:
            saver.save(sess, args.output_dir + '/model-done.ckpt')
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
  main()
