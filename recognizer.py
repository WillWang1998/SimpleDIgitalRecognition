import tensorflow as tf
import mnist_cnn as mnist_inference


class Recognizer:
    def __init__(self, ckpt_path="./model/model-30000"):
        print(ckpt_path)
        self.x = tf.placeholder(tf.float32, shape=[None,
                                                   mnist_inference.IMAGE_SIZE,
                                                   mnist_inference.IMAGE_SIZE,
                                                   mnist_inference.NUM_CHANNEL], name='x-input')
        self.y = mnist_inference.inference(self.x, False, None)
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, ckpt_path)

    def recognize(self, data):
        prediction = tf.argmax(self.y, 1)
        predict_value = prediction.eval(feed_dict={self.x: data}, session=self.sess)
        # print(type(predict_value[0]))
        return predict_value[0]
