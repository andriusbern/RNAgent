import re
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
# import cv2
import numpy as np

class GraphInspector(object):
    """
    A class for inspecting tensorflow graphs
    """
    def __init__(self, graph, sess):
        self.graph = graph
        self.sess = sess
        self.trainable = graph.get_collection('trainable_variables')

    def get_filters(self, name):
        """
        Get filters by name
        """
        filters = []
        for var in self.trainable:
            if len(re.findall('{}./w:'.format(name), var.name)) > 0:
                filters.append(var)

        return filters

    def get_biases(self, name):
        biases = []
        for bias in self.trainable:
            if len(re.findall('{}./b:'.format(name), bias.name)) > 0:
                biases.append(bias)

        return biases

    def visualize_filters(self, filt):
        """
        Visualize the filters learned by the CNN
        """
        name = 'Filter view'
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 800, 800)

        print('Name: {}, Shape: {}'.format(filt.name, filt.shape))
        proc = tf.squeeze(filt, name=filt.name.split('/')[1])

        shape = proc.shape
        print(len(proc.shape))
        if len(proc.shape) == 3:
            values = self.sess.run(proc)
            image = ((values + 1) * 120).astype(np.uint8)
            for i in range(shape[2]):
                cv2.imshow(name, image[:,:,i])
                cv2.waitKey(10)
                input('Filter #{}'.format(i))
        else:
            values = self.sess.run(proc)
            image = ((values + 1)*120).astype(np.uint8)
            cv2.imshow(name, image)
            cv2.waitKey(10)
            input()
        cv2.destroyAllWindows()

    

