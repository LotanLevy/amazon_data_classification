from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NNInterface import NNInterface
from tensorflow.python.keras.applications import vgg16




import tensorflow as tf
import os


class VGGModel(NNInterface):
    def __init__(self):
        super().__init__()
        self.__model = vgg16.VGG16(weights='imagenet')
        self.__model.summary()


    def call(self, x, training=True):
        x = vgg16.preprocess_input(x)
        return self.__model(x, training=training)

    def compute_output_shape(self, input_shape):
        return self.__model.compute_output_shape(input_shape)

    def freeze_layers(self, freeze_idx):

        for i, layer in enumerate(self.__model.layers):
            if freeze_idx > i:
                layer.trainable = False
            else:
                layer.trainable = True


        for i, layer in enumerate(self.__model.layers):
            print("layer {} is trainable {}".format(layer.name, layer.trainable))

    def save_model(self, iter_num, output_path):
        output_path = os.path.join(output_path, "ckpts")
        checkpoint_path = "weights_after_{}_iterations".format(iter_num)
        self.__model.save_weights(os.path.join(output_path, checkpoint_path))

    def load_model(self, ckpt_path):
        self.__model.load_weights(ckpt_path)