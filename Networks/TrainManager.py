import tensorflow as tf
import numpy as np


class TrainTestHelper:
    """
    Manage the train step
    """
    def __init__(self, model, optimizer, loss_func, training=True):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.loss_logger = tf.keras.metrics.Mean(name='loss')
        self.training = training
        self.steps =  []

    def update_optimizer(self, optimizer):
        self.optimizer = optimizer



    def get_step(self):

        @tf.function()
        def train_step(inputs, labels):
            with tf.GradientTape(persistent=True) as tape:
                self.steps.append(0)

                prediction = self.model(inputs, training=self.training)
                self.steps.append(1)

                loss_value = self.loss_func(labels, prediction)
                self.steps.append(3)

                self.loss_logger(loss_value)
                self.steps.append(4)


            if self.training:
                self.steps.append(6)

                grads = tape.gradient(loss_value, self.model.trainable_variables)
                self.steps.append(7)

                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                self.steps.append(8)

        return train_step

    def __del__(self):
        with open("text.txt", 'w') as f:
            f.write(str(self.steps))


