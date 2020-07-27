
import argparse
import tensorflow as tf
import os
from DataLoader import DataLoader
from Networks.TrainManager import TrainTestHelper
from traintest import train

import nn_builder

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nntype', default="VGGModel", help='The type of the network')
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--val_file', required=True)
    parser.add_argument('--model_path', default='vgg_16.ckpt', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs1', default=10, type=int)
    parser.add_argument('--num_epochs2', default=10, type=int)
    parser.add_argument('--learning_rate1', default=1e-3, type=float)
    parser.add_argument('--learning_rate2', default=1e-5, type=float)
    parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--output_path', '-o', type=str )
    parser.add_argument('--cls_num', type=int, default=82, help='The number of classes in the dataset')
    parser.add_argument('--input_size', type=int, nargs=2, default=(224, 224))
    parser.add_argument('--print_freq', type=int, default=500)




    return parser.parse_args()



def main():
    tf.keras.backend.set_floatx('float32')
    args = get_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


    train_dataloader = DataLoader("train_dataset", args.train_file, args.cls_num, args.input_size, args.output_path, augment=True)
    val_dataloader = DataLoader("val_dataset", args.val_file, args.cls_num, args.input_size, args.output_path, augment=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate1)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    network = nn_builder.get_network(args.nntype)

    trainer = TrainTestHelper(network, optimizer, loss, training=True)
    validator = TrainTestHelper(network, optimizer, loss, training=False)
    print("in")

    train(args.num_epochs1, args.batch_size, trainer, validator, train_dataloader, val_dataloader, args.print_freq, args.output_path)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate2)
    trainer.update_optimizer(optimizer)
    validator.update_optimizer(optimizer)
    train(args.num_epochs1, args.batch_size, trainer, validator, train_dataloader, val_dataloader, args.print_freq, args.output_path)

