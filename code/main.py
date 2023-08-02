from DataReader import load_training_data, load_public_test_data, load_private_test_data
from Model import Cifar
import torch
import os
import numpy as np
import argparse
from torchsummary import summary
import ImageUtils

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet_size", type=int, default=6, 
        help='n: number of standard blocks in a stack layer')
    parser.add_argument("--first_num_filters", type=int, default=64, 
        help='number of filter used in the first convolutional layer')
    parser.add_argument("--num_classes", type=int, default=10,
        help='number of classes')

    parser.add_argument("--initial_batch_size", type=int, default=125, 
        help='initial training mini batch size, increases when loss plateaus')
    parser.add_argument("--batch_size_increase", type=int, default=2, 
        help='increase mini batch size by this factor when loss plateaus')
    parser.add_argument("--weight_decay", type=float, default=2e-4,
        help='weight decay constant for L2 regularization')
    parser.add_argument("--checkpoint_dir", type=str, default='../saved_models',
        help='checkopoint directory')
    parser.add_argument("--learning_rate", type=float, default=0.1, 
        help='learning rate')

    parser.add_argument("--mode", type=str, default='train', help='train or evaluate or predict')
    parser.add_argument("--save_interval", type=int, default=10,
        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--resume_checkpoint", type=int, default=0, 
        help='resumes training from this checkpoint')
    return parser.parse_args()

def main(config):

    model = Cifar(config).cuda()

    print("--- Preparing Data ---")
    if config.mode == 'train':
        x_train, y_train = load_training_data()
        model.train(x_train, y_train, max_epoch=90)
    elif config.mode == 'evaluate':
        x_valid, y_valid = load_public_test_data()
        model.evaluate(x_valid, y_valid, evaluate_checkpoints=[90, 80, 70, 60, 30])
    elif config.mode == 'predict':
        x_test = load_private_test_data()
        model.predict_probability(x_test, checkpoint=90)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    config = configure()
    main(config)