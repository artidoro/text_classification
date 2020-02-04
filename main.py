import argparse
import torch

import model
import utils


if __name__ == '__main__':
    """
    main file to run all the models.

    First get the iteratorrs then pass them to the models for training.
    The models will output validation results during training.

    The models take different parameters (see model.py), they all have a train 
    method. 

    Note that for the last version of our model we trained on both validation and 
    and training data. We stopped the training process after the same number of 
    iterations that gave the optimal result when just training on training data.
    Alternatively we could have used cross validation on both training and 
    validation data. This did not significantly increase our accuracy.

    """
    parser = argparse.ArgumentParser(description='Arguments for the text classification model.')
    parser.add_argument('--train_path', default='../topicclass/topicclass_train.txt')
    parser.add_argument('--valid_path', default='../topicclass/topicclass_valid.txt')
    parser.add_argument('--test_path', default='../topicclass/topicclass_test.txt')
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--embed_size', default=300, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--train_epochs', default=20, type=int)
    parser.add_argument('--eval_epochs', default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--device', default='cpu', help='select cuda for the gpu')
    parser.add_argument('--model_name', default='cnn')
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--load_checkpoint_path', default=None)
    parser.add_argument('--overwrite_args', action='store_true')
    args = vars(parser.parse_args())

    train_iter, val_iter, test_iter, TEXT, LABEL = utils.torchtext_iterators(
        args['train_path'], args['valid_path'], args['test_path'],
        batch_size=args['batch_size'], device=torch.device(args['device']))

    utils.train(train_iter, val_iter, TEXT, LABEL, args)