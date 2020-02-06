import argparse
import torch
import os
import logging
import pprint
import sys
import time

import model
import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for the text classification model.')
    parser.add_argument('--train_path', default='../topicclass/topicclass_train.txt')
    parser.add_argument('--valid_path', default='../topicclass/topicclass_valid.txt')
    parser.add_argument('--test_path', default='../topicclass/topicclass_test.txt')

    parser.add_argument('--model_name', default='cnn')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--embed_size', default=300, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--train_epochs', default=20, type=int)
    parser.add_argument('--eval_epochs', default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--device', default='cpu', help='select cuda for the gpu')
    parser.add_argument('--no_pretrained_vectors', action='store_true')

    parser.add_argument('--checkpoint_path')
    parser.add_argument('--load_checkpoint_path', default=None)
    parser.add_argument('--overwrite_args', action='store_true')
    parser.add_argument('--mode', default='train')
    args = vars(parser.parse_args())

    checkpoint_path = os.path.join('log', args['checkpoint_path'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(checkpoint_path, 'log.log'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info('Starting to train text classification model with args:\n{}'.format(pprint.pformat(args)))

    train_iter, val_iter, test_iter, TEXT, LABEL = utils.torchtext_iterators(args, load_vectors_manually=True)

    # Initialize model and optimizer. This requires loading checkpoint if specified in the arguments.
    if args['load_checkpoint_path'] == None:
        module = model.model_dict[args['model_name']](TEXT, LABEL, args)
        module.to(torch.device(args['device']))
        optimizer = model.optimizer_dict[args['optimizer']](filter(lambda p: p.requires_grad, module.parameters()), args['lr'])
    else:
        # Load checkpoint.
        logger.info('Loading checkpoint {}'.format(args['load_checkpoint_path']))
        checkpoint_path = os.path.join('log', args['load_checkpoint_path'])
        assert os.path.exists(checkpoint_path), 'checkpoint path does not exists'
        checkpoint = torch.load(checkpoint_path)

        # Overwrite arguments if required.
        if args['overwrite_args']:
            args = checkpoint['args']

        # Extract parameters.
        module = model.model_dict[checkpoint['args']['model_name']](TEXT, LABEL, args)
        module.to(torch.device(args['device']))
        optimizer = model.optimizer_dict[args['optimizer']](filter(lambda p: p.requires_grad, module.parameters()), args['lr'])
        module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if args['mode'] == 'train':
        logger.info('Starting training.')
        utils.train(module, optimizer, train_iter, val_iter, args)

    elif args['mode'] == 'eval':
        logger.info('Starting evaluation.')
        evaluation_results = {}
        evaluation_results['train'] = utils.eval(module, train_iter)
        evaluation_results['valid'] = utils.eval(module, val_iter)
        logger.info('\n' + pprint.pformat(evaluation_results))

    elif args['mode'] == 'test':
        logger.info('Starting testing.')
        raise NotImplementedError