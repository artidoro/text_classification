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
    parser.add_argument('--alpha', default=1)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--embed_size', default=300)
    parser.add_argument('--train_epochs', default=5)
    parser.add_argument('--eval_epochs', default=1)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--device', default='cpu', help='select cuda for the gpu')
    args = vars(parser.parse_args())


    train_iter, val_iter, test_iter, TEXT, LABEL = utils.torchtext_iterators(
        args['train_path'], args['valid_path'], args['test_path'],
        batch_size=args['batch_size'], device=torch.device(args['device']))

    vocab_size = len(TEXT.vocab)
    label_size = len(LABEL.vocab)

    # MNBC = model.MNBC(vocab_size, args['alpha'])
    # MNBC.train(train_iter, val_iter, test_iter)

    #LogReg = model.LogReg(vocab_size)
    #LogReg.train(train_iter, val_iter, train_epoch=2)

    #CBOW = model.CBOW(vocab_size, TEXT.vocab.vectors, vect_size)
    #CBOW.train(train_iter, val_iter, train_epoch=1)

    CNN = model.CNN(TEXT.vocab.vectors, label_size, args['dropout'])
    CNN.to(torch.device(args['device']))
    utils.train(CNN, train_iter, val_iter, train_epoch=args['train_epochs'], eval_epochs=args['eval_epochs'])

    #CNN2 = model.CNN2(embeddings=TEXT.vocab.vectors)
    #CNN2.train(train_iter, val_iter, test_iter, train_epoch=200)

    # LSTM = model.LSTM(embeddings=TEXT.vocab.vectors, layers=1)
    # LSTM.train(train_iter, val_iter, test_iter, train_epoch=5)

    # LSTMCNN = model.LSTMCNN(embeddings=TEXT.vocab.vectors)
    # LSTMCNN.train(train_iter, val_iter, test_iter, train_epoch=5)