import torch
import torchtext
from torchtext import data
from torchtext.vocab import Vectors, GloVe
import os
from tqdm import tqdm
from datetime import datetime
import logging
import pprint
import io

import model

def to_tsv(path):
    """
    Given a path to a txt file where label and text are separated
    by ||| translates to a .tsv file by replacing the separator to a tab.
    Returns the path of the .tsv version of the files.
    Does not overwrite if the files already exists.
    """
    new_path = path.replace('txt', 'tsv')
    if not os.path.exists(new_path):
        with open(path) as data_file, open(new_path, 'w') as tsv_file:
            tsv_file.write(data_file.read().replace('|||', '\t'))
    return new_path


def load_vectors(fname, train_vocab, device):
    """
    Modified from https://fasttext.cc/docs/en/english-vectors.html.
    This loads fasttext vectors for words that have been encountered in the
    vocabulary `train_vocab`.
    We also build a string to inter map to get inter index for the words.
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    stoi = {}
    for idx, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        if tokens[0] in train_vocab.freqs:
            stoi[tokens[0]] = idx
            data[idx] = torch.tensor(list(map(float, tokens[1:])), device=device)
    return data, stoi

def torchtext_iterators(args, load_vectors_manually=True,
    fast_text_path='/home/artidoro/data/crawl-300d-2M.vec'):
    """
    Builds torchtext iterators from the files. Loads vectors manually
    or automatically. For the assignment manually should be set.
    """
    logger = logging.getLogger('logger')
    logger.info('Starting to load data and create iterators.')

    # `sequential` does not tokenize the label.
    label = data.Field(batch_first=True, sequential=False)
    # TODO: try a better tokenizer than the default.
    text = data.Field(batch_first=True, lower=True)

    fields = [('label', label), ('text', text)]
    train = data.TabularDataset(to_tsv(args['train_path']), 'tsv', fields)
    valid = data.TabularDataset(to_tsv(args['valid_path']), 'tsv', fields)
    test = data.TabularDataset(to_tsv(args['test_path']), 'tsv', fields)

    text.build_vocab(train)
    label.build_vocab(train)

    train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
            (train, valid, test), batch_size=args['batch_size'], repeat=False,
            device=torch.device(args['device']), sort_key=lambda x: len(x.text),
            sort_within_batch=False)

    if not args['no_pretrained_vectors']:
        if not load_vectors_manually:
            logger.info('Starting to load vectors from Glove.')
            text.vocab.load_vectors(vectors=GloVe(name='6B'))
        else:
            logger.info('Starting to manually load vectors from FastText.')
            vector_map, stoi = load_vectors(fast_text_path, text.vocab, torch.device(args['device']))
            text.vocab.set_vectors(stoi, vector_map, 300)

    return train_iter, valid_iter, test_iter, text, label

def eval(module, test_iter):
    mode = module.training
    module.eval()

    correct = 0
    total = 0
    loss_tot = 0
    eval_results = {}

    for batch in tqdm(test_iter):
        scores = module.forward(batch.text)
        loss = model.loss(scores, batch.label)
        loss_tot += loss.item()
        preds = scores.argmax(1).squeeze()
        correct += sum((preds == batch.label)).item()
        total += batch.text.shape[0]
    eval_results['loss'] = loss_tot/len(test_iter)
    eval_results['accuracy'] = correct / total

    module.train(mode)
    return eval_results


def calc_accuracy(module, test_iter):
    correct = 0
    total = 0

    for batch in tqdm(test_iter):
        preds = predict(module, batch).squeeze()
        correct += sum((preds == batch.label)).item()
        total += batch.text.shape[0]
    return correct / total


def predict(module, batch):
    mode = module.training
    module.eval()
    scores = module.forward(batch.text)
    module.train(mode)
    return scores.argmax(1)

def train(module, optimizer, train_iter, val_iter, args):
    loss_tot = 0
    logger = logging.getLogger('logger')

    for epoch in range(args['train_epochs']):
        logger.info('Starting training for epoch {} of {}.'.format(epoch+1, args['train_epochs']))
        for batch in tqdm(train_iter):
            optimizer.zero_grad()
            scores = module.forward(batch.text)
            loss = model.loss(scores, batch.label)
            loss.backward()
            optimizer.step()

            # TODO: add regularization back.
            # # Regularize by capping fc layer weights at norm 3
            # if torch.norm(model.fc1.weight.data) > 3.0:
            #     model.fc1.weight = nn.Parameter(3.0 * model.fc1.weight.data / torch.norm(model.fc1.weight.data))

            loss_tot += loss.item()

        loss_avg = loss_tot/len(train_iter)
        logger.info('Loss: {:.4f}'.format(loss_avg))

        if (epoch + 1) % args['eval_epochs'] == 0:
            logger.info('Starting evaluation.')
            evaluation_results = {}
            evaluation_results['train'] = eval(module, train_iter)
            evaluation_results['valid'] = eval(module, val_iter)
            logger.info('\n' + pprint.pformat(evaluation_results))

            # Checkpoint
            checkpoint_path = os.path.join('log', args['checkpoint_path'])
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
                # datetime object containing current date and time
                now = datetime.now()
                dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
                logger.info('Saving Checkpoint: {}'.format(dt_string))

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': loss_avg,
                    'evaluation_results': evaluation_results,
                    'args': args
                    }, os.path.join(checkpoint_path, dt_string + '.pt'))


# def test_model(model, test_iter, filename):
#     "All models should be able to be run with following command."
#     upload = []
#     # Update: for kaggle the bucket iterator needs to have batch_size 10
#     correct = 0
#     total = 0

#     for batch in test_iter:
#         # Your prediction data here (don't cheat!)
#         preds = model.predict(batch)
#         upload += list(preds.data.numpy())

#         correct += sum(preds == batch.label).data.numpy()[0]
#         total += batch.text.size()[0]

#     print("Test Accuracy: " + str(float(correct) / total))

#     with open(filename, "w") as f:
#         f.write("Id,Cat\n")
#         for i, u in enumerate(upload):
#             f.write(str(i) + "," + str(u) + "\n")



