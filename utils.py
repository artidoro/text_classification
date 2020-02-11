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

def get_average_embedding(vector_map):
    """
    From the dictionary of embeddings gets the average out.
    """
    embeds = torch.cat(list(map(lambda x: x.view(1, -1), vector_map.values())), 0)
    return torch.mean(embeds, 0)


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

    text.build_vocab(train, min_freq=args['min_freq'])
    label.build_vocab(train)

    train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
            (train, valid, test), batch_size=args['batch_size'], repeat=False,
            device=torch.device(args['device']), sort=False,
            sort_within_batch=False)

    if not args['no_pretrained_vectors']:
        if not load_vectors_manually:
            logger.info('Starting to load vectors from Glove.')
            text.vocab.load_vectors(vectors=GloVe(name='6B'))
        else:
            logger.info('Starting to manually load vectors from FastText.')
            vector_map, stoi = load_vectors(fast_text_path, text.vocab, torch.device(args['device']))
            average_embed = get_average_embedding(vector_map)
            text.vocab.set_vectors(stoi, vector_map, 300, unk_init=lambda x: average_embed.clone())
            text.vocab.vectors[text.vocab.stoi['<unk>']] = average_embed.clone()

    logger.info('Built train vocabulary of {} words'.format(len(text.vocab)))
    return train_iter, valid_iter, test_iter, text, label

def eval(module, test_iter, args, write_to_file=False):
    mode = module.training
    module.eval()

    correct = 0
    total = 0
    loss_tot = 0
    eval_results = {}
    predictions = []

    for batch in tqdm(test_iter):
        scores = module.forward(batch.text)
        loss = model.loss(scores, batch.label)
        loss_tot += loss.item()
        preds = scores.argmax(1).squeeze()
        correct += sum((preds == batch.label)).item()
        total += batch.text.shape[0]

        if write_to_file:
            predictions += list(preds.cpu().numpy())

    eval_results['loss'] = loss_tot/len(test_iter)
    eval_results['accuracy'] = correct / total

    # Write predictions to file.
    if write_to_file:
        write_predictions(predictions, args, eval_results)

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

def train(module, optimizer, scheduler, train_iter, val_iter, args):
    logger = logging.getLogger('logger')

    for epoch in range(args['train_epochs']):
        logger.info('Starting training for epoch {} of {}.'.format(epoch+1, args['train_epochs']))
        loss_tot = 0
        for batch in tqdm(train_iter):
            optimizer.zero_grad()
            scores = module.forward(batch.text)
            loss = model.loss(scores, batch.label)
            loss.backward()
            optimizer.step()
            loss_tot += loss.item()

            # Regularize by capping fc layer weights at norm 3
            if torch.norm(module.fc1.weight.data) > 3.0:
                module.fc1.weight = torch.nn.Parameter(3.0 * module.fc1.weight.data / torch.norm(module.fc1.weight.data))

        loss_avg = loss_tot/len(train_iter)
        logger.info('Loss: {:.4f}'.format(loss_avg))

        if (epoch + 1) % args['eval_epochs'] == 0:
            logger.info('Starting evaluation.')
            evaluation_results = {}
            evaluation_results['train'] = eval(module, train_iter, args)
            evaluation_results['valid'] = eval(module, val_iter, args)
            logger.info('\n' + pprint.pformat(evaluation_results))

            # Update the scheduler.
            scheduler.step(evaluation_results['valid']['loss'])

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
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': loss_avg,
                'evaluation_results': evaluation_results,
                'args': args
                }, os.path.join(checkpoint_path, dt_string + '.pt'))


def compute_acc_out_files(refs_path, outs_path):
    refs = open(refs_path).readlines()
    outs = open(outs_path).readlines()

    assert len(refs) == len(outs), 'The number of lines should be equal to 643!' 

    cor = 0
    for r, o in zip(refs, outs):
        if r.lower().rstrip() == o.lower().rstrip():
            cor += 1
    logger = logging.getLogger('logger')
    logger.info('The accuracy number on the dev set is %.2f %%.' % (cor/6.43))
    return cor/643


def write_predictions(predictions, args, eval_results=None):
    logger = logging.getLogger('logger')
    checkpoint_path = os.path.join('log', args['checkpoint_path'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if args['mode'] == train:
        data_path = args['train_path']
        results_path = os.path.join(checkpoint_path, 'train_results.txt')
    elif args['mode'] == 'eval':
        data_path = args['valid_path']
        results_path = os.path.join(checkpoint_path, 'dev_results.txt')
    else:
        data_path = args['test_path']
        results_path = os.path.join(checkpoint_path, 'test_results.txt')

    logger.info('Writing predictions to file {}'.format(results_path))
    with open(results_path, 'w') as res_file, open(data_path) as data_file:
        data_lines = data_file.readlines()
        assert len(data_lines) == len(predictions), 'Predictions and data lines do not have the same length'
        for idx, line in enumerate(data_lines):
            line_split = line.split('|||')
            line_split[0] = str(args['LABEL'].vocab.itos[predictions[idx]])
            line_res = '|||'.join(line_split)
            res_file.write(line_res)

    # Sanity check.
    if args['mode'] == 'eval':
        acc_script = compute_acc_out_files(args['valid_path'], results_path)
        assert acc_script == eval_results['accuracy']

def predict_write_to_file(module, test_iter, args):
    mode = module.training
    module.eval()
    predictions = []

    for batch in tqdm(test_iter):
        scores = module.forward(batch.text)
        preds = scores.argmax(1).squeeze()
        predictions += list(preds.cpu().numpy())

    # Write predictions to file.
    write_predictions(predictions, args)
    module.train(mode)


# def ensemble(args, paths):
#     logger = logging.getLogger('logger')

#     for path in paths:
#         checkpoint_path = os.path.join('log', path)
#         assert os.path.exists(checkpoint_path), 'checkpoint path does not exists'


#     checkpoint_path = os.path.join('log', args['checkpoint_path'])
#     if not os.path.exists(checkpoint_path):
#         os.makedirs(checkpoint_path)
