import torch
import torchtext
from torchtext import data
from torchtext.vocab import Vectors, GloVe
import os
from tqdm import tqdm
from datetime import datetime

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

def torchtext_iterators(train_path, valid_path, test_path,
    batch_size=10, device=torch.device('cpu')):
    """
    Builds torchtext iterators from the files.
    """

    # `sequential` does not tokenize the label.
    label = data.Field(batch_first=True, sequential=False)
    # TODO: try a better tokenizer than the default.
    text = data.Field(batch_first=True, lower=True)

    fields = [('label', label), ('text', text)]
    train = data.TabularDataset(to_tsv(train_path), 'tsv', fields)
    valid = data.TabularDataset(to_tsv(valid_path), 'tsv', fields)
    test = data.TabularDataset(to_tsv(test_path), 'tsv', fields)

    text.build_vocab(train)
    label.build_vocab(train)

    train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
            (train, valid, test), batch_size=batch_size, repeat=False, device=device,
            sort_key=lambda x: len(x.text), sort_within_batch=False)

    # TODO: Use FastText embeddings.
    #url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    #TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

    text.vocab.load_vectors(vectors=GloVe(name='6B'))

    return train_iter, valid_iter, test_iter, text, label

def predict(module, batch):
    mode = module.training
    module.eval()
    scores = module.forward(batch.text)
    module.train(mode)
    return scores.argmax(1)

def train(train_iter, val_iter, TEXT, LABEL, args):

    # Initialize model and optimizer. This requires loading checkpoint if specified in the arguments.
    if args['load_checkpoint_path'] == None:
        module = model.model_dict[args['model_name']](TEXT.vocab.vectors, len(LABEL.vocab), args)
        optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, module.parameters()))
        epoch_start = 0
    else:
        checkpoint = torch.load(args['load_checkpoint_path'])

        # Overwrite arguments if required.
        if args['overwrite_args']:
            args = checkpoint['args']
        module = model.model_dict[checkpoint['args']['model_name']]()
        optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, module.parameters()))

        module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']

    module.to(torch.device(args['device']))
    loss_function = torch.nn.CrossEntropyLoss()

    loss_tot = 0

    for epoch in tqdm(range(epoch_start, args['train_epochs'])):
        for batch in tqdm(train_iter):
            optimizer.zero_grad()
            scores = module.forward(batch.text)
            loss = loss_function(scores, batch.label)
            loss.backward()
            optimizer.step()

            # TODO: add regularization back.
            # # Regularize by capping fc layer weights at norm 3
            # if torch.norm(model.fc1.weight.data) > 3.0:
            #     model.fc1.weight = nn.Parameter(3.0 * model.fc1.weight.data / torch.norm(model.fc1.weight.data))

            loss_tot += loss.item()

        loss_avg = loss_tot/len(train_iter)
        print ("Loss: " + str(loss_avg))

        if (epoch + 1) % args['eval_epochs'] == 0:
            val_accuracy = calc_accuracy(module, val_iter)
            train_accuracy = calc_accuracy(module, train_iter)
            print ("Epoch: " + str(epoch))
            print ("Train Accuracy: " + str(train_accuracy))
            print ("Validation Accuracy: " + str(val_accuracy))

            # Checkpoint
            checkpoint_path = os.path.join('log', args['checkpoint_path'])
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            with open(os.path.join(checkpoint_path, 'log.txt'), 'a+') as log_file:
                log_file.write("Epoch: " + str(epoch) + "\n")
                log_file.write("Train Accuracy: " + str(train_accuracy) + "\n")
                log_file.write("Validation Accuracy: " + str(val_accuracy) + "\n")
                log_file.write("Loss: " + str(loss_avg) + "\n")

                # datetime object containing current date and time
                now = datetime.now()
                dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
                log_file.write("Saving Checkpoint: " + str(dt_string) + "\n")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': loss_avg,
                    'val_accuracy': val_accuracy,
                    'train_accuracy': train_accuracy,
                    'args': args
                    }, os.path.join(checkpoint_path, dt_string + '.pt'))

def calc_accuracy(module, test_iter):
    correct = 0
    total = 0

    for batch in tqdm(test_iter):
        preds = predict(module, batch).squeeze()
        correct += sum((preds == batch.label)).item()
        total += batch.text.shape[0]
    return correct / total

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



