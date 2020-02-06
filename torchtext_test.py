# %%
import io
import torch
import torchtext
import os
from tqdm import tqdm
from torchtext import data

#%%

fast_text_path = '/home/artidoro/data/crawl-300d-2M.vec'


def load_vectors(fname, train_vocab):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    stoi = {}
    for idx, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        if tokens[0] in train_vocab.freqs:
            stoi[tokens[0]] = idx
            data[idx] = torch.tensor(list(map(float, tokens[1:])))
    return data, stoi

#%%

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
    batch_size=10, device=torch.device('cpu'), load_vectors_manually=True):
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

    if not load_vectors_manually:
        text.vocab.load_vectors(vectors=GloVe(name='6B'))

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



# %%
train_iter, val_iter, test_iter, TEXT, LABEL = torchtext_iterators(
        '/home/artidoro/cmu/courses/11-747 Neural Networks for NLP/Homework1/topicclass/topicclass_train.txt', 
        '/home/artidoro/cmu/courses/11-747 Neural Networks for NLP/Homework1/topicclass/topicclass_valid.txt',
        '/home/artidoro/cmu/courses/11-747 Neural Networks for NLP/Homework1/topicclass/topicclass_test.txt',
        batch_size=10, device=torch.device('cuda'))

#%%
vector_map, stoi = load_vectors(fast_text_path, TEXT.vocab)

#%%



# %%

# %%
TEXT.vocab.set_vectors(stoi, vector_map, 300)

# %%
