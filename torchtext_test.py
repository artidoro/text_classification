# #%%
# from torchtext import data

# #%%
# text = data.Field(batch_first=True)
# label = data.Field(sequential=False)

fields = {'name': ('n', NAME), 'location': ('p', PLACE), 'quote': ('s', SAYING)}

#%%
%reload_ext autoreload
%autoreload 2

#%%
import torch
import torchtext
from torchtext import data

import utils

# %%
utils.to_tsv('../topicclass/topicclass_train.txt')
utils.to_tsv('../topicclass/topicclass_valid.txt')
utils.to_tsv('../topicclass/topicclass_test.txt')



#%%
# `sequential` does not tokenize the label.
label = data.Field(batch_first=True, sequential=False)
# TODO: try a better tokenizer than the default.
text = data.Field(batch_first=True, lower=True)


#%%

train = data.TabularDataset('../topicclass/topicclass_train.tsv', 'tsv', [('label', label), ('text', text)])
valid = data.TabularDataset('../topicclass/topicclass_valid.tsv', 'tsv', [('label', label), ('text', text)])
test = data.TabularDataset('../topicclass/topicclass_test.tsv', 'tsv', [('label', label), ('text', text)])


#%%
text.build_vocab(train)
label.build_vocab(train)

#%%
train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, valid, test), batch_size=10, repeat=False, device=torch.device('cpu'), sort_key=lambda x: len(x.text), sort_within_batch=False)

#%%
for t in valid_iter:
    print(t.label[0])

#%%













#%%
train_data = utils.parse_file('../topicclass/topicclass_train.txt')
valid_data = utils.parse_file('../topicclass/topicclass_valid.txt')
test_data = utils.parse_file('../topicclass/topicclass_test.txt')


#%%



# %%
# fields = {'label': ('l', LABEL), 'text': ('t', TEXT)}


label = data.Field(sequential=False)
text = torchtext.data.Field(batch_first=True)

train_examples = data.Example.fromlist(train_data, [('label', label), ('text', text)])
train = data.Dataset(train_examples, [('label', label), ('text', text)])

valid_examples = data.Example.fromlist(valid_data, [('label', label), ('text', text)])
valid = data.Dataset(valid_examples, [('label', label), ('text', text)])

test_examples = data.Example.fromlist(test_data, [('label', label), ('text', text)])
test = data.Dataset(test_examples, [('label', label), ('text', text)])


# %%
text.build_vocab(train)
label.build_vocab(train)


# %%
