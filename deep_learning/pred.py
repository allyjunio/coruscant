import numpy as np
import pandas as pd

from pathlib import Path
from typing import *

import torch
import torch.optim as optim

import gc
gc.collect()

import fastai

from fastai import *
from fastai.vision import *
from fastai.text import *

from sklearn.model_selection import train_test_split

from torchvision.models import *
import pretrainedmodels

import sys

from fastai.callbacks.tracker import EarlyStoppingCallback
from fastai.callbacks.tracker import SaveModelCallback

from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification, BertForNextSentencePrediction, \
    BertForMaskedLM
from pytorch_pretrained_bert import BertTokenizer

bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")

class FastAiBertTokenizer(BaseTokenizer):
    """Wrapper around BertTokenizer to be compatible with fast.ai"""

    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int = 128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t: str) -> List[str]:
        """Limits the maximum sequence length"""
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]

DATA_ROOT = Path("..") / "dataset/jigsaw"

train, test = [pd.read_csv(DATA_ROOT / fname) for fname in ["train.csv", "test.csv"]]
train, val = train_test_split(train, shuffle=True, test_size=0.2, random_state=42)

print(train.head())

# In following code snippets, we need to wrap BERT vocab and BERT tokenizer with Fastai modules
fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=256), pre_rules=[], post_rules=[])
# Now, we can create our Databunch. Important thing to note here is to use BERT Tokenizer, BERT Vocab. And to and put
# include_bos and include_eos as False as Fastai puts some default values for these
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

databunch_1 = TextDataBunch.from_df(".", train, val,
                                    tokenizer=fastai_tokenizer,
                                    vocab=fastai_bert_vocab,
                                    include_bos=False,
                                    include_eos=False,
                                    text_cols="comment_text",
                                    label_cols=label_cols,
                                    bs=8,
                                    collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
                                    )

# BERT model
bert_model_class = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
# Loss function to be used is Binary Cross Entropy with Logistic Losses
loss_func = nn.BCEWithLogitsLoss()
# Considering this is a multi-label classification problem, we cant use simple accuracy as metrics here. Instead,
# we will use accuracy_thresh with threshold of 25% as our metric here.
acc_02 = partial(accuracy_thresh, thresh=0.25)
model = bert_model_class

# learner function
learner = Learner(
    databunch_1, model,
    loss_func=loss_func, model_dir='model', metrics=acc_02,
)

learner.load('head-2')

# We will now unfreeze the entire model and train it
learner.unfreeze()
learner.lr_find()
learner.recorder.plot(suggestion=True)

# Single prediction
text = "@J2ocean all you do is bully me :("
print(text)
res = learner.predict(text)
print(res)

text = "@Rissuh123 awe! hahah how cute. i hate the mostttt when kids bully him, i seriously want to strangle them all. -___-"
print(text)
print(learner.predict(text))

text = "just get me result for this text"
print(text)
print(learner.predict(text))

text = 'you are so sweet'
print(text)
print(learner.predict(text))

text = 'you are pathetic piece of shit'
print(text)
print(learner.predict(text))

text = "what’s so great about return of the jedi?  the special effects are abysmal,  and the acting is horrible.  it’s like they phoned it in.  it’s a mess."
print(text)
print(learner.predict(text))

text = "i hate myself for being too human.  how do i liberate my soul ?"
print(text)
print(learner.predict(text))

text = "why was guru arjun singh killed by jahangir?"
print(text)
print(learner.predict(text))

text = "funny how the person that bullies you in elementary is ugly as fuck in high school, and your high school bull1, a loser in college..."
print(text)
print(learner.predict(text))

text = "stop making fun of amy winehouse and michael jackso2, #rickcastellano is a bully."
print(text)
print(learner.predict(text))