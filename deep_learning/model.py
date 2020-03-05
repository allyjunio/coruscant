import fastai
import gc
import numpy as np
import pandas as pd
import pretrainedmodels
import sys
import torch
import torch.optim as optim
from fastai import *
from fastai.callbacks.tracker import EarlyStoppingCallback
from fastai.callbacks.tracker import SaveModelCallback
from fastai.text import *
from fastai.vision import *
from pathlib import Path
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification, BertForNextSentencePrediction, \
    BertForMaskedLM
from sklearn.model_selection import train_test_split
from torchvision.models import *
from typing import *

gc.collect()
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


DATA_ROOT = Path("..") / "api/dataset/jigsaw"

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


# Alternatively, we can pass our own list of Preprocessors to the databunch (this is effectively what is happening
# behind the scenes)
class BertTokenizeProcessor(TokenizeProcessor):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)


class BertNumericalizeProcessor(NumericalizeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, vocab=Vocab(list(bert_tok.vocab.keys())), **kwargs)


def get_bert_processor(tokenizer: Tokenizer = None, vocab: Vocab = None):
    """
    Constructing preprocessors for BERT
    We remove sos/eos tokens since we add that ourselves in the tokenizer.
    We also use a custom vocabulary to match the numericalization with the original BERT model.
    """
    return [BertTokenizeProcessor(tokenizer=tokenizer),
            NumericalizeProcessor(vocab=vocab)]


class BertDataBunch(TextDataBunch):
    @classmethod
    def from_df(cls, path: PathOrStr, train_df: DataFrame, valid_df: DataFrame, test_df: Optional[DataFrame] = None,
                tokenizer: Tokenizer = None, vocab: Vocab = None, classes: Collection[str] = None,
                text_cols: IntsOrStrs = 1,
                label_cols: IntsOrStrs = 0, label_delim: str = None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from DataFrames."
        p_kwargs, kwargs = split_kwargs_by_func(kwargs, get_bert_processor)
        # use our custom processors while taking tokenizer and vocab as kwargs
        processor = get_bert_processor(tokenizer=tokenizer, vocab=vocab, **p_kwargs)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        src = src.label_for_lm() if cls == TextLMDataBunch else src.label_from_df(cols=label_cols, classes=classes)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        return src.databunch(**kwargs)


# this will produce a virtually identical databunch to the code above
databunch_2 = BertDataBunch.from_df(".", train_df=train, valid_df=val,
                                    tokenizer=fastai_tokenizer,
                                    vocab=fastai_bert_vocab,
                                    text_cols="comment_text",
                                    label_cols=label_cols,
                                    bs=8,
                                    collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
                                    )

# path = Path('../input/')
#
# print(databunch_2.show_batch())
# print(databunch_1.show_batch())

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


# Below code will help us in splitting the model into desirable parts which will be helpful for us in Discriminative
# Learning i.e. setting up different learning rates and weight decays for different parts of the model.
def bert_clas_split(self) -> List[nn.Module]:
    bert = model.bert
    embedder = bert.embeddings
    pooler = bert.pooler
    encoder = bert.encoder
    classifier = [model.dropout, model.classifier]
    n = len(encoder.layer) // 3
    print(n)
    groups = [[embedder], list(encoder.layer[:n]), list(encoder.layer[n + 1:2 * n]), list(encoder.layer[(2 * n) + 1:]),
              [pooler], classifier]
    return groups


x = bert_clas_split(model)
# Let's split the model now in 6 parts
learner.split([x[0], x[1], x[2], x[3], x[5]])
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(2, max_lr=slice(1e-5, 5e-4), moms=(0.8, 0.7), pct_start=0.2, wd=(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))

learner.save('head')
learner.load('head')

# Now, we will unfreeze last two last layers and train the model again
learner.freeze_to(-2)
learner.fit_one_cycle(2, max_lr=slice(1e-5, 5e-4), moms=(0.8, 0.7), pct_start=0.2, wd=(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))

learner.save('head-2')
learner.load('head-2')

# We will now unfreeze the entire model and train it
learner.unfreeze()
learner.lr_find()
learner.recorder.plot(suggestion=True)

learner.fit_one_cycle(2, slice(5e-6, 5e-5), moms=(0.8, 0.7), pct_start=0.2, wd=(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))

# We will now see our model's prediction power
text = 'you are so sweet'
print(text)
print(learner.predict(text))

text = 'you are pathetic piece of shit'
print(text)
print(learner.predict(text))

text = "what’s so great about return of the jedi?  the special effects are abysmal,  and the acting is horrible.  " \
       "it’s like they phoned it in.  it’s a mess."
print(text)
print(learner.predict(text))

text = "i hate myself for being too human.  how do i liberate my soul ?"
print(text)
print(learner.predict(text))

text = "why was guru arjun singh killed by jahangir?"
print(text)
print(learner.predict(text))

text = "funny how the person that bullies you in elementary is ugly as fuck in high school, and your high school" \
       " bull1, a loser in college..."
print(text)
print(learner.predict(text))

text = "stop making fun of amy winehouse and michael jackso2, #rickcastellano is a bully."
print(text)
print(learner.predict(text))
