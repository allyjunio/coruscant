from fastai.text import Tokenizer, BaseTokenizer, Vocab, NumericalizeProcessor, TextDataBunch, TextList
from fastai.text import TextLMDataBunch, TokenizeProcessor
from fastai.vision import List, PathOrStr, DataFrame, IntsOrStrs, nn, split_kwargs_by_func, Collection, DataBunch
from fastai.vision import Optional, is_listy, ItemLists
from pytorch_pretrained_bert import BertTokenizer


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


def get_bert_processor(tokenizer: Tokenizer = None, vocab: Vocab = None):
    """
    Constructing preprocessors for BERT
    We remove sos/eos tokens since we add that ourselves in the tokenizer.
    We also use a custom vocabulary to match the numericalization with the original BERT model.
    """
    return [BertTokenizeProcessor(tokenizer=tokenizer),
            NumericalizeProcessor(vocab=vocab)]


# Below code will help us in splitting the model into desirable parts which will be helpful for us in Discriminative
# Learning i.e. setting up different learning rates and weight decays for different parts of the model.
def bert_clas_split(model) -> List[nn.Module]:
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


# Alternatively, we can pass our own list of Preprocessors to the databunch (this is effectively what is happening
# behind the scenes)
class BertTokenizeProcessor(TokenizeProcessor):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)
