import uvicorn
from fastai.text import *
from fastai.vision import *
from fastapi import FastAPI
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from sklearn.model_selection import train_test_split
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.responses import Response
from starlette.staticfiles import StaticFiles

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


DATA_ROOT = Path("app/dataset") / "jigsaw"

train, test = [pd.read_csv(DATA_ROOT / fname) for fname in ["train.csv", "test.csv"]]
train, val = train_test_split(train, shuffle=True, test_size=0.2, random_state=42)

# In following code snippets, we need to wrap BERT vocab and BERT tokenizer with Fastai modules
fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=256), pre_rules=[], post_rules=[])
# Now, we can create our Databunch. Important thing to note here is to use BERT Tokenizer, BERT Vocab. And to and put
# include_bos and include_eos as False as Fastai puts some default values for these
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
label_val = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

databunch_1 = TextDataBunch.from_df(".", train, val,
                                    tokenizer=fastai_tokenizer,
                                    vocab=fastai_bert_vocab,
                                    include_bos=False,
                                    include_eos=False,
                                    text_cols="comment_text",
                                    label_cols=label_cols,
                                    bs=10,
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
    loss_func=loss_func, model_dir='app/model', metrics=acc_02,
)

learner.load('BERT_final')

# We will now unfreeze the entire model and train it
learner.unfreeze()
learner.lr_find()


path = Path(__file__).parent
app = FastAPI()


async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        # you probably want some kind of logging here
        return Response("Internal server error", status_code=500)


app.middleware('http')(catch_exceptions_middleware)
app.mount('/static', StaticFiles(directory='app/static'))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    text_data = await request.json()
    # print(text_data['textSearch'])
    pred_class, pred_idx, percentages = learner.predict(text_data['textSearch'])

    pred_idx = pred_idx.cpu().numpy()
    percentages = percentages.cpu().numpy()
    res = {}
    for idx, lab in enumerate(label_cols):
        if int(pred_idx[idx]) == 1:
            res[lab] = percentages[idx] * 100
        else:
            res[lab] = 0

    return JSONResponse(res)


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
