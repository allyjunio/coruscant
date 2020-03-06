import logging, os
from fastai.text import pad_collate
from fastai.vision import Path, pd, gc, partial, accuracy_thresh, Learner

from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from sklearn.model_selection import train_test_split
from fast_ai_BERT import *

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
log = logging.getLogger("CoruscantModel")


class CoruscantModel:
    type_pretrained = None
    data_root = None
    list_files = None
    model_dir = None

    tokenizer_pretrained_coruscant = None
    coruscant_vocab = None
    coruscant_tokenizer = None

    # data bunch
    data_bunch = None
    batch_size = None

    # data to feed the model
    train = None
    test = None
    val = None

    # model
    bert_model_class = None
    loss_func = None
    acc_02 = None
    model = None
    learner = None

    # constants
    label_cols = None
    text_cols = None

    # init constructor
    def __init__(self, type_pretrained='BERT',
                 text_cols="comment_text",
                 list_files=["train.csv", "test.csv"],
                 label_cols=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
                 data_root=Path("..") / "api/app/dataset/jigsaw",
                 model_dir='model',
                 batch_size=12):
        self.data_root = data_root
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.label_cols = label_cols
        self.text_cols = text_cols
        self.list_files = list_files
        self.type_pretrained = type_pretrained
        gc.collect()

        log.debug('type_pretrained: ' + type_pretrained)
        if self.type_pretrained == 'BERT':
            self.tokenizer_pretrained_coruscant = BertTokenizer.from_pretrained("bert-base-uncased")

    def make_model(self):
        log.debug('----- set_train_val_data ------')
        self.set_train_val_data()
        log.debug('----- set_vocab_tokenizer ------')
        self.set_vocab_tokenizer()
        log.debug('----- set_data_bunch ------')
        self.set_data_bunch()
        log.debug('----- create_model ------')
        self.create_model()
        log.debug('----- train_and_save ------')
        self.train_save()

    def set_data_bunch(self):
        self.data_bunch = TextDataBunch.from_df(".", self.train, self.val,
                                                tokenizer=self.coruscant_tokenizer,
                                                vocab=self.coruscant_vocab,
                                                include_bos=False,
                                                include_eos=False,
                                                text_cols=self.text_cols,
                                                label_cols=self.label_cols,
                                                bs=self.batch_size,
                                                collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
                                                )

    def set_train_val_data(self):
        self.train, self.test = [pd.read_csv(self.data_root / fname) for fname in self.list_files]
        self.train, self.val = train_test_split(self.train, shuffle=True, test_size=0.2, random_state=42)
        # log.info(self.train.head())

    def set_vocab_tokenizer(self):
        # In following code snippets, we need to wrap BERT vocab and BERT tokenizer with Fastai modules
        self.coruscant_vocab = Vocab(list(self.tokenizer_pretrained_coruscant.vocab.keys()))
        self.coruscant_tokenizer = Tokenizer(
            tok_func=FastAiBertTokenizer(self.tokenizer_pretrained_coruscant, max_seq_len=256),
            pre_rules=[], post_rules=[])

    def create_model(self):
        # BERT model
        bert_model_class = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
        # Loss function to be used is Binary Cross Entropy with Logistic Losses
        loss_func = nn.BCEWithLogitsLoss()
        # Considering this is a multi-label classification problem, we cant use simple accuracy as metrics here.
        # we will use accuracy_thresh with threshold of 25% as our metric here.
        acc_02 = partial(accuracy_thresh, thresh=0.25)
        self.model = bert_model_class

        # learner function
        self.learner = Learner(self.data_bunch, self.model, loss_func=loss_func, model_dir=self.model_dir,
                               metrics=acc_02)

    def train_save(self):
        x = bert_clas_split(self.model)
        # Let's split the model now in 6 parts
        self.learner.split([x[0], x[1], x[2], x[3], x[5]])
        self.learner.lr_find()
        self.learner.fit_one_cycle(2, max_lr=slice(1e-5, 5e-4), moms=(0.8, 0.7), pct_start=0.2,
                                   wd=(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))

        self.learner.save(self.type_pretrained + '_first')
        self.learner.load(self.type_pretrained + '_first')

        # Now, we will unfreeze last two last layers and train the model again
        self.learner.freeze_to(-2)
        self.learner.fit_one_cycle(2, max_lr=slice(1e-5, 5e-4), moms=(0.8, 0.7), pct_start=0.2,
                                   wd=(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))

        self.learner.save(self.type_pretrained + '_final')
        self.learner.load(self.type_pretrained + '_final')

        # We will now unfreeze the entire model and train it
        self.learner.unfreeze()
        self.learner.lr_find()
        self.learner.fit_one_cycle(2, slice(5e-6, 5e-5), moms=(0.8, 0.7), pct_start=0.2,
                                   wd=(1e-7, 1e-5, 1e-4, 1e-3, 1e-2))

    def test_prediction(self):
        # We will now see our model's prediction power
        text = 'you are so sweet'
        log.info(text)
        log.info(self.learner.predict(text))

        text = 'you are pathetic piece of shit'
        log.info(text)
        log.info(self.learner.predict(text))

        text = "what’s so great about return of the jedi?  the special effects are abysmal,  and the acting is " \
               "horrible. it’s like they phoned it in.  it’s a mess."
        log.info(text)
        log.info(self.learner.predict(text))

        text = "i hate myself for being too human.  how do i liberate my soul ?"
        log.info(text)
        log.info(self.learner.predict(text))

        text = "why was guru arjun singh killed by jahangir?"
        log.info(text)
        log.info(self.learner.predict(text))

        text = "funny how the person that bullies you in elementary is ugly as fuck in high school, and your high " \
               "school bull1, a loser in college..."
        log.info(text)
        log.info(self.learner.predict(text))

        text = "stop making fun of amy winehouse and michael jackso2, #rickcastellano is a bully."
        log.info(text)
        log.info(self.learner.predict(text))


if __name__ == "__main__":
    log.debug('---------------- start model -----------------')
    coruscant_model = CoruscantModel()
    log.debug('---------------- start training -----------------')
    coruscant_model.make_model()
    log.debug('---------------- start prediction -----------------')
    coruscant_model.test_prediction()
