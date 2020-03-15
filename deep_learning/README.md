# Coruscant model

This repo contains the [BERT Machine Learning model](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270) used to generate the Offensiveness Scores identified by Coruscant.

## How To Use The Model

To generate the model you should execute the command below

```
pip install -r requirements.txt
python model.py
```

After running the command above you have to copy the file model to the api model folder to serve the model as API inside the folder deep_learning execute the command.

```
cp model/BERT_final.pth ../api/app/model/ 
```
