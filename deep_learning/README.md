# Coruscant model

This repo can be used to generate the coruscant model.

To generate the model you should execute the command below

```
pip install -r requirements.txt
python model.py
```

After running the command above you have to copy the file model to the api model folder to serve the model as API inside the folder deep_learning execute the command.

```
cp model/BERT_final.pth ../api/app/model/ 
```