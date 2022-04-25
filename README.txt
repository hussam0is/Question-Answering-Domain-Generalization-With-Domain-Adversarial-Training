In order to reproduce the experiment you should run main.py with the model name

To run our model BertQA with domain-adversarial :

python main.py --model_name bert_da

To run BertQA finetuned on all source domains :

python main.py --model_name bert_all

To run BertQA finetuned on squad :

python main.py --model_name bert_squad


because of the nature of the task (question answering with long paragraph input on bert-base) it will take a lot of time to run the evaluation experiment:
~15m per out-of-domain light dataset (BioASQ, DROP, RelationExtraction)
~1h per out of domain heavy (long context - see paper) dataset (DuoRC,RACE,Textbook)
if you want to check only for the light datasets just remove the rest from the dev file. 