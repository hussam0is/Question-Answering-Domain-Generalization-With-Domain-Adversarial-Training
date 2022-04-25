import os

import yaml
from box import Box

from clearml import Task, Logger
from pprint import pprint
import torch
from pytorch_pretrained_bert.optimization import BertAdam
from modeling import BertBaseForQA, BertDAForQA
import argparse
from utils import *
from train import BaseTrainer, BertDATrainer
from dataloading import get_tokenized_datasets, get_dl_train, get_dl_eval

PROJECT_NAME = "QA_PROJECT"
TRAIN = 'train'
DEV = 'dev'
TEST = 'test'

def main(training_args,model_name):
    # Setting up logging

    task = Task.init(project_name=PROJECT_NAME, task_name=training_args.experiment_name)
    task.connect(training_args)
    logger = task.get_logger()

    pprint(training_args)

    # Check args and unpack them
    check_args(training_args)
    data_args, model_args = training_args.data_args, training_args.model_args

    # Set seed for reproducibility
    set_seed(training_args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    epoch = 0
    """
    if training_args.experiment_name == 'table 1':
        PATH = os.path.join(training_args.model_checkpoint_folder, "squad")
        model= torch.load(PATH)
        optimizer = BertAdam(model.parameters(), lr=training_args.learning_rate)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device, n_epochs=training_args.num_epochs,
                                do_eval=True, eval_folder=data_args.in_dom_dev_folder,
                              checkpoint_folder="./models", args=training_args)
        epoch = 0
        trainer.evaluate_model(epoch)

        PATH = os.path.join(training_args.model_checkpoint_folder, "news")
        model = torch.load(PATH)
        optimizer = BertAdam(model.parameters(), lr=training_args.learning_rate)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device, n_epochs=training_args.num_epochs,
                              do_eval=True, eval_folder=data_args.in_dom_dev_folder,
                              checkpoint_folder="./models", args=training_args)
        trainer.evaluate_model(20)

        PATH = os.path.join(training_args.model_checkpoint_folder, "natural")
        model = torch.load(PATH)
        optimizer = BertAdam(model.parameters(), lr=training_args.learning_rate)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device, n_epochs=training_args.num_epochs,
                              do_eval=True, eval_folder=data_args.in_dom_dev_folder,
                              checkpoint_folder="./models", args=training_args)
        trainer.evaluate_model(30)

        PATH = os.path.join(training_args.model_checkpoint_folder, "hotpot")
        model = torch.load(PATH)
        optimizer = BertAdam(model.parameters(), lr=training_args.learning_rate)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device, n_epochs=training_args.num_epochs,
                              do_eval=True, eval_folder=data_args.in_dom_dev_folder,
                              checkpoint_folder="./models", args=training_args)
        trainer.evaluate_model(40)
    """
    if model_name == 'bert_squad':
        PATH = os.path.join(training_args.model_checkpoint_folder, "bert-squad")
        model = torch.load(PATH)
        optimizer = BertAdam(model.parameters(), lr=training_args.learning_rate)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device, n_epochs=training_args.num_epochs,
                              do_eval=True, eval_folder=data_args.dev_folder,
                              checkpoint_folder=training_args.model_checkpoint_folder, args=training_args)
    elif model_name == 'bert_all':
        PATH = os.path.join(training_args.model_checkpoint_folder, "bert-all")
        model = torch.load(PATH)
        optimizer = BertAdam(model.parameters(), lr=training_args.learning_rate)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device, n_epochs=training_args.num_epochs,
                              do_eval=True, eval_folder=data_args.dev_folder,
                              checkpoint_folder=training_args.model_checkpoint_folder, args=training_args)
    elif model_name == 'bert_da':
        PATH = os.path.join(training_args.model_checkpoint_folder, "bert-da")
        model = torch.load(PATH)
        optimizer = BertAdam(model.parameters(), lr=training_args.learning_rate)
        trainer = BertDATrainer(model=model, optimizer=optimizer, device=device, n_epochs=training_args.num_epochs,
                              do_eval=True, eval_folder=data_args.dev_folder,
                              checkpoint_folder=training_args.model_checkpoint_folder, args=training_args)

    trainer.evaluate_model(epoch)
    print("=======/ Done /=====")
    task.close()

    """script for training from scratch:
    print("Initializing BertDAForQA model")
    model = BertDAForQA(training_args)
    model.to(device)
    optimizer = BertAdam(model.parameters(), lr=training_args.learning_rate)
    print("Loading datasets")
    train_features_lst = get_tokenized_datasets(model, data_args)
    print("Training")

    dl_train=get_dl_train(train_features_lst, training_args)

    trainer = BertDATrainer(model=model,optimizer=optimizer,device=device,n_epochs=training_args.num_epochs,dl_train=dl_train,do_eval=True,eval_folder=data_args.dev_folder,
                          checkpoint_folder=training_args.model_checkpoint_folder,args=training_args)
    trainer.train()
    """

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproducing results')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='Path to YAML config file. Defualt: config.yaml')
    parser.add_argument('--model_name', default='bert_squad', type=str,help='run the experiment with the supplied model name')
    args = parser.parse_args()

    with open(args.config) as f:
        training_args = Box(yaml.load(f, Loader=yaml.FullLoader))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(training_args,args.model_name)