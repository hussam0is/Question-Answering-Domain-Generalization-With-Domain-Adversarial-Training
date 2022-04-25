
import numpy as np
from box import Box
from tqdm import tqdm
from evaluation import run_eval
import os
import abc
import torch
from clearml import Task, Logger
import tqdm.auto
from typing import Optional
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from modeling import BertDAForQA, BertBaseForQA


class BaseTrainer(abc.ABC):
    """
    Trainer for base models.
    """
    def __init__(
        self,
        model: BertBaseForQA,
        optimizer: Optimizer = None,
        device: Optional[torch.device] = None,
        n_epochs: int = 0,
        dl_train: DataLoader = None,
        do_eval: bool = True,
        eval_folder: str = None,
        checkpoint_folder: str = None,
        args: Box = None
    ):
        """
        Initialize the trainer.
        :param model: Instance of the bert-like model to train.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        super().__init__()
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.dl_train = dl_train
        self.do_eval = do_eval
        self.eval_folder = eval_folder
        self.checkpoint_folder = checkpoint_folder
        self.model_name = "bert-squad"
        self.is_adv = False
        self.args = args
    def save_checkpoint(self, checkpoint_filename: str, epoch_idx: int):
        """
        Saves the model in it's current state to a file with the given name (treated
        as a relative path).
        :param checkpoint_filename: File name or relative path to save to.
        """
        torch.save(self.model, checkpoint_filename)
        print(f"\n*** epoch = {epoch_idx} Saved checkpoint {checkpoint_filename}")
        if self.do_eval:
            self.evaluate_model(epoch_idx)

    def train(self):
        for epoch_idx in range(self.n_epochs):
            self.model.train()
            print(f'Epoch {epoch_idx + 1:04d} / {self.n_epochs:04d}', end='\n=================\n')

            for batch_idx, (input_ids, input_mask, seq_ids, start_positions, end_positions, _) in enumerate(tqdm.tqdm(self.dl_train, desc=f"Train Epoch {epoch_idx}")):
                if self.device:
                    input_ids, input_mask, seq_ids, start_positions, end_positions = input_ids.to(
                        self.device), input_mask.to(self.device), seq_ids.to(self.device), start_positions.to(
                        self.device), end_positions.to(self.device)
                output = self.model.forward(input_ids=input_ids, attention_mask=input_mask, start_positions=start_positions,
                                    end_positions=end_positions)
                # Compute loss
                batch_loss = output.loss.mean()

                # Backward pass
                self.optimizer.zero_grad()  # Zero gradients of all parameters
                batch_loss.backward()  # Run backprop algorithms to calculate gradients
                # Optimization step
                self.optimizer.step()
                Logger.current_logger().report_scalar(title="training_batch_loss", series=Task.current_task().name,
                                     value=batch_loss, iteration=batch_idx)
            # ==== After each  epoch save checkpoint and evaluate
            model_name='model_{}_epoch_{}'.format(self.model_name, epoch_idx+1)
            checkpoint_filename: str = os.path.join(self.checkpoint_folder, model_name)
            self.save_checkpoint(checkpoint_filename, (epoch_idx+1))
    # ========================


    def evaluate_model(self, epoch_idx):
        self.model.eval()
        files = [f for f in os.listdir(self.eval_folder) if f.endswith(".gz")]
        for dev_file in files:
            file_name = dev_file.split(".")[0]
            prediction_file = os.path.join("./results", "model_{}_epoch_{}_{}.json".format(self.model_name, epoch_idx, file_name))
            file_path = os.path.join(self.eval_folder, dev_file)
            metrics = run_eval(self.model, self.model.tokenizer, file_path, prediction_file, self.is_adv, args=self.args)
            em, f1 = round(metrics["exact_match"], 2), round(metrics["f1"], 2)
            print(f"Evaluation score for {file_name} is Exct Match = {em},F1 Score = {f1}")
            Logger.current_logger().report_scalar(title=f"Evaluation {file_name} at epoch {epoch_idx}", series=Task.current_task().name,
                                                  value=f1, iteration=epoch_idx)


class BertDATrainer(BaseTrainer):
    def __init__(
        self,
        model: BertDAForQA,
        optimizer: Optimizer,
        device: Optional[torch.device] = None,
        n_epochs: int = None,
        dl_train: DataLoader = None,
        do_eval: bool = True,
        eval_folder: str = None,
        checkpoint_folder: str = None,
        args: Box = None
    ):
        super(BertDATrainer, self).__init__(model, optimizer, device,n_epochs,dl_train,do_eval,eval_folder,checkpoint_folder,args)

        self.loss_fn_domain = torch.nn.NLLLoss()
        self.is_adv = True
        self.model_name = "bertDA"
    def train(self):
        max_batches = len(self.dl_train)
        for epoch_idx in range(self.n_epochs):
            self.model.train()
            print(f'Epoch {epoch_idx + 1:04d} / {self.n_epochs:04d}', end='\n=================\n')

            for batch_idx, (input_ids, input_mask, seq_ids, start_positions, end_positions, labels) in enumerate(tqdm.tqdm(self.dl_train, desc=f"Train Epoch {epoch_idx+1}", total=max_batches)):
                if batch_idx==0:
                    print(input_ids.shape,self.model.config.hidden_size)
                # Calculate training progress and GRL λ
                p = float(batch_idx + epoch_idx * max_batches) / (self.n_epochs * max_batches)
                lambdaa = (2. / (1. + np.exp(-10 * p)) - 1) / 10

                y_domain = labels  #  source domain labels
                qa_output, domain_prob = self.model.forward(input_ids=input_ids,attention_mask= input_mask, start_positions = start_positions,end_positions=end_positions,
                                                            lambdaa=lambdaa)
                loss_qa_domain = qa_output.loss.mean()  # qa_loss loss
                loss_disc_domain = self.loss_fn_domain(domain_prob, y_domain)
                loss = (loss_qa_domain + loss_disc_domain)

                # === Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                Logger.current_logger().report_scalar(title=f"qa_loss_epoch{epoch_idx+1}", series=Task.current_task().name,
                                                      value=loss_qa_domain, iteration=batch_idx)
                Logger.current_logger().report_scalar(title=f"domain_loss_epoch{epoch_idx+1}", series=Task.current_task().name,
                                                      value=loss_disc_domain,
                                                      iteration=batch_idx)

            # ==== After each  epoch save checkpoint and evaluate
            model_name = 'model_{}_epoch_{}'.format(self.model_name, epoch_idx+1)
            checkpoint_filename: str = os.path.join(self.checkpoint_folder, model_name)
            self.save_checkpoint(checkpoint_filename, epoch_idx+1)
    def save_checkpoint(self, checkpoint_filename: str, epoch_idx: int):
        """
        Saves the model in it's current state to a file with the given name (treated
        as a relative path).
        :param checkpoint_filename: File name or relative path to save to.
        """
        torch.save(self.model, checkpoint_filename)
        print(f"\n*** epoch = {epoch_idx} Saved checkpoint {checkpoint_filename}")
        if self.do_eval:
            self.evaluate_model(epoch_idx)

