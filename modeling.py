import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import AutoTokenizer, AutoModelForQuestionAnswering,AutoConfig, AutoModelForMaskedLM
import os
import torch.nn.functional as F


class BertBaseForQA(nn.Module):
    def __init__(self,args):
        super(BertBaseForQA, self).__init__()
        # Load baseline model
        self.model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.config = AutoConfig.from_pretrained("bert-base-uncased")
    def forward(self,input_ids, attention_mask, start_positions=None, end_positions=None):
        return self.model(input_ids=input_ids,attention_mask=attention_mask,  start_positions=start_positions,end_positions= end_positions,output_hidden_states=True)


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, λ):
        # Store context for backprop
        ctx.λ = λ

        # Forward pass is a no-op
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is dL/dx (since our forward's output was x)

        # Backward pass is just to apply -λ to the gradient
        # This will become the new dL/dx in the rest of the network
        output = - ctx.λ * grad_output

        # Must return number of inputs to forward()
        return output, None

class BertDAForQA(nn.Module):
    def __init__(self,args):
        super(BertDAForQA, self).__init__()
        # Load baseline model finetined on squad
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        self.config = AutoConfig.from_pretrained("bert-base-uncased")
        self.args = args
        print("==========////============:", args.model_args.dann_model_args.spec_layer)

        self.dom_disc = nn.Sequential(
            nn.Linear(self.config.hidden_size, args.num_domains),
            nn.LogSoftmax(dim=1),
        )
        self.dom_disc_middle = nn.Sequential(
            nn.Linear(self.config.hidden_size, args.num_domains),
            nn.LogSoftmax(dim=1),
        )

    def forward(self,input_ids, attention_mask ,start_positions=None, end_positions=None, lambdaa=1):
        qa_output =  self.qa_model(input_ids=input_ids,attention_mask=attention_mask,
                      start_positions=start_positions, end_positions=end_positions,output_hidden_states=True)

        last_layer_hidden_states = qa_output.hidden_states[-1]  # (B,S,D)
        cls_hidden_state = last_layer_hidden_states[:, 0]  # cls token hid_st at 0 (B,D)
        cls_features = cls_hidden_state.view(-1, self.config.hidden_size)
        cls_features_grl = GradientReversalFn.apply(cls_features,lambdaa)
        domain_prob = self.dom_disc(cls_features_grl)


        return qa_output, domain_prob

