import os
import pickle
from iterator import read_squad_examples, convert_examples_to_features
import torch
import numpy as np
from torch.utils.data import DataLoader,TensorDataset,RandomSampler,SequentialSampler


#Load dataset and tokenize it using the tokenizer - then save as pckl file
def get_tokenized_datasets(model,data_args):
    train_features_lst, dev_features_lst = [], []
    # datasets to finetune on:
    files = [f for f in os.listdir(data_args.train_folder) if f.endswith(".gz")]
    print("Number of train datasets:{}".format(len(files)))
    for filename in files:
        data_name = filename.split(".")[0]
        # Check whether pkl file already exists
        pickle_file_name = '{}.pkl'.format(data_name)
        pickle_file_path = os.path.join(data_args.dataset_checkpoint, pickle_file_name)
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as pkl_f:
                print("Loading {} file as pkl...".format(data_name))
                train_features_lst.append(pickle.load(pkl_f))
        else:
            print("processing {} file".format(data_name))
            file_path = os.path.join(data_args.train_folder, filename)

            train_examples = read_squad_examples(file_path, debug=False)

            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=model.tokenizer,
                max_seq_length=data_args.max_seq_length,
                max_query_length=data_args.max_ques_length,
                doc_stride=data_args.doc_stride,
                is_training=True
            )

            train_features_lst.append(train_features)

            # Save feature lst as pickle (For reuse & fast loading)
            with open(pickle_file_path, 'wb') as pkl_f:
                print("Saving {} file from pkl file...".format(data_name))
                pickle.dump(train_features, pkl_f)

    return train_features_lst

#combine all the target datasets into one dataloader
def get_dl_eval(features_lst, args):
    all_input_ids,all_input_mask,all_segment_ids, all_labels= [],[],[],[]
    print("len featlst,",len(features_lst))
    for i, train_features in enumerate(features_lst):
        eval_portion = 1000
        train_features = train_features[-eval_portion:]
        all_input_ids.append(torch.tensor(np.array([f.input_ids for f in train_features]), dtype=torch.long))
        all_input_mask.append(torch.tensor(np.array([f.input_mask for f in train_features]), dtype=torch.long))
        all_segment_ids.append(torch.tensor(np.array([f.segment_ids for f in train_features]), dtype=torch.long))
        all_labels.append(i * torch.ones_like(all_input_ids[-1]))

    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_input_mask = torch.cat(all_input_mask, dim=0)
    all_segment_ids = torch.cat(all_segment_ids, dim=0)
    all_labels = torch.cat(all_labels,dim=0)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_labels)


    sampler = SequentialSampler(eval_data)

    data_loader = DataLoader(eval_data, batch_size=args.data_args.batch_size, shuffle=None,
                                              sampler=sampler)


    return data_loader

#combine all the target datasets into one dataloader
def get_dl_train(train_features_lst, args):
    all_input_ids,all_input_mask, all_segment_ids = [],[],[]
    all_start_positions, all_end_positions, all_labels = [],[],[]
    for i, train_features in enumerate(train_features_lst):
        train_portion = 25000
        train_features = train_features[:train_portion]
        all_input_ids.append(torch.tensor(np.array([f.input_ids for f in train_features]), dtype=torch.long))
        all_input_mask.append(torch.tensor(np.array([f.input_mask for f in train_features]), dtype=torch.long))
        all_segment_ids.append(torch.tensor(np.array([f.segment_ids for f in train_features]), dtype=torch.long))
        all_start_positions.append(torch.tensor(np.array([f.start_position for f in train_features]), dtype=torch.long))
        all_end_positions.append(torch.tensor(np.array([f.end_position for f in train_features]), dtype=torch.long))
        all_labels.append(i * torch.ones_like(all_start_positions[-1]))
    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_input_mask = torch.cat(all_input_mask, dim=0)
    all_segment_ids = torch.cat(all_segment_ids, dim=0)
    all_start_positions = torch.cat(all_start_positions, dim=0)
    all_end_positions = torch.cat(all_end_positions, dim=0)
    all_labels = torch.cat(all_labels,dim=0)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_start_positions, all_end_positions,all_labels)

    sampler = RandomSampler(train_data)

    data_loader = DataLoader(train_data, batch_size=args.data_args.batch_size, shuffle=None,
                                              sampler=sampler,
                                              worker_init_fn=args.seed,
                                              pin_memory=False, drop_last=True)

    return data_loader


