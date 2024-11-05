import os
import torch
import random
import argparse
import glob
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset, random_split
from torch import autograd
import numpy as np
import taskplan_multi
from learning.data import CSVPickleDataset
from taskplan_multi.models.gcn import AnticipateGCN


def get_model_prep_fn_and_training_strs(args):
    print("Training AnticipateGCN Model... ...")
    model = AnticipateGCN(args)
    lr_ep_st_dc = 'tiny'
    prep_fn = taskplan_multi.utils.preprocess_training_data(args)
    train_writer_str = 'train_ap_' + lr_ep_st_dc
    test_writer_str = 'test_ap_' + lr_ep_st_dc
    lr_writer_str = 'learning_rate/ap_' + lr_ep_st_dc
    model_name_str = 'ap_' + lr_ep_st_dc + '.pt'
    best_model = 'ap_best_' + lr_ep_st_dc + '.pt'

    return {
        'model': model,
        'prep_fn': prep_fn,
        'train_writer_str': train_writer_str,
        'test_writer_str': test_writer_str,
        'lr_writer_str': lr_writer_str,
        'model_name_str': model_name_str,
        'best_model': best_model
    }


def train(args, train_path, test_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.empty_cache()

    # Get the model and other training info
    model_and_training_info = get_model_prep_fn_and_training_strs(args)
    model = model_and_training_info['model']
    prep_fn = model_and_training_info['prep_fn']
    train_writer_str = model_and_training_info['train_writer_str']
    test_writer_str = model_and_training_info['test_writer_str']
    lr_writer_str = model_and_training_info['lr_writer_str']
    model_name_str = model_and_training_info['model_name_str']
    best_model_str = model_and_training_info['best_model']

    # Create the datasets and combine them to split
    train_dataset = CSVPickleDataset(train_path, prep_fn)
    # test_dataset = CSVPickleDataset(test_path, prep_fn)
    # combined_dataset = ConcatDataset([train_dataset, test_dataset])

    # # Calculate the lengths for each split (30% and 70%)
    total_length = len(train_dataset)
    # print("Total number of graphs:", total_length)
    test_length = int(0.3 * total_length)
    train_length = total_length - test_length
    _, test_dataset = random_split(
        train_dataset, [train_length, test_length])
    print("Number of training graphs:", len(train_dataset))
    print("Number of testing graphs:", len(test_dataset))

    # raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    train_iter = iter(train_loader)
    train_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, train_writer_str))
    test_iter = iter(test_loader)
    test_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, test_writer_str))

    model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.epoch_size,
        gamma=args.learning_rate_decay_factor)
    index = 0
    autograd.set_detect_anomaly(False)
    best_test_loss = float('inf')
    best_train_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_limit = 5000
    # warmup_steps = 1000
    # warmup_start_lr = 1e-10
    # warmup_end_lr = args.learning_rate
    # lr_increment = (warmup_end_lr - warmup_start_lr) / warmup_steps
    while index < args.num_steps:
        try:
            train_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            train_batch = next(train_iter)
        # if index < warmup_steps:
        #     lr = warmup_start_lr + index * lr_increment
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # elif index == warmup_steps:
        #     # Set to the initial learning rate after warmup
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = args.learning_rate
        out = model.forward({
            'batch_index': train_batch.batch,
            'edge_data': train_batch.edge_index,
            'edge_features': train_batch.edge_features,
            'latent_features': train_batch.x
        }, device)
        # print(torch.isnan(out).any())
        train_loss = model.loss(out,
                                data=train_batch,
                                device=device,
                                writer=train_writer,
                                index=index)

        if index % args.test_log_frequency == 0:
            print(f"[{index}/{args.num_steps}] "
                  f"Train Loss: {train_loss}")

        # Train the system
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if index % args.test_log_frequency == 0:
            try:
                test_batch = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                test_batch = next(test_iter)

            with torch.no_grad():
                out = model.forward({
                    'batch_index': test_batch.batch,
                    'edge_data': test_batch.edge_index,
                    'edge_features': test_batch.edge_features,
                    'latent_features': test_batch.x
                }, device)
                test_loss = model.loss(out,
                                       data=test_batch,
                                       device=device,
                                       writer=test_writer,
                                       index=index)
                print(f"[{index}/{args.num_steps}] "
                      f"Test Loss: {test_loss.cpu().numpy()}")
        if test_loss < best_test_loss and train_loss < best_train_loss:
            best_test_loss = test_loss
            best_train_loss = train_loss
            early_stopping_counter = 0
            best_model_at = index
            # Save the best model
            torch.save(model.state_dict(), os.path.join(args.save_dir,
                                                        best_model_str))
        else:
            early_stopping_counter += 1
            if early_stopping_counter > early_stopping_limit:
                print("Early stopping triggered.")

        # Log the learning rate
        test_writer.add_scalar(lr_writer_str,
                               scheduler.get_last_lr()[-1],
                               index)
        index += 1
        scheduler.step()
    # Saving the model after training
    torch.save(model.state_dict(),
               os.path.join(args.save_dir, model_name_str))
    print(best_model_at)


def get_args():
    parser = argparse.ArgumentParser(
        description="Data generation using ProcTHOR for LOMDP"
    )
    parser.add_argument('--save_dir', type=str, required=False)
    parser.add_argument(
        '--num_steps', type=int, required=False, default=10000,
        help='Number of steps while iterating')
    parser.add_argument(
        '--epoch_size', type=int, required=False, default=10000,
        help='The number of steps in epoch. total_input_count / batch_size')
    parser.add_argument(
        '--learning_rate', type=float, required=False, default=.001,
        help='Learning rate of the model')
    parser.add_argument(
        '--learning_rate_decay_factor', default=0.5, type=float, required=False,
        help='How much learning rate decreases between epochs.')
    parser.add_argument(
        '--data_csv_dir', type=str, required=False,
        help='Directory in which to save the data csv')
    parser.add_argument(
        '--test_log_frequency', type=int, required=False, default=10,
        help='Frequecy of testing log to be generated')

    return parser.parse_args()


def get_data_path_names(args):
    training_data_files = glob.glob(
        os.path.join(args.data_csv_dir, "data_training_*.csv"))
    testing_data_files = glob.glob(
        os.path.join(args.data_csv_dir, "data_training_*.csv"))
    return training_data_files, testing_data_files


if __name__ == "__main__":
    args = get_args()
    # Always freeze your random seeds
    torch.manual_seed(8616)
    random.seed(8616)
    train_path, test_path = get_data_path_names(args)
    # Train the neural network
    train(args=args, train_path=train_path, test_path=test_path)