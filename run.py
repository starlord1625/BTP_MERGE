from VSMHN import VSMHN, predict
from dataset import TsEventDataset
import argparse
from data_process import get_air_quality_data
import torch
from tqdm.auto import tqdm
from evaluation import calc_crps, RMSE
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import Utils

def main():
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data set parameters
    argparser.add_argument('--X_context', default=168, help='observing time length', type=int)
    argparser.add_argument('--y_horizon', default=24, help='predicting time length', type=int)
    argparser.add_argument('--window_skip', default=12,
                           help='skipping step for data generation', type=int)
    argparser.add_argument('--train_prop', default=0.97,
                           help='percent of data used for trainning', type=float)

    # network structure
    argparser.add_argument('--h_dim', default=200,
                           help='dimension for ts/event encoder and decoder & size of Embedding vector for Events', type=int)
    argparser.add_argument('--z_dim', default=100,
                           help='dimension for latent variable encoder', type=int)
    argparser.add_argument('--use_GRU', default=True,
                           help='RNN cell type(True:GRU, False:LSTM)', type=bool)

    #Transformer netwrok parameters
    argparser.add_argument('--d_rnn', default=256,
                           help='dimension of RNN layer used in Transformer', type=int)
    argparser.add_argument('--d_inner_hid', default=128,
                           help='Hidden dimension in positive feed forward layer', type=int)
    argparser.add_argument('--d_k', default=16,
                           help='Mk in thedot product Attention', type=int)
    argparser.add_argument('-d_v', default=16,
                           help='Mv in thedot product Attention', type=int)

    argparser.add_argument('-n_head', default=4,
                           help='Number of heads in Multi-head attention', type=int)
    argparser.add_argument('-n_layers', default=4,
                           help='Number of Multi-head attention and positive feedforward layers', type=int)

    # trainning setting
    argparser.add_argument('--lr', default=0.001, help='learning_rate', type=float)
    argparser.add_argument('--dec_bound', default=0.05, help='dec_bound for std', type=float)
    argparser.add_argument('--batch_size', default=400, help='batch size', type=int)
    argparser.add_argument('--epochs', default=100, help='trainning epochs', type=int)
    argparser.add_argument('--device', default='cpu', help='select device (cuda:0, cpu)', type=str)
    argparser.add_argument('--mc_times', default=1000, help='num of monte carlo simulations', type=int)
    argparser.add_argument('-dropout', default=0.1, help='dropout rate', type=float)
    argparser.add_argument('-smooth', default=0.1, help='smoothing rate', type=float)

    argparser.add_argument('-log',help='log file name of rmse score' , default='log.txt',type=str)

    args = argparser.parse_args()

    with open(args.log, 'w') as f:
        f.write('Epoch: [RMSE Score] [CRPS Score]\n')

    data_dict, num_event_type = get_air_quality_data()
    print('Dataset downloaded and preprocessed successfully!')
    dataset = TsEventDataset(data_dict, num_event_type, X_context=args.X_context,
                             y_horizon=args.y_horizon, window_skip=args.window_skip, train_prop=args.train_prop)
    device = torch.device(args.device)
    model = VSMHN(device, dataset.x_dim, 3, dataset.t_dim, args.h_dim,
                    args.z_dim, args.y_horizon, dec_bound=args.dec_bound, use_GRU=args.use_GRU,
                    num_types=num_event_type,
                    d_rnn=args.d_rnn,
                    d_inner=args.d_inner_hid,
                    n_layers=args.n_layers,
                    n_head=args.n_head,
                    d_k=args.d_k,
                    d_v=args.d_v,
                    dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    """ prediction loss function, either cross entropy or label smoothing """
    if args.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(args.smooth, num_event_type, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    for epoch in range(args.epochs):
        epoch_i = epoch + 1
        print('[ Epoch', epoch_i, ']')
        dataset.train_shuffle()
        print('Training:')
        while True:
            end = False
            for batch_loop in tqdm(range(int(((data_dict['time_series'].shape[0]-args.X_context-args.y_horizon)/args.window_skip)*args.train_prop/args.batch_size)),desc='  - (Training)   ', leave=False):
                (X_ts_batch, X_tf_batch, X_event_batch, X_event_arrays), (y_ts_batch,
                                                                        y_tf_batch, y_target), end = dataset.next_batch(args.batch_size, train=True)
                optimizer.zero_grad()
                # loss_like, loss_kl, loss_Transformer = model(X_ts_batch.to(device), X_event_batch.to(device), X_tf_batch.to(device),
                                        # y_ts_batch.to(device), y_tf_batch.to(device), pred_loss_func)
                # loss = -loss_like + loss_kl + loss_Transformer
                loss_like, loss_kl = model(X_ts_batch.to(device), X_event_batch.to(device), X_tf_batch.to(device),
                                        y_ts_batch.to(device), y_tf_batch.to(device), pred_loss_func)
                loss = -loss_like + loss_kl
                loss.backward()
                optimizer.step()
                if end:
                    break
            if end:
                break
        print('Testing:')
        dataset.test_shuffle()
        (X_ts_batch, X_tf_batch, X_event_batch, X_event_arrays), (y_ts_batch, y_tf_batch, y_target), end = dataset.next_batch(100000, train=False)
        indexs = range(-60, 0, 2)
        (X_ts_batch, X_tf_batch, X_event_batch, X_event_arrays), (y_ts_batch, y_tf_batch, y_target) = dataset._get_batch(indexs)
        ts_past, _, _, _, _ = X_ts_batch.to(device), X_event_batch.to(device), X_tf_batch.to(device), y_ts_batch.to(device), y_tf_batch.to(device)
        preds = predict(model, X_ts_batch.to(device), X_event_batch.to(device), X_tf_batch.to(device),  y_tf_batch.to(device), mc_times=args.mc_times)
        (y_ts_batch, y_tf_batch, y_target) = [x.numpy() for x in (y_ts_batch, y_tf_batch, y_target)]
        preds_ori = dataset.dataset['time_series_scaler'].inverse_transform(preds.reshape(-1, 720, 4).reshape(-1, 4)).reshape(-1, 720, 4)
        ts_ori = dataset.dataset['time_series_scaler'].inverse_transform(y_ts_batch.reshape(-1, 4))
        rmse = RMSE(ts_ori, preds_ori.mean(axis=0))
        crps = calc_crps(ts_ori, preds_ori.mean(axis=0), preds_ori.std(axis=0)).mean(axis=0)
        print(f"RMSE scores: {rmse} CRPS scores: {crps}")

        if epoch_i == 1:
            valid_rmse = rmse.reshape(1,-1)
            valid_crps = crps.reshape(1,-1)
        else:
            valid_rmse = np.append(valid_rmse, rmse.reshape(1,-1),axis=0)
            valid_crps = np.append(valid_crps,crps.reshape(1,-1),axis=0)
        print('minimum:')
        print(f"RMSE scores: {np.amin(valid_rmse,axis=0)} CRPS scores: {np.amin(valid_crps,axis=0)}")

        # logging
        with open(args.log, 'a') as f:
            f.write('{epoch}, {rmse}, {crps} \n'
                    .format(epoch=epoch_i, rmse=rmse, crps=crps))
        
    dataset.test_shuffle()
    (X_ts_batch, X_tf_batch, X_event_batch, X_event_arrays), (y_ts_batch, y_tf_batch, y_target), end = dataset.next_batch(100000, train=False)
    print('Forecasting and Plotting:')
    indexs = range(-60, 0, 2)  # plot last 720 time stamps
    plot_size = len(indexs)
    (X_ts_batch, X_tf_batch, X_event_batch, X_event_arrays), (y_ts_batch, y_tf_batch, y_target) = dataset._get_batch(indexs)
    ts_past, _, _, _, _ = X_ts_batch.to(device), X_event_batch.to(device), X_tf_batch.to(device), y_ts_batch.to(device), y_tf_batch.to(device)
    preds = predict(model, X_ts_batch.to(device), X_event_batch.to(device), X_tf_batch.to(device),  y_tf_batch.to(device), mc_times=args.mc_times)
    (y_ts_batch, y_tf_batch, y_target) = [x.numpy() for x in (y_ts_batch, y_tf_batch, y_target)]
    ts_mean = preds.mean(axis=0)
    ts_std = preds.std(axis=0)
    num_plots = min(plot_size, X_ts_batch.shape[0])
    fig, axes = plt.subplots(num_plots, 1, figsize=(4, 2*num_plots))
    for idx in range(num_plots):
        ax = axes[idx]
        X_idx = X_tf_batch[idx, -30:, 0]
        y_idx = y_tf_batch[idx, :, 0]
        for i in range(4):
            ts_past = X_ts_batch[idx, -30:, i]
            ts_future = y_ts_batch[idx, :, i]
            ts_pred = ts_mean[idx, :, i]
            std_pred = ts_std[idx, :, i]
            ax.plot(X_idx, ts_past, color='k', alpha=0.5)
            ax.plot(y_idx, ts_future, color='k', alpha=0.5)
            ax.plot(y_idx, ts_pred, color='r', alpha=0.5)
            ax.fill_between(y_idx, ts_pred-std_pred, ts_pred+std_pred, color='r', alpha=0.1)
    fig.savefig('forecast_plots.png')
    preds_ori = dataset.dataset['time_series_scaler'].inverse_transform(preds.reshape(-1, 720, 4).reshape(-1, 4)).reshape(-1, 720, 4)
    ts_ori = dataset.dataset['time_series_scaler'].inverse_transform(y_ts_batch.reshape(-1, 4))
    rmse = RMSE(ts_ori, preds_ori.mean(axis=0))
    crps = calc_crps(ts_ori, preds_ori.mean(axis=0), preds_ori.std(axis=0)).mean(axis=0)
    print(f"RMSE scores: {rmse}")
    print(f"CRPS scores: {crps}")


if __name__ == '__main__':
    main()
