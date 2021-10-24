import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, kl_divergence
from tqdm.auto import tqdm
from transformer.Models import Transformer
import transformer.Constants as Constants
import Utils


class Encoder_TS(nn.Module):
    def __init__(self, x_dim, h_dim, phi_x, phi_tf, use_GRU=True):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.phi_x = phi_x
        self.phi_tf = phi_tf
        self.use_GRU = use_GRU
        if (use_GRU):
            self.rnn = nn.GRU(2*h_dim, h_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(2*h_dim, h_dim, batch_first=True)

    def forward(self, src, tf):
        # src : (batch_size, seq_len, x_dim)
        x_h = self.phi_x(src)
        tf_h = self.phi_tf(tf)
        joint_h = torch.cat([x_h, tf_h], -1)
        if (self.use_GRU):
            outputs, hidden = self.rnn(joint_h)
        else:
            outputs, (hidden, cell_state) = self.rnn(joint_h)
        return hidden


class Encoder_Event(nn.Module):
    def __init__(self, x_dim, h_dim, bound=0.05, use_GRU=True):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.use_GRU = use_GRU
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim))
        if (use_GRU):
            self.rnn = nn.GRU(h_dim, h_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(h_dim, h_dim, batch_first=True)

    def forward(self, src):
        x_h = self.phi_x(src)
        if (self.use_GRU):
            outputs, hidden = self.rnn(x_h)
        else:
            outputs, (hidden, cell_state) = self.rnn(x_h)
        return hidden


class Hidden_Encoder(nn.Module):
    def __init__(self, h_dim, z_dim):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.enc_mean = nn.Sequential(
            nn.Linear(h_dim, z_dim))
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

    def forward(self, h):
        return Normal(self.enc_mean(h), self.enc_std(h))


class Hidden_Decoder(nn.Module):
    def __init__(self, h_dim, z_dim):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.dec = nn.Sequential(
            nn.Linear(z_dim, h_dim))

    def forward(self, z):
        return self.dec(z)


class Decoder_TS(nn.Module):
    def __init__(self, x_dim, h_dim, phi_x, phi_tf, bound=0.05, use_GRU=True):
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.phi_x = phi_x
        self.phi_tf = phi_tf
        self.use_GRU = use_GRU
        if (use_GRU):
            self.rnn = nn.GRUCell(2*h_dim, 2*h_dim)
        else:
            self.rnn = nn.LSTMCell(2*h_dim, 2*h_dim)
        self.dec_mean = nn.Sequential(
            nn.Linear(2*h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim))
        self.dec_std = nn.Sequential(
            nn.Linear(2*h_dim, x_dim),
            nn.Softplus())
        self.bound = bound

    def forward(self, x_t, tf_t, hidden):
        x_h = self.phi_x(x_t)
        tf_h = self.phi_tf(tf_t)
        joint_h = torch.cat([x_h, tf_h], -1)
        if (self.use_GRU):
            hidden = self.rnn(joint_h, hidden)
        else:
            (hidden, cell_state) = self.rnn(joint_h, hidden)
        x_mu = self.dec_mean(hidden)
        x_std = self.dec_std(hidden) + x_t.new_tensor([self.bound])
        if (self.use_GRU):
            return Normal(x_mu, x_std), hidden
        else:
            return Normal(x_mu, x_std), (hidden, cell_state)


class VSMHN(nn.Module):
    def __init__(
            self, device, ts_dim, event_dim, tf_dim, h_dim, z_dim, forecast_horizon, dec_bound=0.1, use_GRU=True,k=5,
            num_types=3, d_rnn=128, d_inner=1024,n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()
        self.device = device
        self.ts_dim = ts_dim
        self.event_dim = event_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.forecast_horizon = forecast_horizon
        self.use_GRU = use_GRU
        self.k=k
        self.num_types=num_types
        self.d_rnn=d_rnn
        self.d_inner=d_inner
        self.n_layers=n_layers
        self.n_head=n_head
        self.d_k=d_k
        self.d_v=d_v
        self.dropout=dropout
        self.phi_ts = nn.Sequential(nn.Linear(ts_dim, h_dim),
                                    nn.ReLU(),
                                    nn.Linear(h_dim, h_dim))
        self.phi_tf = nn.Sequential(nn.Linear(tf_dim, h_dim),
                                    nn.ReLU(),
                                    nn.Linear(h_dim, h_dim))
        self.ts_encoder = Encoder_TS(ts_dim, h_dim, self.phi_ts, self.phi_tf, self.use_GRU)
        # self.event_encoder = Encoder_Event(event_dim, h_dim, self.use_GRU)
        self.event_encoder = Transformer(
                                    num_types=num_types,
                                    d_model=h_dim,
                                    d_rnn=d_rnn,
                                    d_inner=d_inner,
                                    n_layers=n_layers,
                                    n_head=n_head,
                                    d_k=d_k,
                                    d_v=d_v,
                                    dropout=dropout,
                                )
        self.ts_decoder = Decoder_TS(ts_dim, h_dim, self.phi_ts,
                                     self.phi_tf, bound=dec_bound, use_GRU=self.use_GRU)
        self.posterior_encoder = Hidden_Encoder(2*h_dim, z_dim)
        self.prior_encoder = Hidden_Encoder(2*h_dim, z_dim)
        self.hidden_decoder = Hidden_Decoder(h_dim, z_dim)
        self.temporal_decay = np.linspace(1.5, 0.5, forecast_horizon)
        self.phi_dec = nn.Sequential(nn.Linear(3*h_dim, 2*h_dim))

    def forward(self, ts_past, event_past, ts_tf_past, ts_trg, tf_future, pred_loss_func):
        # seq : shape [batch_size, seq_len, x_dim]
        # trg : shape [batch_size, seq_len, x_dim]
        # tf_fugure shape [batch_size, seq_len, tf_dim]
        # event_past: shape [batch_size, seq_len, event_dim]
        #print('sos')
        #print(ts_past.shape)
        #print(event_past.shape)
        #print(ts_tf_past.shape)
        #print(ts_trg.shape)
        #print(tf_future.shape)
        ts_hidden = self.ts_encoder(ts_past, ts_tf_past).squeeze(0)
        ts_hidden_tau = self.ts_encoder(torch.cat([ts_past, ts_trg], dim=1),
                                        torch.cat([ts_tf_past, tf_future], dim=1)).squeeze(0)
        # event_hidden = self.event_encoder(event_past).squeeze(0)
        event_hidden, prediction = self.event_encoder(event_past[:,:,0].to(torch.int64),event_past[:,:,-1])
        #print(self.ts_encoder(ts_past, ts_tf_past).shape)
        #print(ts_hidden.shape)
        # joint_hidden = torch.cat([ts_hidden, event_hidden], dim=-1)
        # joint_hidden_tau = torch.cat([ts_hidden_tau, event_hidden], dim=-1)
        joint_hidden = torch.cat([ts_hidden, event_hidden[:,-1,:]], dim=-1)     # Taking only last hidden representation
        joint_hidden_tau = torch.cat([ts_hidden_tau, event_hidden[:,-1,:]], dim=-1) # Taking only last hidden representation
        pz_rv = self.prior_encoder(joint_hidden)
        qz_rv = self.posterior_encoder(joint_hidden_tau)
        z = qz_rv.rsample()
        z_dec = self.hidden_decoder(z)
        hidden_dec = self.phi_dec(torch.cat([joint_hidden, z_dec], dim=-1))
        if not self.use_GRU:
            hidden_dec = (hidden_dec, torch.zeros(hidden_dec.shape).to(self.device))
        ts_t = ts_past[:, -1, :]
        likelihoods = []
        for t in range(self.forecast_horizon):
            tf_t = tf_future[:, t, :]
            ts_t_rv, hidden_dec = self.ts_decoder(ts_t, tf_t, hidden_dec)
            likelihoods.append(ts_t_rv.log_prob(ts_trg[:, t, :]))
            ts_t = ts_t_rv.sample()
        likelihoods = torch.stack(likelihoods, dim=1)
        #my code
        #hidden_dec_array = torch.cat([hidden_dec.unsqueeze(0)]*self.k,dim=0)
        #ts_t_array = torch.cat([ts_past[:, -1, :].unsqueeze(0)]*self.k,dim=0)
        #likelihoods = []
        #for t in range(self.forecast_horizon):
        #    tf_t_array = torch.cat([tf_future[:, t, :].unsqueeze(0)]*self.k,dim=0)
        #    for idx in range(self.k): 
        #        ts_t = ts_t_array[idx,:,:]
        #        tf_t = tf_t_array[idx,:,:]
        #        ts_t_rv, hidden_dec_array[idx] = self.ts_decoder(ts_t, tf_t, hidden_dec_array[idx])
        #        likelihoods.append(ts_t_rv.log_prob(ts_trg[:, t, :]))
        #        ts_t_array[idx] = (ts_t_rv.sample()).unsqueeze(0)
        #print('sos')
        #likelihoods = torch.stack(likelihoods, dim=1)
        #print(likelihoods.shape)
        kls = kl_divergence(qz_rv, pz_rv)

        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(self.event_encoder, event_hidden, event_past[:,:,-1], event_past[:,:,0].to(torch.int64))
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_past[:,:,0].to(torch.int64), pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_past[:,:,-1])

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss_Transformer = event_loss + pred_loss + se / scale_time_loss
        # return torch.mean(torch.sum(likelihoods, (-1, -2))), kls.mean(), loss_Transformer
        return torch.mean(torch.sum(likelihoods, (-1, -2))), kls.mean()


def predict(model, ts_past, event_past, ts_tf_past, tf_future, mc_times=100):
    # seq : shape [batch_size, seq_len, x_dim]
    # trg : shape [batch_size, seq_len, x_dim]
    # tf_fugure shape [batch_size, seq_len, tf_dim]
    # event_past: shape [batch_size, seq_len, event_dim]
    ts_hidden = model.ts_encoder(ts_past, ts_tf_past).squeeze(0)
    # event_hidden = model.event_encoder(event_past).squeeze(0)
    event_hidden, prediction = model.event_encoder(event_past[:,:,0].to(torch.int64),event_past[:,:,-1])
    # joint_hidden = torch.cat([ts_hidden, event_hidden], dim=-1)
    joint_hidden = torch.cat([ts_hidden, event_hidden[:,-1,:]], dim=-1) # Taking only last hidden representation
    pz_rv = model.prior_encoder(joint_hidden)
    predictions = np.zeros(
        shape=(mc_times, tf_future.shape[0], tf_future.shape[1], tf_future.shape[2]))
    for idx in tqdm(range(mc_times),desc='  - (Testing) ', leave=False):
        z = pz_rv.sample()
        z_dec = model.hidden_decoder(z)
        hidden_dec = model.phi_dec(torch.cat([joint_hidden, z_dec], dim=-1))
        if (not model.use_GRU):
            hidden_dec = (hidden_dec, torch.zeros(hidden_dec.shape).to(model.device))
        ts_t = ts_past[:, -1, :]
        for t in range(model.forecast_horizon):
            tf_t = tf_future[:, t, :]
            ts_t_rv, hidden_dec = model.ts_decoder(ts_t, tf_t, hidden_dec)
            ts_t = ts_t_rv.sample()
            predictions[idx, :, t, :] = ts_t.cpu().numpy()
    return predictions
