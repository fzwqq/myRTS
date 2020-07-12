import torch.nn as nn
import torch
import torch.nn.functional as F


class ActorCriticNNet(nn.Module):
    def __init__(self, args):
        super(ActorCriticNNet, self).__init__()
        self.args = args
        # state
        self.conv_1 = nn.Sequential(
            nn.Conv2d(args.map_channels, args.num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(args.num_channels),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(args.num_channels, args.num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(args.num_channels),
            nn.LeakyReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(args.num_channels, args.num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(args.num_channels),
            nn.LeakyReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(args.num_channels, args.num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(args.num_channels),
            nn.LeakyReLU()
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(args.num_channels, args.num_map_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(args.num_map_channels),  # final should be reduced to decrease the parameters
            nn.LeakyReLU()
        )
        self.ada_pool = nn.AdaptiveMaxPool2d(args.pooled_size)
        # utt
        self.mlp_1 = nn.Sequential(
            nn.Linear(args.utt_features_size, args.num_utt_out),
            nn.LeakyReLU(),
            # nn.Linear(64, 64),
            # nn.LeakyReLU,
        )
        mu_size = self._get_map_utt_size(args.num_utt_out, args.num_map_channels, args.pooled_size)
        self.mlp_2 = nn.Sequential(
            nn.Linear(mu_size, args.num_mu_size),
            nn.LeakyReLU(),
            # nn.Linear(args.num_channels, 1),
            # nn.Tanh()
        )
        # value head
        self.value = nn.Sequential(
            nn.Linear(args.num_mu_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

        # unit
        self.rnn = nn.LSTM(
            input_size=args.unit_features_size,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.lstm_num_layers,
            batch_first=True,
        )
        muu_size = self._get_map_utt_unit_size(args.num_mu_size, args.lstm_hidden_size)
        self.mlp_pi = nn.Sequential(
            nn.Linear(muu_size, args.action_size),
            nn.LeakyReLU(),
        )

        # self.mlp_unit = nn.Sequential(
        #     nn.Linear(args.unit_features_size, args.num_unit_out),
        #     nn.LeakyReLU(),
        # )
        # muu_size = self._get_map_utt_unit_size(args.num_mu_size, args.num_unit_out)
        # self.rnn = nn.LSTM(
        #     input_size=muu_size,
        #     hidden_size=args.action_size,
        #     num_layers=args.lstm_num_layers,
        #     batch_first=True,
        # )
        # self.fc = nn.Linear(hidden_size, out_size)

    def _get_map_utt_size(self, num_utt_out, num_map_channels, pooled_size):
        return pooled_size * pooled_size * num_map_channels + num_utt_out

    def _get_map_utt_unit_size(self, num_mu_size, num_hidden_size):
        return num_mu_size + num_hidden_size

    def forward(self, states, utts, units, length):
        # print("batch_size:", self.args.batch_size)
        # print(states.size()[0])
        # self.args.batch_size = states.size()[0]
        """
        :param states:
        :param utts:
        :param units:
        :param length:
        :param hidden:
        :return:
        """
        # assert states.size()[0] == self.args.batch_size
        batch_size = states.size()[0]
        hidden = self._init_hidden(batch_size, self.args.lstm_num_layers, self.args.lstm_hidden_size)
        # assert self.args.batch_size == len(length)
        seq_size = length[0]  # maximum seq_len
        # map conv
        x = self.conv_1(states)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.ada_pool(x)
        # m = x.view(self.args.batch_size, -1)
        m = x.contiguous().view(batch_size, self.args.pooled_size * self.args.pooled_size * self.args.num_map_channels)
        # utt mlp
        utt = self.mlp_1(utts)
        mu = torch.cat((m, utt), dim=-1)
        # map & utt mlp
        mu = self.mlp_2(mu)

        # value
        v = self.value(mu)

        # dim, expand, cat
        x = mu.unsqueeze(1)
        x = x.expand(-1, seq_size, -1)

        # unit
        u = torch.nn.utils.rnn.pack_padded_sequence(units, length, batch_first=True)
        out, _ = self.rnn(u, hidden)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # policy
        x = torch.cat((x, out), dim=-1)
        pi = self.mlp_pi(x)
        # print(pi)
        pi = pi.contiguous().view(batch_size * seq_size, self.args.action_size)
        return -F.log_softmax(pi, dim=1), torch.tanh(v)
        # assert units.is_contiguous() == True
        # u = units.view(-1, self.args.unit_features_size)
        # u = self.mlp_unit(u)
        # u = u.view(self.args.batch_size, seq_size, -1)
        # cat
        # o = torch.cat((o, u), dim=-1)
        # lstm
        # o = torch.nn.utils.rnn.pack_padded_sequence(o, length, batch_first=True)
        # out, _ = self.rnn(o, hidden)
        # out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # pi = out.contiguous().view(self.args.batch_size * seq_size, self.args.action_size)
        # utt = f1(utts)
        # state = f2(states)
        # unit_i = f5(unit_i)
        # features = f3(utt, state)
        # v =  f4(feature
        # h0 = (num_layers, batch_size, out_features)
        # c0 = torch.randn(2, 3, 20)
        # out, hn, cn = self.rnn(input=train_units, (h0, c0))
        # # 1 to 1 ?
        # out[:,:,]
        # print(output.size(), hn.size(), cn.size())
        # pi = f6(features, unit_i)
        # data = rnn_utils.pack_padded_sequence(data1, length, batch_first=True)
        #     print(data1.shape, data2, length)
        #     output, hidden = net(data.float())
        #     if flag == 0:
        #         output, out_len = rnn_utils.pad_packed_sequence(output, batch_first=True)
        #         print(output.shape)
        #         print(output)
        #         flag = 1
        # assert units.is_contiguous == True
        # units = units.contiguous.view(batch_size * seq_size, -1)
        # units = self.fc(units)
        # units = units.contiguous.view(batch_size, seq_size, -1)
        # # Left:
        # out_l = out_l.unsqueeze(1)
        # out_l = out_l.expand(-1, seq_size, -1)
        # com   = torch.cat((out_l, out), dim=-1)
        #
        # com = torch.nn.utils.rnn.pack_padded_sequence(com, length, batch_first=True)
        # out, hidden = self.rnn(com, hidden)
        # out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        #
        # pi = out.contiguous().view(batch_size * seq_size, out_size)
        # add? mlp on pi

        # 自己能力得判断
        # 全局信息得判断
        # 队友决策信息得补充。 （option ： mlp）
        # [b, s, f] -> [b, s, f] + [b, f2]     ->   [b, s, f2]  -> [b, s, o] -> [b * s, o]
        # in.contiguous().view(batch_size * seq_size, hidden_size), cat

        # pi: shape[batch_size * seq_size, out_size]

        # value: shape[batch_size, 1]
        #
        # TODU: to check order more effeicient, and then check the value and shape are correct.

    def _init_hidden(self, bsz, nlayers, nhid):
        weight = next(self.parameters())
        return (weight.new_zeros(nlayers, bsz, nhid),
                weight.new_zeros(nlayers, bsz, nhid))
