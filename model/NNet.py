import os
import time

import torch
from torch import optim #distribution.optim if para train
from torch.utils.data import DataLoader

from .ActorCriticNN import ActorCriticNNet as acnnet
from ..NNet import NeuralNet
from ..env.misc import AverageMeter
from ..env.utils import DotDict, MicroRTSData, collate_fn

args = DotDict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 500,
    'batch_size': 64,
    'map_channels': 107,
    'num_channels': 64,         # conv_in of the map filters size
    'num_map_channels': 64,     # conv_out of the map filters size
    'pooled_size': 8,            # target output size of conv_out
    'utt_features_size': 168,        #
    'num_utt_out': 64,
    'num_mu_size': 256,
    'unit_features_size': 17,
    'lstm_hidden_size': 64,
    'action_size': 65,
    'lstm_num_layers': 2,
    'cuda': torch.cuda.is_available(),
})


class NNetWrapper(NeuralNet):
    def __init__(self):
        super(NNetWrapper, self).__init__()
        self.nnet = acnnet(args)
        print(self.nnet)
        if args.cuda:
            self.nnet.cuda()

    def train(self, data_iter):
        """
        Data_iter: data iteration(batch_size)
        Data_length: number of examples
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('Epoch ：：： ' + str(epoch + 1))
            self.nnet.train()
            batch_idx = 0
            # batch_size = args.batch_size
            # hidden = self.nnet.init_hidden(args.batch_size, args.lstm_num_layers, args.lstm_hidden_size)

            for states, utts, units, target_pis, target_vs, length in data_iter:

                if args.cuda:
                    states, utts, units, target_pis, target_vs = states.contiguous().cuda(), utts.contiguous().cuda(), \
                                                            units.contiguous().cuda(), target_pis.contiguous().cuda(), \
                                                            target_vs.contiguous().cuda()
                # measure data loading time
                # state[batch_size, channel, height, width]
                # utt[batch_size, length1]
                # units[batch_size, seq_len, length2]
                # seq==len(length) * length2==length[0]
                # + + + +   length[0] 4
                # + + +     length[1] 3
                # + + +     length[2] 3
                # + +       length[3] 2
                # target_pis[batch_size, seq_len, length3]
                # target_vs[batch_size, length4]
                # compute output
                # 1. for units_length, compute total_loss / batch_size；
                # 2.    compute the seq_units_loss
                # 3.
                # Q: 怎么定义损失函数
                # 确定了输入一个batch * 1seq_len * units_feature; for seq_len, in the each round, you can compute the batch
                # the batch inputting to the nn depends on the column of the seq_len.
                # when cat batch' and map_features, the dim of map_features == batch >= batch'
                # the total_loss = all the loss from label_units_output and target.
                # the avg_loss = the total_loss / batch_size
                # Q: batch' <= batch, 但是拼接要求维度只能一个不同，所以这里就得相同。也就意味着输入得一定是batch个。只能再计算loss得时候
                # Q: 测试时怎么求，
                #  if mlp:
                #     statck(units)-> expand(map_features) -> cat -> [units_len, features] -> 输入网络 -> getoutput
                #  else lstm:
                #     input units one by one,
                out_pi, out_v = self.nnet(states, utts, units, length)

                seq_size = length[0]  # after sorting, index:0 maximum
                target_pis = target_pis.view(args.batch_size * seq_size, args.action_size)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v  = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # compute gradient and do Adam step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # # record loss
                # pi_losses.update(l_pi.item(), states.size(0))
                # v_losses.update(l_v.item(), states.size(0))
                # measure elapsed time
                if batch_idx % 10 == 0:
                    print("batch_", batch_idx, " total_loss: ", total_loss.item() )
                batch_idx += 1
        self.save_checkpoint()

    @staticmethod
    def load_data():
        # dataset = MicroRTSData(game_files='/Users/analysis.jinger/Repository/uRTS/rubensolv_MicroRTS/MicroRTS/uRTSMap16_SttvsMix',
        # reward_file='/Users/analysis.jinger/Repository/uRTS/rubensolv_MicroRTS/MicroRTS/Map16_SttvsMixWinner.csv') # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # local
        # dataset = MicroRTSData('/Users/analysis.jinger/Repository/uRTS/Jin_MicroRTS/MicroRTS/uRTSMap16_SttvsMix',
        #                        '/Users/analysis.jinger/Repository/uRTS/Jin_MicroRTS/MicroRTS/Map16_SttvsMixWinner.csv')
        # remote
        dataset = MicroRTSData('/home/Jinger/MicroRTS/uRTSMap16_SttvsMix',
                               '/home/Jinger/MicroRTS/Map16_SttvsMixWinner.csv')
        # dataset = MicroRTSData(game_files= '/home/Jinger/MicroRTS/uRTSMap16_SttvsMix', reward_file= '/home/Jinger/MicroRTS/Map16_SttvsMixWinner.csv')  # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_data_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn,
                                       drop_last=True, num_workers=0)                      # nmber_workers
        return train_data_loader

    def loss_pi(self, targets, outputs):
        # check targets.size()
        return torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(filepath)
            print("Checkpoint dir does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("checkpoint directory exists!")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

    @staticmethod
    def getCheckpointFile(iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    # def wrapper(inp0, inp1, inp2, inp3):
    #     print(inp0, inp1, inp2, inp3)
    #     inp0 = inp0.cuda().detach()
    #     inp1 = inp1.cuda().detach()
    #     inp2 = inp2.cuda().detach()
    #     inp3 = inp3.cuda().detach()
    #     out, v = nnet(inp0, inp1,
    #                   inp2, inp3)
    #     print(out, v)
    #     return out.to("cpu"), v.to("cpu")

    def script_mod(self, model_dir):
        #  out_pi, out_v = self.nnet(states, utts, units, length, hidden)
        nnet = acnnet(args)
        if args.cuda:
            nnet.cuda()
        nnet.eval()
        print("-------------")
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print("-----------------cuda---------------")
        # nnet.load_state_dict(torch.load(model_dir))
        exp_stats = torch.ones((1, 107, 8, 8))
        exp_utts = torch.zeros((1, 168))
        exp_units = torch.ones((1, 2, 17))
        exp_length = torch.tensor([2])
        # out, v = nnet(exp_stats, exp_utts, exp_units, exp_length)
        # print("train--------------------results\n", out, v)
        exp = (exp_stats, exp_utts, exp_units, exp_length)
        module = torch.jit.trace(nnet, exp)
        module.save("model.pt")
        self.jni_model()

    def jni_model(self):
        # method 1
        # device = torch.device("cuda:1")
        # net.to(device)   "to" method requires parameters
        # method 2
        # tensor.cuda()
        model = torch.jit.load('model.pt1')
        exp_stats = torch.ones((1, 107, 8, 8))
        exp_utts = torch.zeros((1, 168))
        exp_units = torch.ones((1, 1, 17))
        exp_length = torch.tensor([1])
        # exp_h0 = torch.zeros((2, 1, 64))
        # exp_c0 = torch.zeros((2, 1, 64))
        # exp_hd = (exp_h0, exp_c0)
        # exp = exp_stats, exp_utts, exp_units, exp_length
        for i in range(10):
            out, v = model(exp_stats, exp_utts, exp_units, exp_length)
            print("test results\n", out, v)