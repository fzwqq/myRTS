import os

from torch.utils.data import DataLoader

from microRTS.env.utils import DotDict, MicroRTSData, collate_fn
from microRTS.model.NNet import NNetWrapper as nn
from microRTS.model.ActorCriticNN import ActorCriticNNet as acnnet
import torch


args = DotDict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 4,
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



# nnet = acnnet(args)
# for param in nnet.parameters():
#     param.requires_grad = False
# nnet.cuda()
# nnet.eval()

nnet = acnnet(args)
filepath = os.path.join('checkpoint', 'checkpoint.pth.tar')
if not os.path.exists(filepath):
    print("No model in path {}".format(filepath))
map_location = None if args.cuda else 'cpu'
checkpoint = torch.load(filepath, map_location=map_location)
nnet.load_state_dict(checkpoint['state_dict'])
for param in nnet.parameters():
    param.requires_grad = False
nnet.cuda()
nnet.eval()

# 这个坑，我相信研究microrts的没有人踩过。史无前例的大坑！


def wrapper(inp0, inp1, inp2, inp3):
    return nnet(inp0.cuda(),  inp1.cuda(),
                inp2.cuda(), inp3.cuda())


def jni_model():
    # method 1
    # device = torch.device("cuda:1")
    # net.to(device)   "to" method requires parameters
    # method 2
    # tensor.cuda()
    model = torch.jit.load('model.pt1')
    exp_stats = torch.randn((1, 107, 16, 16))
    exp_utts = torch.zeros((1, 168))
    exp_units = torch.ones((1, 1, 17))
    exp_length = torch.tensor([1])
    # exp = exp_stats, exp_utts, exp_units, exp_length
    for i in range(1):
        out, v = model(exp_stats, exp_utts, exp_units, exp_length)
        print("test results\n", out, v)


def jni_data_test():
    # method 1
    # device = torch.device("cuda:1")
    # net.to(device)   "to" method requires parameters
    # method 2
    # tensor.cuda()
    model = torch.jit.load('model.pt1')
    dataset = MicroRTSData('/home/Jinger/MicroRTS/uRTSMap16_SttvsMix',
                           '/home/Jinger/MicroRTS/Map16_SttvsMixWinner.csv')
    # dataset = MicroRTSData(game_files= '/home/Jinger/MicroRTS/uRTSMap16_SttvsMix', reward_file= '/home/Jinger/MicroRTS/Map16_SttvsMixWinner.csv')  # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_iter = DataLoader(dataset, shuffle=True, batch_size=1, collate_fn=collate_fn,
                                   drop_last=True, num_workers=0)  # nmber_workers
    for states, utts, units, target_pis, target_vs, length in data_iter:
        length = torch.tensor(length)
        out, v = model(states, utts, units, length)
        print("test results\n", out, v)
        print("target results\n", target_pis, target_vs)
        break
    # exp = exp_stats, exp_utts, exp_units, exp_length
    # for i in range(1):
    #     out, v = model(exp_stats, exp_utts, exp_units, exp_length)
    #     print("test results\n", out, v)


# if __name__ == '__main__':
    # data_iter = nn.load_data()
    # import torch
    # torch.__version__
    # torch.version.cuda
    # torch.cuda.is_available()
    # net = nn()
    # net.script_model(None)

# TODO: java test jni,
# save model
# load model
# compare two outputs between python origin model and java output with the same input data
# 1. java jni input -> output
# 2. python jni input -> ouput
# 3. python input -> output : ! will bsz has effection
# test gpu env
# TODO: done
# 0. train the model and save
# 1. load the model parameters and init model' without batch_size setting and save the jni
# 2 .load the jni and compare the results with and without jni


exp_stats = torch.ones((1, 107, 8, 8))
exp_utts = torch.zeros((1, 168))
exp_units = torch.ones((1, 2, 17))
exp_length = torch.tensor([2])
exp = (exp_stats, exp_utts, exp_units, exp_length)
with torch.no_grad():
    module = torch.jit.trace(wrapper, exp)
module.save("model.pt1")
jni_data_test()

def script_model(model_dir):
    #  out_pi, out_v = self.nnet(states, utts, units, length, hidden)
    exp_stats = torch.ones((1, 107, 8, 8))
    exp_utts = torch.zeros((1, 168))
    exp_units = torch.ones((1, 2, 17))
    exp_length = torch.tensor([2])
    # out, v = nnet(exp_stats, exp_utts, exp_units, exp_length)
    # print("train--------------------results\n", out, v)
    exp = (exp_stats, exp_utts, exp_units, exp_length)
    module = torch.jit.trace_module(wrapper, exp)
    module.save("model.pt")
    # self.jni_model()



def load_checkpoint(folder='checkpoint', filename='checkpoint.pth.tar'):
    nnet = acnnet(args)
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        raise("No model in path {}".format(filepath))
    map_location = None if args.cuda else 'cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    nnet.load_state_dict(checkpoint['state_dict'])
    for param in nnet.parameters():
        param.requires_grad = False
    nnet.cuda()
    nnet.eval()
    return nnet
    #TODO 1. multi maps