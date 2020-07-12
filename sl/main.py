from microRTS.model.NNet import NNetWrapper as nn
# from torchsummary import summary

if __name__ == '__main__':
    data_iter = nn.load_data()
    net = nn()
    net.train(data_iter)
