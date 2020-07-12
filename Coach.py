from torch.utils.data import DataLoader

from microRTS.env.utils import MicroRTSData, collate_fn


class Coach(object):
    """
    This class executes the learning. It uses the funcitons defined in NeuralNet. args are specified in main.py.
    """
    def __int__(self, nnet, args):
        self.nnet = nnet
        self.args = args

    def learn(self):
        """train network with examples"""
        train_dataloader = self.load_data()
        self.nnet.train(train_dataloader)

    #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.args.checkpointFile(i))
    #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def load_data(self):
        dataset = MicroRTSData(game_files='/Users/analysis.jinger/Repository/uRTS/rubensolv_MicroRTS/MicroRTS/uRTSMap16_SttvsMix',
                               reward_file='/Users/analysis.jinger/Repository/uRTS/rubensolv_MicroRTS/MicroRTS/Map16_SttvsMixWinner.csv') # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_dataloader = DataLoader(dataset, shuffle=True, batch_size=self.args.batch_size, collate_fn=collate_fn)                      # nmber_workers
        return train_dataloader

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

