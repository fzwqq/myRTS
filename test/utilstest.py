from torch.utils.data import DataLoader

# from ..env.utils import MicroRTSDataset, MicroRTSData, collate_fn
# from ..env.utils import MicroRTSData, collate_fn, MicroRTSDataset
from microRTS.env.utils import MicroRTSData, collate_fn, MicroRTSDataset


def main():
    dataset = MicroRTSDataset('/Users/analysis.jinger/Repository/uRTS/rubensolv_MicroRTS/MicroRTS/uRTSMap16_SttvsMix',
                              '/Users/analysis.jinger/Repository/uRTS/rubensolv_MicroRTS/MicroRTS/Map16_SttvsMixWinner.csv')
    data_iter_fn = dataset.data_iter_random
    # cnt = 0
    for sample in data_iter_fn(10):
        print(type(sample))
        print(sample['state'].size())
        print(sample['utt'].size)
        print(sample['rewards'])
        print(sample['counts'])
        for i, count in enumerate(sample['counts'].tolist()):
            print(count)
            print(type(sample['units'][i].shape))
            print(sample['units'][i].shape)
            print(sample['actions'][i].shape)
        # print(sample['state'].size(), sample['utt'].size(), sample['units'].size(), sample['actions'].size(),
        #       sample['rewards'], sample['counts'])
        # for sample in samples:
        #     print(sample[])
        # cnt += 1
        # print(cnt)



def testDataLoader():
    dataset = MicroRTSData('/Users/analysis.jinger/Repository/uRTS/Jin_MicroRTS/MicroRTS/uRTSMap16_SttvsMix',
                              '/Users/analysis.jinger/Repository/uRTS/Jin_MicroRTS/MicroRTS/Map16_SttvsMixWinner.csv')
    train_dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    epoches = 2
    for i in range(epoches):
        tmp = 1
        for states, utts, train_units, target_actions, rewards, length in train_dataloader:
            if tmp % 200 == 0:
                print("length is: ", length)
                print("target_actions are", target_actions)
                print("rewards are:", rewards)
                print("states shape:", states.shape)
                print("utts shape:", utts.shape)
                print("units shape:", train_units.shape)
                print("actions shape:", target_actions.shape)
                print("rewards shape:", rewards.shape)
                print("lengths shape:", length.shape)
            tmp += 1
        print('--------------------', 'epoch: ', i)
        print(tmp)


if __name__ == "__main__":
    # main()
    testDataLoader()