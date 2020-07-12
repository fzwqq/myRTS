from __future__ import print_function, division
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

import warnings
import csv
import sys

from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils

warnings.filterwarnings("ignore")


# load x files compose 1 ndArray.


# test how many data could be loaded once and the status keep normal
def load_data(game_files, reward_file):
    data = []
    reward_file = reward_file
    # reward_file =
    # print(type(reward_file), reward_file)
    reward_file = reward_file
    with open(reward_file, 'r') as file:
        reward = file.readlines()
    print(reward)
    file_name = game_files
    for i in range(4):
        file_path = file_name + str(i) + '.csv'
        with open(file_path, 'r') as file:
            # print(file_path)
            csv_file = csv.reader(file)
            head = True
            for _, line in enumerate(csv_file):
                if head:
                    head = False
                    print(reward[i + 1][0])  # [0] : value, [1] : '/n'
                    continue
                data.append({"game_data": line, "reward": reward[i + 1][0]})
            print("size of: ", sys.getsizeof(data))
            print("len of: ", len(data))
    len(data)
    return data


def collate_fn(batch_samples):
    #     a.sort(key=lambda data: len(data), reverse=True)
    #     data_length = [len(data) for data in a]
    #     train_data = rnn_utils.pad_sequence(a, batch_first=True, padding_value=0)
    #     print("train_data", train_data,data_length)
    #     print(ab)
    # 只对这一个batch的数据进行了排序，并非对整个数据集进行排序
    # batch_samples: [{'k1':v1, 'k2':v2}, {'k1':v3, 'k2':v4},]
    # for item in batch_samples:
    #     print(len(item['units']))
    batch_samples.sort(key=lambda data: len(data['units']), reverse=True)

    states = [item['state'] for item in batch_samples]
    utts   = [item['utt'] for item in batch_samples]
    rewards= [item['reward'] for item in batch_samples]

    units  = [torch.tensor(item['units']).float() for item in batch_samples]
    actions= [torch.tensor(item['actions']).float() for item in batch_samples]
    length = [len(data) for data in units]

    train_units = rnn_utils.pad_sequence(units, batch_first=True, padding_value=0)
    target_actions = rnn_utils.pad_sequence(actions, batch_first=True, padding_value=0)

    return torch.tensor(states).float(), torch.tensor(utts).float(), train_units, target_actions, torch.tensor(rewards).float(), length


class MicroRTSData(Dataset):
    """microRTS dataset."""

    def __init__(self, game_files, reward_file, transform=None):
        self.dataset = load_data(game_files, reward_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.dataset[idx]
        spatial_encode, utt_encode, units_encode, actions_encode, units_size, player = encode_data(data['game_data'])
        reward = int(data['reward'])
        reward = 1.0 if reward == int(player) else -1.0
        sample = {'state': spatial_encode, 'utt': utt_encode,
                  'units': units_encode, 'actions': actions_encode,
                  'reward': reward}

        if self.transform:
            sample = self.transform(sample)
        return sample


class MicroRTSDataset(object):
    """microRTS dataset."""

    def __init__(self, game_files, reward_file):
        """
        Args:
            dataset (list[{game_data, reward}])
        """
        self.dataset = load_data(game_files, reward_file)
        self.length = len(self.dataset)

    # could change to random index in every time visit, not create the random indices once
    def data_iter_random(self, batch_size):
        example_indices = list(range(self.length - 1))
        random.shuffle(example_indices)
        epoch_size = self.length // batch_size
        # [batch_size, channels, height, width],[batch_size, utt_features],
        # [batch_size, units_size, units_features], [batch_size, units_size, (actions=65)],
        # [batch_size, reward], [batch_size, units_size]
        # spatial_features, utt_features, input_units_features, units_actions_labels

        for i in range(epoch_size):
            states, utts, units, probs, rewards, counts = [], [], [], [], [], []
            indices = i * batch_size
            batch_index = example_indices[indices: batch_size + indices]
            for idx in batch_index:
                data = self.dataset[idx]
                spatial_encode, utt_encode, units_encode, actions_encode, units_size, player = encode_data(data['game_data'])
                states.append(spatial_encode)
                utts.append(utt_encode)
                units.append(units_encode)
                probs.append(actions_encode)
                if player == data['reward']:
                    rewards.append(1.0)
                else:
                    rewards.append(-1.0)
                counts.append(int(units_size))
            states_feature, utts_feature, units_feature, probs_feature, rewards_label, num_counts = np.array(
                states), np.array(utts), np.array(units), np.array(probs), np.array(rewards), np.array(counts)
            # if necessary reshape
            # sample = {'state': spatial_encode, 'utt': utt_encode,
            #           'units': units_encode, 'actions': actions_encode,
            #           'rewards': np.array(int(data['reward'])), 'counts': np.array(int(units_size))}
            #     sample = {'spatial_features': states, 'utts_features': utts,
            #               'input_units': units, 'action_prob': probs,
            #               'rewards': rewards, 'counts': counts}
            #     if self.transform:
            #         sample = self.transform(sample)
            sample = {'state': torch.from_numpy(states_feature), 'utt': torch.from_numpy(utts_feature),
                      'units': units_feature, 'actions': probs_feature,
                      'rewards': torch.from_numpy(rewards_label), 'counts': torch.from_numpy(num_counts)}
            yield sample


def encode_unit_type(value):
    if value == 0.0:
        return -1
    if value == 1.0:
        return 0
    if value == 2.0:
        return 1
    if value == 3.0:
        return 2
    if value == 4.0:
        return 3
    if value == 5.0:
        return 4
    if value == 6.0:
        return 5
    if value == 7.0:
        return 6
    print("err type")


def encode_unit_hp_ratio(hp_ratio):
    """one hot encoding of hp ratio range: None, 0%~10%, 10%~20%, 20%~40%, 40%~80%, 80%~100%
        6 bits
        Arguments:
            hp_ratio {float} -- The given hp ratio
        return: position in one hot encoding
        """
    if hp_ratio == 0:
        return 0
    elif hp_ratio <= 0.2:
        return 1
    elif hp_ratio <= 0.4:
        return 2
    elif hp_ratio <= 0.6:
        return 3
    elif hp_ratio <= 0.8:
        return 4
    else:
        return 5


def encode_resources(num_res):
    if num_res == 0:
        return 0
    if num_res == 1:
        return 1
    if num_res == 2:
        return 2
    if num_res <= 4:
        return 3
    if num_res <= 8:
        return 4
    if num_res <= 16:
        return 5
    if num_res <= 32:
        return 6
    if num_res > 32:
        return 7
    print("error resource")
    return 0


def find_id(action):
    idx = [i for i in range(len(action)) if action[i] == 1]
    if idx:
        return idx[0]
    return -1


def encode_data(data):
    # 1. spatial_features
    # 2. utts_features
    # 3. input_units
    # 4. action_prob
    # column            index
    # Height            0
    # Width             1
    # P0Res             2
    # P1Res             3
    # walkable          4 + [0,height*width)
    # utt               [4 + height*width , 4 + height * width + 18*7 )
    # (units, actions, )  [4 + height * width + 18*7 + 70*pos_i, 4 + height * width + 18*7 + 4 + 70*pos_i),
    #                   [4 + height * width + 18*7 + 4 + 70*pos_i, 4 + height * width + 18*7 + 4 + 65 + 70*pos_i)
    # unit's action degree 4 + height * width + 18*7 + 4 + 65 + 70*pos_i
    # (unit's input, actions prob) 4 + height * width + 18 * 7 + 70 * width * height , 4 + height * width + 18 * 7 + 70 * width * height +
    # input units                [4 + height*width + 7 * 18 + height*width * 70,      4 + height*width + 7 * 18 + height*width * 70 + 6)
    # input units actions prob   [4 + height*width + 7 * 18 + height*width * 70 + 6,  4 + height*width + 7 * 18 + height*width * 70 + 6 + 65)
    # player       idx: [-1] ==   4 + height*width + 7 * 18 + height*width * 70 + (6 + 65) * input_units_num  == col_length - 1; start from idx 0
    data = [float(data[i]) for i in range(len(data))]
    height = int(data[0])
    width = int(data[1])
    p0res = data[2]
    p1res = data[3]
    walkable = data[4: 4 + height * width]
    utt = data[4 + height * width: 4 + height * width + 18 * 7]
    pos_units = []
    pos_actions = []
    pos_action_degree = []
    idx, idx_l, idx_r = 0, 0, 0
    for pos in range(height * width):
        idx_l = 4 + height * width + 18 * 7 + pos * 70
        idx_r = idx_l + 4
        pos_units.append(data[idx_l: idx_r])
        idx_l = idx_r
        idx_r = idx_l + 65
        pos_actions.append((data[idx_l: idx_r]))
        idx = idx_r
        pos_action_degree.append(data[idx])
    input_units = []
    actions_prob = []
    data_len = len(data)
    idx += 1
    debug = 0
    if debug == 1:
        if idx != 4 + height * width + 18 * 7 + 70 * width * height:
            print("error! idx")
        else:
            print("check spatial features correct")
    input_units_size = 0
    player = -1
    while True:
        idx_l = idx
        idx_r = idx + 6
        input_units.append(data[idx_l: idx_r])
        idx_l = idx + 6
        idx_r = idx + 6 + 65
        actions_prob.append(data[idx_l: idx_r])
        idx = idx_r
        input_units_size += 1
        if idx == data_len - 1:  # index of player
            player = data[-1]
            if debug == 1:
                print("all is correct")
            break
    # print(data)
    # 1. spatial_features
    # feature 1: player to play
    if player == 0:
        channel_player = np.zeros((1, height, width))
    else:
        channel_player = np.ones((1, height, width))
    # feature 2：walkable
    channel_walkable = np.array(walkable).reshape((1, height, width))
    channel_action_degree = np.array(pos_action_degree).reshape((1, height, width))
    # feature 3：player 0's units
    # compare two float values may need to recheck
    channel_unit_types = np.zeros((7, height, width))
    channel_unit_hp_ratio = np.zeros((6, height, width))
    channel_unit_resources = np.zeros((8, height, width))
    channel_player0_units = np.empty((1, height, width))
    channel_player1_units = np.empty((1, height, width))
    channel_player0_resources = np.zeros((8, height, width))
    channel_player1_resources = np.zeros((8, height, width))
    channel_player0_resources[encode_resources(p0res)][:][:] = 1
    channel_player1_resources[encode_resources(p1res)][:][:] = 1

    h, w = 0, 0
    nutt_features = 18
    hp_idx = 2
    for unit in pos_units:
        dim = encode_unit_type(unit[0])
        if dim >= 0:
            channel_unit_types[dim][h][w] = 1
        channel_player0_units[0][h][w] = 1 if int(unit[1]) == 0 or int(unit[0]) == 1 else 0
        channel_player1_units[0][h][w] = 1 if int(unit[1]) == 1 or int(unit[0]) == 1 else 0
        hp_ratio = 0.0 if unit[3] < 0 else unit[3] / (utt[int(unit[0]) * nutt_features + hp_idx])  # hp
        # no unit, should the value be set 1 or all zero?
        channel_unit_hp_ratio[encode_unit_hp_ratio(hp_ratio)][h][w] = 1
        channel_unit_resources[encode_resources(unit[2])][h][w] = 1
        if w + 1 == width:
            w = (w + 1) % width
            h += 1
        else:
            w += 1

    channel_unit_actions = np.zeros((65, height, width))
    # if no op -1
    # else action_id,
    h, w = 0, 0
    for action in pos_actions:
        pos_dim = find_id(action)
        if pos_dim != -1:
            channel_unit_actions[pos_dim][h][w] = 1
        if w + 1 == width:
            w = (w + 1) % width
            h += 1
        else:
            w += 1

    spatial_features = np.vstack(  # vstack axis=0
        (channel_player,  # 1
         channel_walkable,  # 1
         channel_player0_units,  # 1
         channel_player0_resources,  # 8
         channel_player1_units,  # 1
         channel_player1_resources,  # 8
         channel_unit_types,  # 7
         channel_unit_hp_ratio,  # 6
         channel_unit_resources,  # 8
         channel_unit_actions,  # 65
         channel_action_degree)  # 1
    )  # 107 channels, 11 kinds features
    # 2. utts_features
    utypes = 7
    utt_features = []
    for i in range(utypes):
        one_hot_unit_types = np.zeros(7)
        one_hot_unit_types[i] = 1
        utt_other_features = np.array(utt[i * nutt_features + 1: (i + 1) * nutt_features])
        utt_features = np.hstack(
            (
                utt_features,
                one_hot_unit_types,
                utt_other_features
            )
        )

    # 3. input_units shape
    ninput_unit_features = 17
    input_units_features = np.empty((input_units_size, ninput_unit_features))
    for i in range(input_units_size):
        input_unit_type = np.zeros(7)
        input_unit_type[encode_unit_type(input_units[i][0])] = 1
        input_unit_xyor = np.array([input_units[i][1] / width,
                                    input_units[i][2] / height,
                                    input_units[i][3], input_units[i][4]])  # ! [1 : 5)
        # input_unit_xyor[0] = input_unit_xyor[0] / width
        # input_unit_xyor[1] = input_unit_xyor[1] / height
        # input_unit_x = input_units[i][1]
        # input_unit_y = input_units[i][2]
        # input_unit_owner = input_units[i][3]
        # input_unit_resource = input_units[i][4]
        hp_ratio = input_units[i][5] / (utt[int(input_units[i][0]) * nutt_features + hp_idx])  # hp
        input_unit_hp_ratio = np.zeros(6)
        input_unit_hp_ratio[encode_unit_hp_ratio(hp_ratio)] = 1
        input_units_features[i] = np.hstack(
            (
                input_unit_type,
                input_unit_xyor,
                input_unit_hp_ratio
            )
        )

    # 4. action_prob
    # TODO: set temp to increase exploration
    # if tau == 0:
    #
    # else:
    #
    nactions_features = 65
    units_actions_labels = np.empty((input_units_size, nactions_features))
    for i in range(input_units_size):
        probs = [0.0] * nactions_features
        probs[np.argmax(actions_prob[i])] = 1.0
        units_actions_labels[i] = np.array(probs)

    return spatial_features, utt_features, input_units_features, units_actions_labels, input_units_size, player


class ToTensor(object):
    """convery ndarrays in sample to Tensors"""

    def __call__(self, sample):
        state, utt, units, actions, rewards, counts = sample['state'], sample['utt'], sample['units'], \
                                                      sample['actions'], sample['rewards'], sample['counts']
        return {'state': torch.from_numpy(state),
                'utt': torch.from_numpy(utt),
                'units': torch.from_numpy(units),
                'actions': torch.from_numpy(actions),
                'rewards': torch.from_numpy(rewards),
                'counts': torch.from_numpy(counts)}


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


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
    dataset = MicroRTSData('/Users/analysis.jinger/Repository/uRTS/rubensolv_MicroRTS/MicroRTS/uRTSMap16_SttvsMix',
                              '/Users/analysis.jinger/Repository/uRTS/rubensolv_MicroRTS/MicroRTS/Map16_SttvsMixWinner.csv')
    train_dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for states, utts, train_units, target_actions, rewards, length in train_dataloader:
        print(train_units, target_actions, length)



if __name__ == "__main__":
    # main()
    testDataLoader()
