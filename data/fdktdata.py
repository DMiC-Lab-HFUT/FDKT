import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from config import DefaultConfig

if DefaultConfig.use_gpu:
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)


def takeFirst(elem):
    return elem[0]


def split_by_maxlen(list, n):
    l = []
    for i in list:
        splited = split_list_by_n(i, n)
        for s in splited:
            l.append(s)
    return l


def split_list_by_n(list_collection, n):
    for i in range(0, len(list_collection), n):
        yield list_collection[i:i+n]


class FDKTData(Dataset):
    def __init__(self, path, kc_dict, isTrain):
        self.path = path
        self.isTrain = isTrain
        self.kc_dict = kc_dict
        if 'assist' in self.path:
            self.sep = '\t'
        else:
            self.sep = '\t'
        if self.isTrain:
            if 'assist2009' in path:
                self.data_path = path + '/training.csv'
            else:
                self.data_path = path + '/training.txt'
            if DefaultConfig.code_input:
                self.input_skills, self.input_scores, self.data_length, self.max_len, self.len, self.input, self.flag = self.generate_coded_data()
            else:
                self.input_skills, self.input_scores, self.data_length, self.max_len, self.len, self.input = self.generate_coded_data()
        else:
            if 'assist2009' in path:
                self.data_path = path + '/testing.csv'
            else:
                self.data_path = path + '/testing.txt'

            if DefaultConfig.code_input:
                self.input_skills, self.input_scores, self.data_length, self.max_len, self.len, self.input, self.flag = self.generate_coded_data()
            else:
                self.input_skills, self.input_scores, self.data_length, self.max_len, self.len, self.input = self.generate_coded_data()

    def __getitem__(self, item):

        if DefaultConfig.code_input:
            return self.input[:, item, :], self.input_skills[:, item], self.input_scores[:, item], self.data_length[
                item]
        else:
            return self.input[:, item], self.input_skills[:, item], self.data_length[item]

    def __len__(self):
        return self.len

    def generate_coded_data(self):
        input_skills, input_scores, data_length, max_len = self.get_data()
        len_ = input_scores.size()[1]
        if DefaultConfig.code_input:
            input_, flag = self.data_coding(input_skills, input_scores)
            return input_skills, input_scores, data_length, max_len, len_, input_, flag
        else:
            input_ = input_scores
            return input_skills, input_scores, data_length, max_len, len_, input_

    def data_coding(self, input_skills, input_scores):
        maxlen, size = input_scores.size()
        kc_number = len(self.kc_dict)
        data_input = torch.zeros(maxlen, size, kc_number)
        data_flag = torch.zeros(maxlen, size, kc_number)
        for t in range(maxlen):
            skill_no = input_skills[t, :]
            for b in range(size):
                data_input[t, b, skill_no[b]] = input_scores[t, b]
                data_flag[t, b, skill_no[b]] = 1
        return data_input, data_flag

    def get_data(self):

        with open(self.data_path) as file_object:
            lines = file_object.readlines()
        skills = []
        scores = []
        data_length = []
        data = []
        if DefaultConfig.small_data:
            num = round((len(lines) // 3) / DefaultConfig.small_ratio)
        else:
            num = len(lines) // 3
        for i in range(num):
            each_line1 = lines[i * 3 + 1].rstrip().rstrip('\n').rstrip(',').split(self.sep)
            each_line1 = np.array([self.kc_dict[int(e)] for e in each_line1])
            each_line2 = lines[i * 3 + 2].rstrip().rstrip('\n').rstrip(',').split(self.sep)
            each_line2 = np.array([float(e) for e in each_line2])
            each = [len(each_line1), each_line1, each_line2]
            data.append(each)
            data_length.append(len(each_line1))
            skills.append(each_line1)
            scores.append(each_line2)
        sorted(data, key=takeFirst, reverse=False)
        # todo：按照时间步切割
        input_skills = [torch.from_numpy(each[1]) for each in data]
        input_scores = [torch.from_numpy(each[2]) for each in data]
        # data_length = [each[0] for each in data]
        max_len = DefaultConfig.max_len
        input_skills = split_by_maxlen(input_skills, max_len)
        input_scores = split_by_maxlen(input_scores, max_len)
        data_length = [len(each) for each in input_scores]
        input_skills = pad_sequence(input_skills)
        input_scores = pad_sequence(input_scores)

        # data_length = split_by_maxlen(data_length, max_len)


        # input_skills = pad_sequence([torch.from_numpy(each[1]) for each in data])
        # input_scores = pad_sequence([torch.from_numpy(each[2]) for each in data])
        # data_length = [each[0] for each in data]
        max_len = max(data_length)
        return input_skills, input_scores, data_length, max_len

