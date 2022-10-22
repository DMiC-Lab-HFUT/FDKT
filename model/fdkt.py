import logging
import os
import joblib
import torch
import torch.optim as optimization
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchnet import meter
from tqdm import tqdm
from config import DefaultConfig
from data.data_deal import *
from data.fdktdata import FDKTData
from model.fnn import FNN
from utils import obtain_metrics
from utils import write_csv
from utils.makdir import mkdir
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s   %(levelname)s   %(message)s')

config = DefaultConfig()
Epoch = config.Epoch
batch_size = config.train_batch_size
term_numbers = config.term_numbers
cog_numbers = config.cog_numbers
rule_numbers = config.rule_numbers
weight_decay = config.weight_decay
learning_rate = config.learning_rate
our_dir = config.dir
print_freq = config.print_freq
result_dir = config.result_dir
model_dir = config.model_dir
training_prediction_dir = config.training_prediction_dir
testing_prediction_dir = config.testing_prediction_dir

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ids = [0, 1]

if config.use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device("cpu")


def step(fnn_model, input_, isTrain):
    global len_change, max_data_length
    input_scores, input_skills, data_length = Variable(input_[0]), Variable(input_[1]), Variable(input_[2])
    if config.use_gpu:
        max_data_length = torch.Tensor([max(data_length)] * torch.cuda.device_count())
        input_scores = input_scores.to(device)
        input_skills = input_skills.to(device)
        max_data_length = max_data_length.to(device)
    else:
        max_data_length = [max(data_length)]

    batch_target = input_scores[:, 1:]
    max_len = int(max(max_data_length))
    if isTrain:
        if max_len > config.train_len:
            len_change = True
            max_len = config.train_len
        else:
            len_change = False
        T = max_len - 1
    else:
        T = max_len - 1
        len_change = False

    batch_pred_t, batch_pred_tplus1, batch_target_t, batch_target_tplus1 = fnn_model(T, input_scores, input_skills, isTrain)

    dl_mask = data_length.gt(max_len)
    dl_max_len = torch.tensor([max_len]*len(data_length))
    dl = dl_max_len.mul(dl_mask) + data_length.mul(~dl_mask)

    total_steps = torch.sum(dl)


    pred_t, pred_tplus1, target_t, target_tplus1 = [], [], [], []
    norm_l1, norm_l2 = [], []
    for count, l in enumerate(dl):
        pred_t.extend(batch_pred_t[count, :l-1])
        target_t.extend(batch_target_t[count, :l - 1])
        pred_tplus1.extend(batch_pred_tplus1[count, :l-2])
        target_tplus1.extend(batch_target_tplus1[count, :l - 2])
        # todo: 计算pred_tplus1[count, 1:]与pred_tplus1[count, :-1]的差
        norm_l1.extend(torch.abs(batch_pred_tplus1[count, 1:] - batch_pred_tplus1[count, :-1]))
        norm_l2.extend(torch.square(batch_pred_tplus1[count, 1:] - batch_pred_tplus1[count, :-1]))

    p_t = torch.stack(pred_t)
    p_tplus1 = torch.stack(pred_tplus1)
    t_t = torch.stack(target_t)
    t_tplus1 = torch.stack(target_tplus1)
    norm_l1 = torch.stack(norm_l1)
    norm_l2 = torch.stack(norm_l2)

    return p_t, t_t, p_tplus1, t_tplus1, norm_l1, norm_l2, total_steps


def epoch_train(fnn_model, kc_number, train_loader, optimizer, loss_meter, kc_dict, train_max_len):
    target_t, pred_t, target_tplus1, pred_tplus1, = [], [], [], []
    for input_ in (train_loader):
        p_t, t_t, p_tplus1, t_tplus1,norm_l1, norm_l2, total_steps = step(fnn_model, input_, isTrain=True)
        pred_t.append(p_t.detach().cpu().numpy().tolist())
        target_t.append(t_t.detach().cpu().numpy().tolist())
        pred_tplus1.append(p_tplus1.detach().cpu().numpy().tolist())
        target_tplus1.append(t_tplus1.detach().cpu().numpy().tolist())
        # cog.append(c.detach().cpu().numpy().tolist())

        criterion = torch.nn.MSELoss(reduction='sum')

        if config.use_gpu:
            criterion.cuda()

        loss = criterion(p_tplus1, t_tplus1) + config.lambda_o * criterion(p_t, t_t)

        # regularization 1
        waviness_l1 = torch.sum(norm_l1)

        loss += config.lambda_w1 * waviness_l1
        # regularization 2
        waviness_l2 = torch.sum(norm_l2)

        loss += config.lambda_w2 * waviness_l2


        loss.requires_grad_(True)
        optimizer.zero_grad()

        torch.autograd.set_detect_anomaly(True)
        loss.backward(retain_graph=True)


        loss_meter.add(loss.item())

        optimizer.step()

    return target_tplus1, pred_tplus1


def epoch_predict(fnn_model, data_loader, max_len):
    target_t, pred_t, target_tplus1, pred_tplus1, = [], [], [], []
    for input_ in tqdm(data_loader):
        p_t, t_t, p_tplus1, t_tplus1,_,_,_ = step(fnn_model, input_, isTrain=False)
        pred_t.append(p_t.detach().cpu().numpy().tolist())
        target_t.append(t_t.detach().cpu().numpy().tolist())
        pred_tplus1.append(p_tplus1.detach().cpu().numpy().tolist())
        target_tplus1.append(t_tplus1.detach().cpu().numpy().tolist())

    return target_tplus1, pred_tplus1


def collate_fn(batch):
    batch = list(zip(*batch))
    input_scores, input_skills, data_length = batch[0], batch[1], batch[2]
    del batch
    return default_collate(input_scores), default_collate(input_skills), default_collate(data_length)


class FDKT:
    def __init__(self, arg):
        self.arg = arg
        self.path = our_dir + arg[0] + '/Deep_data/' + arg[1]
        self.tr_cc_rmse = [1] * 100
        self.tr_cc_mae = [1] * 100

    def ob_metrics(self, pred, act, epoch):
        rmse, mae, auc = obtain_metrics.obtain_metrics(c_actual=act, c_pred=pred)
        self.tr_cc_rmse[epoch] = rmse
        self.tr_cc_mae[epoch] = mae
        print(str(self.arg[0]) + str(self.arg[1]))
        print('Epoch' + str(epoch))
        return rmse, mae, auc

    def train_and_test(self, arg):
        global start_epoch
        kc_dict, kc_number, train_typeset, test_typeset = get_kc_set(self.path)
        threshold = 1e-5
        if DefaultConfig.code_input:
            if os.path.isfile(self.path + "/coded_train.model"):
                train_data = joblib.load(self.path + "/coded_train.model")
            else:
                train_data = FDKTData(self.path, kc_dict, isTrain=True)
                joblib.dump(filename=self.path + "/coded_train.model", value=train_data)

            if os.path.isfile(self.path + "/coded_test.model"):
                test_data = joblib.load(self.path + "/coded_test.model")
            else:
                test_data = FDKTData(self.path, kc_dict, isTrain=False)
                joblib.dump(filename=self.path + "/coded_test.model", value=test_data)
        else:
            train_data = FDKTData(self.path, kc_dict, isTrain=True)
            test_data = FDKTData(self.path, kc_dict, isTrain=False)

        train_loader = DataLoader(dataset=train_data, batch_size=config.train_batch_size, shuffle=False, drop_last=True,
                                  collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test_data, batch_size=config.test_batch_size, shuffle=False, drop_last=True,
                                 collate_fn=collate_fn)
        fnn_model = FNN(term_numbers, cog_numbers, rule_numbers, kc_number, kc_dict, arg, batch_size=batch_size)
        start_epoch = 0
        if config.use_gpu:
            if torch.cuda.device_count() > 1:
                print("Use", torch.cuda.device_count(), 'gpus')
                fnn_model = nn.DataParallel(fnn_model)
        fnn_model.to(device)
        loss_meter = meter.AverageValueMeter()
        optimizer = optimization.Adam(fnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # optimizer = optimization.Adam(fnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_max_len = train_data.max_len
        test_max_len = test_data.max_len
        best_epoch = 0

        for epoch in tqdm(range(start_epoch, Epoch)):
            loss_meter.reset()
            train_target, train_pred = epoch_train(fnn_model, kc_number, train_loader, optimizer, loss_meter,
                                                              kc_dict, train_max_len)
            print('Epoch ' + str(epoch) + ' training done.')

            actual = sum(train_target, [])
            pred = sum(train_pred, [])

            if 'bi' in self.path:
                rmse, mae, auc = self.ob_metrics(pred, actual, epoch)
                print('train auc: ' + str(auc))
                print('train rmse: ' + str(rmse))
                print('train mae: ' + str(mae))
            else:
                rmse, mae, _ = self.ob_metrics(pred, actual, epoch)
                print('train rmse: ' + str(rmse))
                print('train mae: ' + str(mae))


            test_target, test_pred = epoch_predict(fnn_model, test_loader, test_max_len)
            actual = sum(test_target, [])
            pred = sum(test_pred, [])
            if 'bi' in self.path:
                rmse, mae, auc = self.ob_metrics(pred, actual, epoch)
                print('test auc: ' + str(auc))
                print('test rmse: ' + str(rmse))
                print('test mae: ' + str(mae))
            else:
                rmse, mae, _ = self.ob_metrics(pred, actual, epoch)
                print('test rmse: ' + str(rmse))
                print('test mae: ' + str(mae))

