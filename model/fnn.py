import torch
from torch import nn
import time
from config import DefaultConfig
from tqdm import tqdm


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.linspace(0, 1, steps=self.in_features).unsqueeze(dim=1).expand(self.in_features,
                                                                                 self.out_features).clone())
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        x = x.mm(self.weight)
        if self.bias:
            return x + self.bias.expand_as(x)
        else:
            return x


class FuzzyLayer(nn.Module):
    def __init__(self, batch_size, max_len, kc_numbers, term_numbers):
        super(FuzzyLayer, self).__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.kc_numbers = kc_numbers
        self.term_numbers = term_numbers

        self.mean = torch.nn.Parameter(self.set_mean(), requires_grad=True)
        self.sigma = torch.nn.Parameter(self.set_sigma(), requires_grad=True)

    def forward(self, input_data):
        input_expand = input_data.unsqueeze(dim=2).expand(self.batch_size, self.max_len, self.term_numbers)
        output = torch.exp(-((input_expand - self.mean).pow(2) / (self.sigma.pow(2))))
        return output

    def set_mean(self):
        return torch.linspace(0, 1, steps=self.term_numbers)

    def set_sigma(self):
        return torch.rand(self.term_numbers)


class FNN(nn.Module):
    def __init__(self, term_numbers, cog_numbers, rule_numbers, kc_numbers, kc_dict, arg, batch_size):
        super(FNN, self).__init__()
        self.kc_numbers = kc_numbers
        self.term_numbers = term_numbers
        self.cog_numbers = cog_numbers
        self.rule_numbers = rule_numbers
        self.kc_dict = kc_dict
        self.arg = arg
        self.mean = torch.nn.Parameter(self.set_mean(), requires_grad=True)
        self.sigma = torch.nn.Parameter(self.set_sigma(), requires_grad=True)
        self.theta = torch.nn.Parameter(self.set_theta(), requires_grad=True)
        self.cognition_0 = torch.nn.Parameter(self.set_cognition_0(), requires_grad=True)


        self.layer_predict = torch.nn.Linear(in_features=self.cog_numbers, out_features=1, bias=True)
        self.layer_cognition = torch.nn.Linear(in_features=self.term_numbers * self.cog_numbers,
                                               out_features=self.cog_numbers,
                                               bias=False)
        if torch.cuda.device_count():
            self.batch_size = int(batch_size / torch.cuda.device_count())
        else:
            self.batch_size = batch_size

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = DefaultConfig.model_dir
            name = time.strftime(prefix + str(self.arg[0]) + '_' + str(self.arg[1]) + '.pth')
        torch.save(self.state_dict(), name)
        return name

    def get_fuzzy_score(self, step_input):
        size_1, size_2 = step_input.size()
        out1_extend = step_input.unsqueeze(dim=2).expand(size_1, size_2, self.term_numbers)
        out2 = torch.exp(-((out1_extend - self.mean).pow(2) / (self.sigma.pow(2))))
        return out2

    def get_fuzzy_rule(self, cognition_mem, step_skill, fuzzy_score):

        fuzzy_rule = fuzzy_score.clone().unsqueeze(dim=2).expand(self.batch_size * self.kc_numbers, self.term_numbers,
                                                                 self.cog_numbers).mul(
            cognition_mem.clone().unsqueeze(dim=2).expand(self.batch_size * self.kc_numbers, self.cog_numbers,
                                                           self.term_numbers).transpose(1, 2).contiguous()).reshape(
            self.batch_size * self.kc_numbers, self.term_numbers * self.cog_numbers)
        return fuzzy_rule

    def update_cognition_mem(self, cognition_mem, cognitive_norm, step_skill):
        for b in range(self.batch_size):
            cognition_mem[b, step_skill[b], :] = cognitive_norm[b]
        return cognition_mem

    def forward(self, T, input_scores, input_skills, isTrain):
        if isTrain:
            if torch.cuda.device_count():
                self.batch_size = int(DefaultConfig.train_batch_size / torch.cuda.device_count())
            else:
                self.batch_size = DefaultConfig.train_batch_size
        else:
            if torch.cuda.device_count():
                self.batch_size = int(DefaultConfig.test_batch_size / torch.cuda.device_count())
            else:
                self.batch_size = DefaultConfig.test_batch_size
        pred_t = torch.zeros(self.batch_size, T)
        pred_tplus1 = torch.zeros(self.batch_size, T)
        target_t = torch.zeros(self.batch_size, T)
        target_tplus1 = torch.zeros(self.batch_size, T)
        cog = torch.zeros(self.batch_size, T, self.cog_numbers)
        cognition_mem = self.cognition_0.unsqueeze(dim=0).expand(self.batch_size, self.kc_numbers, self.cog_numbers).reshape(self.batch_size * self.kc_numbers, self.cog_numbers)


        for t in (range(T)):
            from torch.nn import functional
            input_mask = functional.one_hot(input_skills[:, t].clone().detach().to(torch.int64), num_classes=self.kc_numbers)
            input_ = input_scores[:, t].unsqueeze(1).expand(self.batch_size, self.kc_numbers).mul(input_mask)
            fuzzy_score = self.get_fuzzy_score(input_).reshape(self.batch_size*self.kc_numbers,self.term_numbers)
            fuzzy_rule = self.get_fuzzy_rule(cognition_mem, input_skills[:, t], fuzzy_score).reshape(self.batch_size*self.kc_numbers,self.term_numbers*self.cog_numbers)

            # cognition_now = torch.nn.Sigmoid()(self.layer_cognition(fuzzy_rule))
            cognition_now = torch.nn.Sigmoid()(self.layer_cognition(fuzzy_rule))

            cognitive_norm = cognition_now / torch.sum(cognition_now, dim=1).unsqueeze(dim=1)
            cog_mask = input_mask.reshape(self.batch_size * self.kc_numbers).unsqueeze(1).expand(self.batch_size * self.kc_numbers, self.cog_numbers).bool()
            cognition_mem = cognition_mem.clone().mul(~cog_mask) + cognitive_norm.clone().mul(cog_mask)

            predict_score = torch.nn.Sigmoid()(self.layer_predict(cognition_mem)).reshape(self.batch_size, self.kc_numbers)

            pred_mask_t = input_mask
            pred_mask_tplus1 = functional.one_hot(input_skills[:, t + 1].clone().detach().to(torch.int64), num_classes=self.kc_numbers)
            pred_t[:, t] = predict_score.mul(pred_mask_t).sum(dim=1)
            pred_tplus1[:, t] = predict_score.mul(pred_mask_tplus1).sum(dim=1)


            target_t[:, t] = input_.sum(dim=1)
            target_tplus1[:, t] = input_scores[:, t + 1].unsqueeze(1).expand(self.batch_size, self.kc_numbers).mul(pred_mask_tplus1).sum(dim=1)


        return pred_t, pred_tplus1, target_t, target_tplus1

    def set_mean(self):
        return torch.linspace(0, 1, steps=self.term_numbers)

    def set_sigma(self):
        return torch.rand(self.term_numbers)

    def set_theta(self):
        return torch.rand(self.kc_numbers, self.rule_numbers)

    def set_cognition_0(self):
        cognition_m = torch.ones(self.kc_numbers, self.cog_numbers)
        cog_mem_sum = torch.sum(cognition_m, dim=1).unsqueeze(dim=1)
        cognition_mem = cognition_m / cog_mem_sum
        return cognition_mem
