import jittor as jt
import math
import jittor.nn as nn


def _get_optim(children_module_names, name_list, module_list):
    if len(name_list) == 0:
        return module_list
    pos = 0
    for idx, name in enumerate(children_module_names):
        same = True
        if pos >= len(name_list):
            same = False
        for i in range(min(len(name), len(name_list[pos]))):
            if name[i] != name_list[pos][i]:
                same = False
                break
        if same:
            module_list.append(idx)
            name_list.pop(pos)
            pos = 0
        else:
            pos += 1
        if len(name_list) == 0:
            break
    return module_list


def split_params(model, opt_cfg):
    module_list = []
    name_list = opt_cfg.model_type_list.copy()
    children_module_names = []
    for name, child in model.named_modules():
        if name == "":
            continue
        children_module_names.append(name)
    module_list = _get_optim(children_module_names, name_list, module_list)
    normal_bias = []
    normal_weight = []
    special_bias = []
    special_weight = []
    special_module = []
    for i, module in enumerate(model.modules()):
        if i in module_list:
            for name, param in module.named_parameters():
                if name == "":
                    continue
                if name.find("bias") != -1:
                    special_bias.append(param)
                else:
                    special_weight.append(param)
            special_module.append(module)
    for name, param in model.named_parameters():
        for special_m in special_module:
            flag = False
            for special_p in special_m.parameters():
                if param is special_p:
                    flag = True
                    break
            if flag:
                break
        else:
            if name.find("bias") != -1:
                normal_bias.append(param)
            else:
                normal_weight.append(param)
    return special_weight, special_bias, normal_weight, normal_bias


def split_optimizer(model, opt_cfg):
    special_weight, special_bias, normal_weight, normal_bias = split_params(model, opt_cfg)
    params = [
        {"params": normal_weight, "lr": opt_cfg.normal_lr, "weight_decay": opt_cfg.normal_weight_decay},
        {"params": normal_bias, "lr": opt_cfg.normal_lr, "weight_decay": opt_cfg.normal_bias_decay},
        {"params": special_weight, "lr": opt_cfg.special_lr, "weight_decay": opt_cfg.special_weight_decay},
        {"params": special_bias, "lr": opt_cfg.special_lr, "weight_decay": opt_cfg.special_bias_decay},
    ]
    if opt_cfg.optim_type == "SGD":
        optimizer = jt.optim.SGD(params=params, momentum=opt_cfg.momentum)
    elif opt_cfg.optim_type == "Adam":
        optimizer = jt.optim.Adam(params=params)
    else:
        raise NotImplementedError("optimizer {} not implemented".format(opt_cfg.optim_type))
    return optimizer


class IterWarmUpCosineDecayMultiStepLRAdjust(object):
    def __init__(self, init_lr, epochs, iter_per_epoch, warmup_epoch=0, multi_step=None, multi_gamma=0.1):
        self.init_lr = init_lr
        self.epochs = epochs
        self.iter_per_epoch = iter_per_epoch
        self.warmup_iter = iter_per_epoch * warmup_epoch
        self.total_iter = iter_per_epoch * epochs
        self.multi_step = multi_step
        self.multi_gamma = multi_gamma

    def adjust_lr(self, optimizer, iter_count):
        if iter_count < self.warmup_iter:
            cur_lr = self.init_lr * (iter_count / self.warmup_iter)
        else:
            if self.multi_step is not None:
                cur_lr = self.init_lr
                for step in self.multi_step:
                    if iter_count >= self.iter_per_epoch * step:
                        cur_lr = cur_lr * self.multi_gamma
                    else:
                        break
            else:
                # use cosine decay
                cur_lr = 0.5 * self.init_lr * (
                    1 + math.cos(math.pi * (iter_count - self.warmup_iter) / (self.total_iter - self.warmup_iter))
                )

        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr
        return cur_lr


class EpochWarmUpCosineDecayLRAdjust(object):
    def __init__(self, init_lr, epochs, warmup_epoch=0):
        self.init_lr = init_lr
        self.epochs = epochs
        self.warmup_epoch = warmup_epoch

    def adjust_lr(self, optimizer, epoch):
        if epoch < self.warmup_epoch:
            cur_lr = self.init_lr * ((epoch + 1) / self.warmup_epoch)
        else:
            # use cosine decay
            cur_lr = 0.5 * self.init_lr * (
                1 + math.cos(math.pi * (epoch - self.warmup_epoch) / (self.epochs - self.warmup_epoch))
            )

        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr
        return cur_lr 