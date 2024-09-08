from ..base import *
from ..finetune import FineTune


class EWC(FineTune):
    r"""
    [1]Kirkpatrick J, Pascanu R, Rabinowitz N, et al.
    Overcoming catastrophic forgetting in neural networks[J].
    Proceedings of the national academy of sciences, 2017, 114(13): 3521-3526.

    [2]Schwarz J, Czarnecki W, Luketina J, et al.
    Progress & compress: A scalable framework for continual learning[C]
    International Conference on Machine Learning. PMLR, 2018: 4528-4537.

    Args:
        lambd: the coefficient of EWC loss
        online: online EWC or not
        gamma: EMA coefficient of online EWC, memory strength of old tasks
    """

    def __init__(self, model, traindata, taskid):
        super(EWC, self).__init__(model, traindata, taskid)
        self.epsilon = 1e-32
        self.lambd = args.lambd

        self.online = args.online
        self.gamma = args.gamma

    def train(self):
        self.ewc_train()
        self.compute_fisher()
        return self.model

    def ewc_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt or 'nni' in args.opt)
        self.progress.set_description('train')
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                y_ = self.model(x)
                loss = self.criterion(y_, y, t)
                if self.taskid > 1:
                    ewc_loss = self.ewc_loss() * self.lambd
                    loss += ewc_loss
                else:
                    ewc_loss = 0
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item(),
                                           'ewcloss': ewc_loss.item() if self.taskid > 1 else 0})
                self.progress.update(1)
        self.progress.close()

    def compute_fisher(self):
        # compute empirical fisher matrix
        fisher = {}
        self.model.eval()
        # theoretically it should compute each sample's gradient,
        # here we compute each batch's gradient for saving time.
        progress = tqdm(self.trainloader, disable='slient' in args.opt or 'nni' in args.opt)
        progress.set_description('fisher')
        for x, y, t in progress:
            t = t + 1
            x = x.to(self.device)
            y = y.to(self.device)
            outputs = self.model(x)
            loss = self.nca_loss(outputs["logits"], y,
                                 scale=self.model.module.post_processor.factor
                                 if args.mode in ('DP', 'DDP') else self.model.post_processor.factor,
                                 margin=args.nca['margin'],
                                 exclude_pos_denominator=args.nca['exclude_pos_denominator'])
            self.model.zero_grad()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and (p.grad is not None):
                    if n not in fisher:
                        fisher[n] = p.grad.clone().detach() ** 2
                    else:
                        fisher[n] += p.grad.clone().detach() ** 2
        min_value = []
        max_value = []
        # expection
        for n, p in fisher.items():
            scaled = p / len(self.trainloader)
            fisher[n] = scaled
            min_value.append(torch.min(scaled).unsqueeze(dim=0))
            max_value.append(torch.max(scaled).unsqueeze(dim=0))
        min_value = torch.min(torch.cat(min_value))
        max_value = torch.max(torch.cat(max_value))
        # normalize
        for n, p in fisher.items():
            fisher[n] = (p - min_value) / (max_value - min_value + self.epsilon)
        param_ = {n: p.clone().detach() for n, p in self.model.named_parameters()}
        if hasattr(memory, 'fisher_param'):
            if self.online:
                for n, p in memory.fisher_param['fisher'].items():
                    memory.fisher_param['fisher'][n] = memory.fisher_param['fisher'][n] * self.gamma + \
                                                       fisher[n] * (1 - self.gamma)
                memory.fisher_param['param_'] = param_
            else:
                memory.fisher_param[self.taskid] = {'fisher': fisher, 'param_': param_}
        else:
            if self.online:
                memory.__setattr__('fisher_param', {'fisher': fisher, 'param_': param_})
            else:
                memory.__setattr__('fisher_param', {self.taskid: {'fisher': fisher, 'param_': param_}})
        self.model.train()

    def ewc_loss(self):
        loss = []
        if self.online:
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in memory.fisher_param['fisher']:
                    f = memory.fisher_param['fisher'][n]
                    p_ = memory.fisher_param['param_'][n]
                    loss.append((f * ((p - p_) ** 2)).sum())
            return sum(loss)
        else:
            for task in range(1, self.taskid):
                fisher = memory.fisher_param[task]['fisher']
                param_ = memory.fisher_param[task]['param_']
                for n, p in self.model.named_parameters():
                    if p.requires_grad and n in memory.fisher_param[task]['fisher']:
                        f = fisher[n]
                        p_ = param_[n]
                        loss.append((f * ((p - p_) ** 2)).sum())
            return sum(loss) / (self.taskid - 1)
