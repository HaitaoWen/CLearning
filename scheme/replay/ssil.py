from scheme.base import *
from scheme.replay.er import ER
from tool import *
from copy import deepcopy
import torch.nn.functional as F
from scheme.regularization.gpm.dataset.dataset import UnifiedDataset


class SSIL(ER):
    """
    Ahn H, Kwak J, Lim S, et al.
    Ss-il: Separated softmax for incremental learning[C]
    //Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 844-853.
    """
    def __init__(self, model, traindata, taskid):
        if args.dataset in ('PMNIST', 'FIVE') and hasattr(args, 'datafun'):
            traindata = UnifiedDataset(traindata['train']['x'], traindata['train']['y'],
                                       [traindata['t']-1] * len(traindata['train']['x']),
                                       dataset=args.dataset)
        super(ER, self).__init__(model, traindata, taskid)
        assert args.mcat
        assert args.mbs > 0
        assert args.tau > 0

        if self.taskid == 1:
            return
        if args.resume is not None:
            id1 = args.resume.rfind('/task')
            id2 = args.resume.rfind('.pkl')
            taskid_ = eval(args.resume[id1 + 5:id2])
            if taskid < taskid_ + 1:
                return

        if args.dataset in ('CIFAR10', 'CIFAR100', 'SVHN'):
            dataset = CIFARData(self.traindata, memory)
        elif 'ImageNet' in args.dataset:
            x = traindata._x.tolist() + memory.x
            y = traindata._y.tolist() + memory.y
            t = traindata._t.tolist() + memory.t
            dataset = ImageNetData(x, y, t, traindata.trsf)
        elif 'MNIST' in args.dataset:
            dataset = []
            trans = self.traindata.trsf
            if isinstance(trans, list):
                # This is due to dirty implementation of the new version continuum
                trans = trans[self.taskid - 1]
            _x, _y, _t = self.traindata._x, self.traindata._y, self.traindata._t
            for x, y, t in zip(_x, _y, _t):
                if trans is not None:
                    x = Image.fromarray(x.astype("uint8"))
                    x = trans(x).squeeze()
                dataset.append([x, y, int(t)])
            dataset.extend(memory.x)
            dataset = MNISTData(dataset)
        else:
            raise ValueError('Unsupported dataset: {}'.format(args.dataset))

        cur_data_num = len(self.traindata)
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=[i for i in range(cur_data_num)])
        batch_sampler = MemoryBatchSampler_extra(sampler, batch_size=args.bs, mem_size=len(memory.x),
                                                 mem_batch_size=args.mbs, drop_last=False)
        self.trainloader = DataLoader(dataset, batch_sampler=batch_sampler,
                                      num_workers=args.workers, pin_memory=args.pin_memory)
        if not hasattr(memory, 'iteration'):
            memory.__setattr__('iteration', 0)
        self.pre_minclass, self.pre_maxclass = get_minmax_class(self.taskid - 1)

    def train(self):
        self.ssil_train()
        self.construct_exemplar_set()
        pre_model = deepcopy(self.model)
        setattr(memory, 'pre_model', self.freeze_model(pre_model))
        return self.model

    def ssil_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt)
        self.progress.set_description('train')
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                y_ = self.model(x)
                if self.taskid == 1:
                    loss = self.criterion(y_, y, t)
                else:
                    mask = t == self.taskid
                    loss_ce_pre = F.cross_entropy(y_[~mask, self.pre_minclass: self.pre_maxclass+1],
                                                  y[~mask] - self.pre_minclass, reduction='sum')
                    loss_ce_cur = F.cross_entropy(y_[mask, self.pre_maxclass+1: self.maxclass+1],
                                                  y[mask] - (self.pre_maxclass + 1), reduction='sum')
                    loss = (loss_ce_pre + loss_ce_cur) / (args.bs + args.mbs)

                    loss_div = 0
                    y_pre = memory.pre_model(x).detach()
                    for taskid in range(1, self.taskid):
                        minclass, maxclass = get_minmax_class(taskid)
                        pre_minclass, pre_maxclass = get_minmax_class(taskid - 1)
                        y_log_ = F.log_softmax(y_[:, pre_maxclass+1: maxclass+1] / args.tau, dim=1)
                        pre_output = F.softmax(y_pre[:, pre_maxclass+1: maxclass+1] / args.tau, dim=1)
                        loss_div += F.kl_div(y_log_, pre_output, reduction='batchmean') * args.lambd

                    TBWriter.add_scalar('Loss_CE', loss.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_KL', loss_div.detach().cpu().data.numpy(), memory.iteration)
                    memory.iteration += 1

                    loss = loss + loss_div
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item()})
                self.progress.update(1)
            if self.scheduler is not None:
                self.scheduler.step()
        self.progress.close()

    def evaluate(self, scenario):
        time_memory_snapshot('Task{} [end]:'.format(self.taskid), device=self.device)
        self.model.eval()
        time.sleep(1)
        if args.scheme == 'MultiTask':
            taskid = args.tasks
        else:
            taskid = self.taskid
        progress = tqdm(range(1, taskid + 1), disable='slient' in args.opt)
        progress.set_description('eval ')
        for t in progress:
            evaldata = scenario[t - 1]
            if args.dataset == 'PMNIST' and hasattr(args, 'datafun'):
                evaldata = UnifiedDataset(evaldata['test']['x'], evaldata['test']['y'],
                                           [evaldata['t']] * len(evaldata['test']['x']),
                                           dataset=args.dataset)
            if args.mode == 'DDP':
                sampler = utils.data.DistributedSampler(evaldata, shuffle=False)
            else:
                sampler = None
            evalloader = DataLoader(evaldata, args.bs, shuffle=False, num_workers=args.workers, sampler=sampler)
            if args.scenario == 'class':
                minclass, maxclass = get_minmax_class(taskid)
            else:
                minclass, maxclass = get_minmax_class(t)
            targets, predicts = [], []
            with torch.no_grad():
                for x, y, _ in evalloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    y_ = self.model(x)
                    y_, _ = activate_head(minclass, maxclass, y_, y)
                    y_ = y_.topk(k=1, dim=1)[1]
                    y_ = remap_label(y_, minclass)
                    targets.append(y)
                    predicts.append(y_)
            targets = torch.cat(targets, dim=0)
            predicts = torch.cat(predicts, dim=0)
            if args.mode == 'DDP':
                torch.distributed.barrier()
                targets = distributed_concat(targets)
                predicts = distributed_concat(predicts)
            targets = targets.unsqueeze(dim=1)
            top1_acc = targets.eq(predicts[:, :1]).sum().item() / targets.shape[0]
            memory.AccMatrix[taskid - 1, t - 1] = top1_acc
            memory.task_size[t] = targets.shape[0]
        progress.close()
        if args.snapshot:
            save_memory(memory, self.taskid)
            save_model(self.model, self.taskid)
        print_log_metrics(memory.AccMatrix, taskid)
        time.sleep(1)
