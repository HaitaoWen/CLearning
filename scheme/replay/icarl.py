from scheme.replay.er import ER
from scheme.base import *
from copy import deepcopy
import torch.nn.functional as F


class iCaRL(ER):
    def __init__(self, model, traindata, taskid):
        super(iCaRL, self).__init__(model, traindata, taskid)
        # assert args.tau > 0, 'Temperature should be larger than 0.'
        assert args.memory > 0
        assert args.scenario == 'class'
        self.nme = None
        if self.taskid > 1:
            self.pre_minclass, self.pre_maxclass = get_minmax_class(self.taskid - 1)
        if not hasattr(memory, 'iteration'):
            memory.__setattr__('iteration', 0)

    def train(self):
        self.icarl_train()
        self.construct_exemplar_set()
        self.construct_nme_classifier()
        pre_model = deepcopy(self.model)
        setattr(memory, 'pre_model', self.freeze_model(pre_model))
        return self.model

    def icarl_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt)
        self.progress.set_description('train')
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                y_ = self.model(x)
                loss = self.criterion(y_, y, t)
                if self.taskid > 1:
                    y_log_ = F.log_softmax(y_[:, self.pre_minclass: self.pre_maxclass + 1] / args.tau, dim=1)
                    pre_output = F.softmax(memory.pre_model(x).detach()[:, self.pre_minclass: self.pre_maxclass + 1]
                                           / args.tau, dim=1)
                    loss_div = F.kl_div(y_log_, pre_output, reduction='batchmean')

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

    def construct_nme_classifier(self):
        self.model.eval()
        hook_handle = self.register_hook(module_name='fc', variable_name='representations',
                                         call_back=self.hook_abstract_call_back, side='input')
        if args.mode in ('DP', 'DDP'):
            self.nme = torch.nn.Linear(self.model.module.fc.in_features, len(np.unique(memory.y)),
                                       bias=False, device=self.device)
        else:
            self.nme = torch.nn.Linear(self.model.fc.in_features, len(np.unique(memory.y)), bias=False,
                                       device=self.device)
        trsf = self.traindata.trsf
        if args.dataset == 'CIFAR100':
            memdata = Memorydata(memory, trsf)
        elif 'ImageNet' in args.dataset:
            memdata = ImageNetData(memory.x, memory.y, memory.t, trsf)
        else:
            raise ValueError
        if args.mode == 'DDP':
            sampler = utils.data.DistributedSampler(memdata, shuffle=False)
            loader = DataLoader(memdata, args.bs, num_workers=args.workers, sampler=sampler)
        else:
            loader = DataLoader(memdata, args.bs, num_workers=args.workers, shuffle=False)
        labels = []
        representations = []
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.model(x)
                labels.append(y)
                representations.append(self.get_hook_result(variable_name='representations'))
        labels = torch.cat(labels, dim=0)
        representations = torch.cat(representations, dim=0)
        if args.mode == 'DDP':
            torch.distributed.barrier()
            labels = distributed_concat(labels)
            representations = distributed_concat(representations)
        representations = torch.nn.functional.normalize(representations, p=2, dim=1)
        for label in labels.unique():
            mask = labels == label
            representation = representations[mask]
            representation_mean = representation.mean(dim=0)
            representation_mean = torch.nn.functional.normalize(representation_mean, p=2, dim=0)
            self.nme.weight.data[label] = representation_mean
        self.destroy_hook(hooks=hook_handle)

    def evaluate(self, scenario):
        if hasattr(memory, 'AccMatrix_NME') is False:
            memory.AccMatrix_NME = np.zeros((args.tasks, args.tasks))
        self.model.eval()
        self.register_hook(module_name='fc', variable_name='representations',
                           call_back=self.hook_abstract_call_back, side='input')
        time.sleep(1)
        taskid = self.taskid
        progress = tqdm(range(1, taskid + 1), disable='slient' in args.opt)
        progress.set_description('eval ')
        for t in progress:
            evaldata = scenario[t - 1]
            if args.mode == 'DDP':
                sampler = utils.data.DistributedSampler(evaldata, shuffle=False)
            else:
                sampler = None
            evalloader = DataLoader(evaldata, args.bs, shuffle=False, num_workers=args.workers, sampler=sampler)
            if args.scenario == 'class':
                minclass, maxclass = get_minmax_class(taskid)
            else:
                minclass, maxclass = get_minmax_class(t)
            targets, cnn_predicts, nme_predicts = [], [], []
            with torch.no_grad():
                for x, y, _ in evalloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    y_cnn = self.model(x)
                    y_cnn, _ = activate_head(minclass, maxclass, y_cnn, y)
                    y_cnn = y_cnn.topk(k=1, dim=1)[1]
                    y_cnn = remap_label(y_cnn, minclass)
                    # NME classifier
                    representation = self.get_hook_result(variable_name='representations')
                    representation = torch.nn.functional.normalize(representation, p=2, dim=1)
                    y_nme = self.nme(representation)
                    y_nme = y_nme.topk(k=1, dim=1)[1]
                    y_nme = remap_label(y_nme, minclass)
                    targets.append(y)
                    cnn_predicts.append(y_cnn)
                    nme_predicts.append(y_nme)
            targets = torch.cat(targets, dim=0)
            cnn_predicts = torch.cat(cnn_predicts, dim=0)
            nme_predicts = torch.cat(nme_predicts, dim=0)
            if args.mode == 'DDP':
                torch.distributed.barrier()
                targets = distributed_concat(targets)
                cnn_predicts = distributed_concat(cnn_predicts)
                nme_predicts = distributed_concat(nme_predicts)
            targets = targets.unsqueeze(dim=1)
            cnn_top1_acc = targets.eq(cnn_predicts[:, :1]).sum().item() / targets.shape[0]
            nme_top1_acc = targets.eq(nme_predicts[:, :1]).sum().item() / targets.shape[0]
            memory.AccMatrix[taskid - 1, t - 1] = cnn_top1_acc
            memory.AccMatrix_NME[taskid - 1, t - 1] = nme_top1_acc
            # It is important to record sizes of tasks to compute AIAcc.
            memory.task_size[t] = targets.shape[0]
        progress.close()
        self.destroy_hook()
        if args.snapshot:
            save_memory(memory, self.taskid)
            save_model(self.model, self.taskid)
        print_log_metrics(memory.AccMatrix, taskid, remarks='CNN')
        print_log_metrics(memory.AccMatrix_NME, taskid, remarks='NME')
        time.sleep(1)
