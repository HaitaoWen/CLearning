"""
Code is borrowed from: https://github.com/hshustc/CVPR19_Incremental_Learning.git
Modified by Haitao Wen
"""
import math
from scheme.base import *
from scheme.replay.icarl import iCaRL
from copy import deepcopy
import torch.nn.functional as F
from model.modified_linear import CosineLinear, SplitCosineLinear


class LUCIR(iCaRL):
    def __init__(self, model, traindata, taskid):
        super(LUCIR, self).__init__(model, traindata, taskid)
        self.deparallelization()
        if self.taskid == 1:
            in_size = self.model.fc.in_features
            out_size = traindata.nb_classes
            printlog('adjust CosineLinear, in:{}, out:{}'.format(in_size, out_size))
            self.model.fc = CosineLinear(in_size, out_size, sigma=True).to(self.device)
            self.allocate_model()
            self.init_optimizer()
        else:
            if self.taskid == 2:
                in_size = self.model.fc.in_features
                out_size1 = self.model.fc.out_features
                out_size2 = traindata.nb_classes
                printlog('adjust SplitCosineLinear, in:{}, out1:{}, out2:{}'.format(in_size, out_size1, out_size2))
                new_fc = SplitCosineLinear(in_size, out_size1, out_size2).to(self.device)
                new_fc.fc1.weight.data = self.model.fc.weight.data
                new_fc.sigma.data = self.model.fc.sigma.data
                self.model.fc = new_fc
                # adjust hyperparameters
                coef = out_size1 * 1.0 / traindata.nb_classes
            else:
                in_size = self.model.fc.in_features
                out_size1 = self.model.fc.fc1.out_features
                out_size2 = self.model.fc.fc2.out_features
                out_size3 = traindata.nb_classes
                printlog('adjust SplitCosineLinear, in:{}, out1:{}, out2:{}'.format(in_size, out_size1 + out_size2,
                                                                                    out_size3))
                new_fc = SplitCosineLinear(in_size, out_size1 + out_size2, out_size3).to(self.device)
                new_fc.fc1.weight.data[:out_size1] = self.model.fc.fc1.weight.data
                new_fc.fc1.weight.data[out_size1:] = self.model.fc.fc2.weight.data
                new_fc.sigma.data = self.model.fc.sigma.data
                self.model.fc = new_fc
                coef = (out_size1 + out_size2) * 1.0 / out_size3
            self.lamda = args.lamda * math.sqrt(coef)
            printlog('default lambda:{}, current lambda:{}'.format(args.lamda, self.lamda))
            # reparallelization
            self.allocate_model()
            # imprint weights
            self.initialize_new_weights()
            # reinit optimizer, weights of old tasks is frozen
            self.init_optimizer_custom()
            # register hook
            self.create_hook()
            # previous model parametres
            self.pre_params = self.detach_parameters()
        self.model.train()

    def train(self):
        self.lucir_train()
        self.construct_exemplar_set()
        self.cbf_finetune()
        self.construct_nme_classifier()
        pre_model = deepcopy(self.model)
        setattr(memory, 'pre_model', self.freeze_model(pre_model))
        return self.model

    def lucir_train(self):
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
                    memory.pre_model(x)

                    fc1_scores = self.get_hook_result(self.model, variable_name='fc1_scores')
                    fc2_scores = self.get_hook_result(self.model, variable_name='fc2_scores')
                    cur_representations = self.get_hook_result(self.model, variable_name='cur_representations')
                    pre_representations = self.get_hook_result(memory.pre_model, variable_name='pre_representations')

                    loss_embed = F.cosine_embedding_loss(cur_representations, pre_representations.detach(),
                                                         torch.ones(x.shape[0]).to(self.device)) * self.lamda
                    # scores before scale, [-1, 1]
                    outputs_bs = torch.cat((fc1_scores, fc2_scores), dim=1)
                    assert (outputs_bs.size() == y_.size())
                    # get ground truth scores
                    gt_index = torch.zeros(outputs_bs.size()).to(self.device)
                    gt_index = gt_index.scatter(1, y.view(-1, 1), 1).ge(0.5)
                    gt_scores = outputs_bs.masked_select(gt_index)
                    # get top-K scores on novel classes
                    max_novel_scores = outputs_bs[:, self.pre_maxclass + 1:].topk(args.K, dim=1)[0]
                    # the index of hard samples, i.e., samples of old classes
                    hard_index = y.lt(self.pre_maxclass + 1)
                    hard_num = torch.nonzero(hard_index).size(0)
                    if hard_num > 0:
                        gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, args.K)
                        max_novel_scores = max_novel_scores[hard_index]
                        assert (gt_scores.size() == max_novel_scores.size())
                        assert (gt_scores.size(0) == hard_num)
                        loss_mr = F.margin_ranking_loss(gt_scores.view(-1, 1), max_novel_scores.view(-1, 1),
                                                        torch.ones(hard_num * args.K, 1).to(self.device),
                                                        margin=args.dist) * args.lw_mr
                    else:
                        loss_mr = torch.zeros(1).to(self.device)

                    TBWriter.add_scalar('Loss_CE', loss.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_EB', loss_embed.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_MR', loss_mr.detach().cpu().data.numpy(), memory.iteration)
                    memory.iteration += 1

                    loss = loss + loss_embed + loss_mr

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
        self.destroy_hook()

    def cbf_finetune(self):
        if self.taskid == 1 or not hasattr(args, 'finetune'):
            return
        printlog('class-balanced finetuning', delay=1.0)
        # ****************** load memory data ****************** #
        trsf = self.traindata.trsf
        if args.dataset == 'CIFAR100':
            memdata = Memorydata(memory, trsf)
        elif 'ImageNet' in args.dataset:
            memdata = ImageNetData(memory.x, memory.y, memory.t, trsf)
        else:
            raise ValueError
        # *************** overwrite self.trainloader *************** #
        if args.mode == 'DDP':
            sampler = utils.data.DistributedSampler(memdata, shuffle=True)
            self.trainloader = DataLoader(memdata, args.bs, num_workers=args.workers, sampler=sampler,
                                          pin_memory=args.pin_memory)
        else:
            self.trainloader = DataLoader(memdata, args.bs, num_workers=args.workers, shuffle=True,
                                          pin_memory=args.pin_memory)
        # ******************** freeze batch normalization ******************** #
        self.model.train()
        for module in self.model.module.modules() \
                if args.mode in ('DP', 'DDP') else self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        # *********** overwrite self.optimizer and self.scheduler *********** #
        # both the old and novel embeddings are updated with the feature extractor fixed
        ignored_params = list(map(id, self.model.module.fc.parameters() if args.mode in ('DP', 'DDP')
                              else self.model.fc.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())
        params = [{'params': self.model.module.fc.fc1.parameters() if args.mode in ('DP', 'DDP')
                  else self.model.fc.fc1.parameters(), 'lr': args.finetune['lr'], 'weight_decay': args.decay},
                  {'params': self.model.module.fc.fc2.parameters() if args.mode in ('DP', 'DDP')
                  else self.model.fc.fc2.parameters(), 'lr': args.finetune['lr'], 'weight_decay': args.decay},
                  {'params': base_params, 'lr': 0, 'weight_decay': 0}]
        self.optimizer = torch.optim.SGD(params, lr=args.finetune['lr'], momentum=args.momentum,
                                         weight_decay=args.decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.finetune['steps'],
                                                              gamma=args.finetune['gamma'])
        # ************************** overwrite configs ************************** #
        args_copy = deepcopy(args)
        args.epochs = args.finetune['epochs']
        # ********************** multiplexing self.lucir_train ********************** #
        self.create_hook()
        self.lucir_train()
        self.destroy_hook()
        args.__dict__.update(args_copy.__dict__)

    def initialize_new_weights(self):
        if hasattr(args, 'init_new'):
            assert args.init_new in ('kaiming', 'random', 'zero', 'imprint')
            if args.mode in ('DP', 'DDP'):
                fc_new = self.model.module.fc.fc2
            else:
                fc_new = self.model.fc.fc2
            if args.init_new == 'kaiming':
                nn.init.kaiming_normal_(fc_new.weight.data, nonlinearity="relu", mode="fan_out")
            elif args.init_new == 'random':
                fc_new.weight.data = torch.randn(fc_new.weight.data.shape).to(fc_new.weight.data.device)
            elif args.init_new == 'zero':
                fc_new.weight.data.zero_()
            elif args.init_new == 'imprint':
                self.imprint_weights()
            else:
                self.imprint_weights()
        else:
            self.imprint_weights()

    def imprint_weights(self):
        printlog('imprint weights')
        self.model.eval()
        self.register_hook(module_name='fc', variable_name='representations',
                           call_back=self.hook_abstract_call_back, side='input')
        if args.mode in ('DP', 'DDP'):
            pre_weight_norm = self.model.module.fc.fc1.weight.data.norm(dim=1, keepdim=True)
        else:
            pre_weight_norm = self.model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
        pre_weight_norm_mean = torch.mean(pre_weight_norm, dim=0)
        init_weights = []
        classes = np.unique(self.traindata._y)
        for c in classes:
            mask = c == self.traindata._y
            images = self.traindata._x[mask]
            imagesdata = Images(images, self.traindata.trsf)
            if args.mode == 'DDP':
                sampler = utils.data.DistributedSampler(imagesdata, shuffle=False)
                loader = DataLoader(imagesdata, args.bs, num_workers=args.workers, sampler=sampler)
            else:
                loader = DataLoader(imagesdata, args.bs, num_workers=args.workers, shuffle=False)
            representations = []
            for x_, _ in loader:
                x_ = x_.to(self.device)
                with torch.no_grad():
                    self.model(x_)
                representations.append(self.get_hook_result(variable_name='representations'))
            representations = torch.cat(representations, dim=0)
            if args.mode == 'DDP':
                torch.distributed.barrier()
                representations = distributed_concat(representations)
            representations = F.normalize(representations, p=2, dim=1)
            mean_representation = representations.mean(dim=0, keepdim=True)
            init_weights.append(F.normalize(mean_representation, p=2, dim=0) * pre_weight_norm_mean)
        init_weights = torch.cat(init_weights, dim=0)
        if args.mode in ('DP', 'DDP'):
            self.model.module.fc.fc2.weight.data = init_weights.data
        else:
            self.model.fc.fc2.weight.data = init_weights.data
        self.destroy_hook()

    def init_optimizer_custom(self):
        ignored_params = list(map(id, self.model.module.fc.fc1.parameters() if args.mode in ('DP', 'DDP')
        else self.model.fc.fc1.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())
        params = [{'params': base_params, 'lr': args.lr, 'weight_decay': args.decay},
                  {'params': self.model.module.fc.fc1.parameters() if args.mode in ('DP', 'DDP')
                  else self.model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
        if args.optim == 'SGD':
            self.optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.decay,
                                             momentum=args.momentum)
        else:
            self.optimizer = None
        if args.steps is not None and self.optimizer is not None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.steps,
                                                                  gamma=args.gamma)
        else:
            self.scheduler = None

    def create_hook(self):
        self.register_hook(model=memory.pre_model, module_name='fc',
                           variable_name='pre_representations',
                           call_back=self.hook_abstract_call_back, side='input')
        self.register_hook(model=self.model, module_name='fc',
                           variable_name='cur_representations',
                           call_back=self.hook_abstract_call_back, side='input')
        self.register_hook(model=self.model, module_name='fc.fc1',
                           variable_name='fc1_scores',
                           call_back=self.hook_abstract_call_back, side='output')
        self.register_hook(model=self.model, module_name='fc.fc2',
                           variable_name='fc2_scores',
                           call_back=self.hook_abstract_call_back, side='output')
