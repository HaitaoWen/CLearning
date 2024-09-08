import math
from scheme.replay.er import ER
from scheme.base import *
from copy import deepcopy
from scheme.replay.podnet.podnet import PODNet


class AFC(PODNet):
    """
    Kang M, Park J, Han B.
    Class-Incremental Learning by Knowledge Distillation with Adaptive Feature Consolidation
    [C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
    2022: 16071-16080.
    """
    def __init__(self, model, traindata, taskid):
        super(AFC, self).__init__(model, traindata, taskid)
        warnings.filterwarnings('ignore')
        # TODO, temporal code
        if self.taskid > 1:
            model_ = self.model.module if args.mode in ('DP', 'DDP') else self.model
            if not hasattr(model_, 'importance'):
                importance = self.get_importance()
                setattr(model_, 'importance', importance)

    def train(self):
        self.podnet_train()
        #if self.taskid > 1:
        # for cbf_finetune
        self.construct_exemplar_set()
        # finetune only with memory (pre + cur)
        self.cbf_finetune()
        # use current data and previous
        # memory to update importance
        #self.drop_curtask_exemplar()
        self.update_importance()
        # *************************** #
        # statistic of BN is updated,
        # so exemplars are not matched
        # self.construct_exemplar_set()
        # *************************** #
        self.construct_nme_classifier()
        pre_model = deepcopy(self.model)
        setattr(memory, 'pre_model', self.freeze_model(pre_model))
        return self.model

    def podnet_train(self):
        dis_factor = args.feature_distil["scheduled_factor"] * math.sqrt(sum(args.increments[:self.taskid])
                                                                         / args.increments[self.taskid - 1])
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt)
        self.progress.set_description('train')
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                loss = self.nca_loss(outputs["logits"], y,
                                     scale=self.model.module.post_processor.factor
                                     if args.mode in ('DP', 'DDP') else self.model.post_processor.factor,
                                     margin=args.nca['margin'],
                                     exclude_pos_denominator=args.nca['exclude_pos_denominator'])
                if self.taskid > 1:
                    old_outputs = memory.pre_model(x)

                    loss_dis = self.pod_loss(old_outputs["attention"], outputs["attention"],
                                             feature_distil_factor=self.model.module.importance
                                             if args.mode in ('DP', 'DDP') else self.model.importance,
                                             **args.feature_distil) * dis_factor

                    TBWriter.add_scalar('Loss_NCA', loss.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_DIS', loss_dis.detach().cpu().data.numpy(), memory.iteration)
                    memory.iteration += 1

                    loss = loss + loss_dis

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

    def get_importance(self, model=None):
        if model is None:
            model = self.model
        if isinstance(model, nn.DataParallel) or \
                isinstance(model, nn.parallel.DistributedDataParallel):
            model_ = model.module
        else:
            model_ = model
        importance = [model_.convnet.stage_1_importance.importance,
                      model_.convnet.stage_2_importance.importance,
                      model_.convnet.stage_3_importance.importance,
                      model_.convnet.stage_4_importance.importance]
        return importance

    def update_importance(self):
        """
        statistic of BN is free, 
        be consistent with original implementation
        and we de-parallelize model, because hooks for importance
        are not suitable for DP/DDP.
        """
        self.model.eval()  # Unsatisfied operation
        ER.__init__(self, self.model, self.traindata, self.taskid)
        if args.mode in ('DP', 'DDP'):
            self.model.module.convnet.reset_importance()
            self.model.module.convnet.start_cal_importance()
        else:
            self.model.convnet.reset_importance()
            self.model.convnet.start_cal_importance()
        printlog('update importance')
        progress = tqdm(range(1, len(self.trainloader) + 1),
                        disable='slient' in args.opt)
        progress.set_description('\^_^/')
        for x, y, t in self.trainloader:
            x = x.to(self.device)
            y = y.to(self.device)
            outputs = self.model(x)
            loss = self.nca_loss(outputs["logits"], y,
                                 scale=self.model.module.post_processor.factor
                                 if args.mode in ('DP', 'DDP') else self.model.post_processor.factor,
                                 margin=args.nca['margin'],
                                 exclude_pos_denominator=args.nca['exclude_pos_denominator'])
            loss.backward()
            progress.update(1)
        progress.close()
        if args.mode in ('DP', 'DDP'):
            self.model.module.convnet.stop_cal_importance()
            self.model.module.convnet.normalize_importance()
            importance = self.get_importance()
            setattr(self.model.module, 'importance', importance)
        else:
            self.model.convnet.stop_cal_importance()
            self.model.convnet.normalize_importance()
            importance = self.get_importance()
            setattr(self.model, 'importance', importance)
