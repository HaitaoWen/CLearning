import torch.nn
from scheme.base import *
from copy import deepcopy
from types import MethodType
import torch.nn.functional as F
from scheme.replay.er import ER
from scheme.replay.icarl import iCaRL
from scheme.replay.lucir import LUCIR
from scheme.replay.podnet.podnet import PODNet
from scheme.replay.ssil import SSIL
from scheme.replay.afc.afc import AFC
from scheme.replay.ancl import ANCL
from scheme.regularization.lwf import LwF
from scheme.replay.opc.curve import *
from scheme.replay.podnet.my_resnet import Stage
from scheme.replay.podnet.classifiers import CosineClassifier
from scheme.replay.podnet.postprocessors import FactorScalar
from scheme.regularization.gpm.dataset.dataset import UnifiedDataset
from scheme.replay.mtd.misc import (PermutableResidualBlockCifarScalable,
                                    PermutableBasicBlockImageNetScalable,
                                    PermutableBasicBlockLUCIRImageNet,
                                    PermutableBasicBlockSSILCifar,
                                    initialize_instanceA_from_instanceB,
                                    ModelWrapper, LUCIRModelWraaper, SSILModelWraaper)

if hasattr(args, 'base') and args.scheme == 'MTD':
    if args.base == 'iCaRL':
        BASE = iCaRL
    elif args.base == 'LUCIR':
        BASE = LUCIR
    elif args.base == 'PODNet':
        BASE = PODNet
    elif args.base == 'AFC':
        BASE = AFC
    elif args.base == 'ANCL':
        BASE = ANCL
    elif args.base == 'SSIL':
        BASE = SSIL
    elif args.base == 'LwF':
        BASE = LwF
    else:
        printlog('\033[1;30;41Predetermined base {} does not exist, '
                 'use default base: LUCIR\033[0m'.format(args.scheme))
        BASE = iCaRL
else:
    BASE = iCaRL


class MTD(BASE):
    # TODO, although task 2 uses the same KL-divergence as the baseline (PODNet + KL),
    #  the final accuracy of original head on tasks 1 and 2 is not same as the baseline,
    #  for the original classifier head, this is because we use the diversity
    #  regularization to make heads (including the original) dissimilar.
    #  The accuracy is lower than the baseline, that may be the reason
    def __init__(self, model, traindata, taskid):
        if args.resume is not None:
            id1 = args.resume.rfind('/task')
            id2 = args.resume.rfind('.pkl')
            taskid_ = eval(args.resume[id1 + 5:id2])
            if taskid == taskid_ + 1:
                # only support resume from the first checkpoint
                # (i.e., task1.pkl) of BASE, taskid = 2
                Base.__init__(self, model, traindata, taskid - 1)  # Do not support FIVE
                self.inject_branch_wrap_model()
                # resume from MTD, pre_model in the memory already has branches
                if 'mtd' not in args.resume:
                    if args.base in ('PODNet', 'AFC', 'ANCL'):
                        self.cbf_finetune()  # only for resuming from PODNet/AFC starting from the 1st task
                    elif args.base == 'LUCIR':
                        self.lucir_cbf_finetune()
                    elif args.base == 'SSIL':
                        self.ssil_cbf_finetune()
                    elif args.base == 'LwF':
                        self.lwf_cbf_finetune()
                    else:
                        raise NotImplementedError
                    pre_model = deepcopy(self.model)
                    if args.mode in ('DP', 'DDP'):
                        pre_model.module.freeze_branch()
                        pre_model.module.forward_with_branch = True
                    else:
                        pre_model.freeze_branch()  # freeze the branch
                        pre_model.forward_with_branch = True
                    setattr(memory, 'pre_model', self.freeze_model(pre_model))  # freeze the model
                self.remove_branch_recover_forward(model)
            elif taskid > taskid_ + 1:
                self.remove_branch_recover_forward(model)
        elif taskid > 1:
            self.remove_branch_recover_forward(model)
        super(MTD, self).__init__(model, traindata, taskid)

    def train(self):
        if args.base == 'PODNet':
            time_memory_snapshot('Train [begin]:', device=self.device)
            self.podnet_train()  # OK!
            time_memory_snapshot('Train [end]:', device=self.device)
            self.construct_exemplar_set()
            self.inject_branch_wrap_model()
            time_memory_snapshot('CBF [begin]:', device=self.device)
            self.cbf_finetune()
            time_memory_snapshot('CBF [end]:', device=self.device)
            if args.mode in ('DP', 'DDP'):
                self.model.module.forward_with_branch = False
            else:
                self.model.forward_with_branch = False
            self.construct_nme_classifier()
            if args.mode in ('DP', 'DDP'):
                self.model.module.forward_with_branch = True
            else:
                self.model.forward_with_branch = True
            pre_model = deepcopy(self.model)
            setattr(memory, 'pre_model', self.freeze_model(pre_model))
            return self.model

        elif args.base == 'AFC':
            self.podnet_train()
            self.construct_exemplar_set()
            if 'separate' in args.mtd and args.mtd['separate']:
                super(MTD, self).cbf_finetune()
            self.inject_branch_wrap_model()
            self.cbf_finetune()
            if args.mode in ('DP', 'DDP'):
                self.model.module.forward_with_branch = False
            else:
                self.model.forward_with_branch = False
            self.update_importance()
            self.construct_nme_classifier()
            if args.mode in ('DP', 'DDP'):
                self.model.module.forward_with_branch = True
            else:
                self.model.forward_with_branch = True
            pre_model = deepcopy(self.model)
            setattr(memory, 'pre_model', self.freeze_model(pre_model))
            return self.model

        elif args.base == 'LUCIR':
            self.lucir_train()
            self.construct_exemplar_set()
            self.inject_branch_wrap_model()
            self.lucir_cbf_finetune()
            if args.mode in ('DP', 'DDP'):
                self.model.module.forward_with_branch = False
            else:
                self.model.forward_with_branch = False
            self.construct_nme_classifier()
            if args.mode in ('DP', 'DDP'):
                self.model.module.forward_with_branch = True
            else:
                self.model.forward_with_branch = True
            pre_model = deepcopy(self.model)
            setattr(memory, 'pre_model', self.freeze_model(pre_model))
            return self.model

        elif args.base == 'ANCL':
            if self.taskid > 1:
                self.aux_model_train()
                self.aux_model = self.freeze_model(self.aux_model)
            self.ancl_train()
            self.construct_exemplar_set()
            self.inject_branch_wrap_model()
            self.cbf_finetune()
            if args.mode in ('DP', 'DDP'):
                self.model.module.forward_with_branch = False
            else:
                self.model.forward_with_branch = False
            self.construct_nme_classifier()
            if args.mode in ('DP', 'DDP'):
                self.model.module.forward_with_branch = True
            else:
                self.model.forward_with_branch = True
            pre_model = deepcopy(self.model)
            setattr(memory, 'pre_model', self.freeze_model(pre_model))
            return self.model

        elif args.base == 'SSIL':
            self.ssil_train()
            self.construct_exemplar_set()
            self.inject_branch_wrap_model()
            self.ssil_cbf_finetune()
            if args.mode in ('DP', 'DDP'):
                self.model.module.forward_with_branch = False
            else:
                self.model.forward_with_branch = False
            self.ssil_construct_nme_classifier()
            if args.mode in ('DP', 'DDP'):
                self.model.module.forward_with_branch = True
            else:
                self.model.forward_with_branch = True
            pre_model = deepcopy(self.model)
            setattr(memory, 'pre_model', self.freeze_model(pre_model))
            return self.model

        elif args.base == 'LwF':
            self.lwf_train()
            self.lwf_construct_train_subset()
            self.inject_branch_wrap_model()
            self.lwf_cbf_finetune()
            if args.mode in ('DP', 'DDP'):
                self.model.module.forward_with_branch = False
            else:
                self.model.forward_with_branch = False
            self.nme = torch.nn.Linear(self.model.fc.in_features, sum(args.increments[:self.taskid]), bias=False,
                                       device=self.device)
            memory.x, memory.y, memory.t = None, None, None
            if args.mode in ('DP', 'DDP'):
                self.model.module.forward_with_branch = True
            else:
                self.model.forward_with_branch = True
            pre_model = deepcopy(self.model)
            setattr(memory, 'pre_model', self.freeze_model(pre_model))
            return self.model

        else:
            raise NotImplementedError

    def podnet_train(self):
        if args.base == 'PODNet':
            flat_factor = args.pod_flat["scheduled_factor"] * math.sqrt(sum(args.increments[:self.taskid])
                                                                        / args.increments[self.taskid - 1])
            pod_factor = args.pod_spatial["scheduled_factor"] * math.sqrt(sum(args.increments[:self.taskid])
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

                        # TODO
                        old_outputs = memory.pre_model(x)

                        if ('dis_emb_avg' in args.mtd and args.mtd['dis_emb_avg'] and
                                "raw_features_avg" in old_outputs):
                            old_features = old_outputs["raw_features_avg"]
                        else:
                            old_features = old_outputs["raw_features"]

                        loss_flat = F.cosine_embedding_loss(outputs["raw_features"], old_features.detach(),
                                                            torch.ones(old_features.shape[0]).to(self.device)
                                                            ) * flat_factor

                        loss_spatial = self.pod_loss(old_outputs["attention"], outputs["attention"],
                                                     **args.pod_spatial) * pod_factor

                        y_log_ = F.log_softmax(outputs['logits'][:, self.pre_minclass: self.pre_maxclass + 1]
                                               / args.mtd['tau'], dim=1)
                        pre_output = F.softmax(
                            old_outputs['logits'].detach()[:, self.pre_minclass: self.pre_maxclass + 1]
                            / args.mtd['tau'], dim=1)
                        loss_kl = F.kl_div(y_log_, pre_output, reduction='batchmean') * args.mtd['lambd']

                        TBWriter.add_scalar('Loss_NCA', loss.detach().cpu().data.numpy(), memory.iteration)
                        TBWriter.add_scalar('Loss_FLAT', loss_flat.detach().cpu().data.numpy(), memory.iteration)
                        TBWriter.add_scalar('Loss_SPATIAL', loss_spatial.detach().cpu().data.numpy(), memory.iteration)
                        TBWriter.add_scalar('Loss_KL', loss_kl.detach().cpu().data.numpy(), memory.iteration)
                        memory.iteration += 1

                        loss = loss + loss_flat + loss_spatial + loss_kl

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

        elif args.base == 'AFC':
            flat_factor = args.pod_flat["scheduled_factor"] * math.sqrt(sum(args.increments[:self.taskid])
                                                                        / args.increments[self.taskid - 1])
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

                        if ('dis_emb_avg' in args.mtd and args.mtd['dis_emb_avg'] and
                                "raw_features_avg" in old_outputs):
                            old_features = old_outputs["raw_features_avg"]
                        else:
                            old_features = old_outputs["raw_features"]

                        loss_flat = F.cosine_embedding_loss(outputs["raw_features"], old_features.detach(),
                                                            torch.ones(old_features.shape[0]).to(self.device)
                                                            ) * flat_factor

                        loss_dis = self.pod_loss(old_outputs["attention"], outputs["attention"],
                                                 feature_distil_factor=self.model.module.importance
                                                 if args.mode in ('DP', 'DDP') else self.model.importance,
                                                 **args.feature_distil) * dis_factor

                        y_log_ = F.log_softmax(outputs['logits'][:, self.pre_minclass: self.pre_maxclass + 1]
                                               / args.mtd['tau'], dim=1)
                        pre_output = F.softmax(
                            old_outputs['logits'].detach()[:, self.pre_minclass: self.pre_maxclass + 1]
                            / args.mtd['tau'], dim=1)
                        loss_kl = F.kl_div(y_log_, pre_output, reduction='batchmean') * args.mtd['lambd']

                        TBWriter.add_scalar('Loss_NCA', loss.detach().cpu().data.numpy(), memory.iteration)
                        TBWriter.add_scalar('Loss_FLAT', loss_flat.detach().cpu().data.numpy(), memory.iteration)
                        TBWriter.add_scalar('Loss_DIS', loss_dis.detach().cpu().data.numpy(), memory.iteration)
                        TBWriter.add_scalar('Loss_KL', loss_kl.detach().cpu().data.numpy(), memory.iteration)
                        memory.iteration += 1

                        if 'dis_emb_avg' in args.mtd and args.mtd['dis_emb_avg']:
                            loss = loss + loss_dis + loss_kl + loss_flat
                        else:
                            loss = loss + loss_dis + loss_kl

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

    def ancl_train(self):
        flat_factor = args.pod_flat["scheduled_factor"] * math.sqrt(sum(args.increments[:self.taskid])
                                                                    / args.increments[self.taskid - 1])
        pod_factor = args.pod_spatial["scheduled_factor"] * math.sqrt(sum(args.increments[:self.taskid])
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

                    if ('dis_emb_avg' in args.mtd and args.mtd['dis_emb_avg'] and
                            "raw_features_avg" in old_outputs):
                        old_features = old_outputs["raw_features_avg"]
                    else:
                        old_features = old_outputs["raw_features"]

                    loss_flat = F.cosine_embedding_loss(outputs["raw_features"], old_features.detach(),
                                                        torch.ones(old_features.shape[0]).to(self.device)
                                                        ) * flat_factor

                    loss_spatial = self.pod_loss(old_outputs["attention"], outputs["attention"],
                                                 **args.pod_spatial) * pod_factor

                    aux_outputs = self.aux_model(x)
                    aux_features = aux_outputs["raw_features"]

                    loss_flat_aux = F.cosine_embedding_loss(outputs["raw_features"], aux_features.detach(),
                                                            torch.ones(aux_features.shape[0]).to(self.device)
                                                            ) * flat_factor

                    loss_spatial_aux = self.pod_loss(aux_outputs["attention"], outputs["attention"],
                                                     **args.pod_spatial) * pod_factor

                    y_log_ = F.log_softmax(outputs['logits'][:, self.pre_minclass: self.pre_maxclass + 1]
                                           / args.mtd['tau'], dim=1)
                    pre_output = F.softmax(
                        old_outputs['logits'].detach()[:, self.pre_minclass: self.pre_maxclass + 1]
                        / args.mtd['tau'], dim=1)
                    loss_kl = F.kl_div(y_log_, pre_output, reduction='batchmean') * args.mtd['lambd']

                    TBWriter.add_scalar('Loss_NCA', loss.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_FLAT', loss_flat.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_SPATIAL', loss_spatial.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_FLAT_AUX', loss_flat_aux.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_SPATIAL_AUX', loss_spatial_aux.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_KL', loss_kl.detach().cpu().data.numpy(), memory.iteration)
                    memory.iteration += 1

                    loss = loss + (loss_flat + loss_spatial) * args.ANCL['lambda'] \
                                + (loss_flat_aux + loss_spatial_aux) * args.ANCL['lambda_a'] \
                                + loss_kl

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
                    old_outputs = memory.pre_model(x)

                    fc1_scores = self.get_hook_result(self.model, variable_name='fc1_scores')
                    fc2_scores = self.get_hook_result(self.model, variable_name='fc2_scores')
                    cur_representations = self.get_hook_result(self.model, variable_name='cur_representations')
                    pre_representations = self.get_hook_result(memory.pre_model, variable_name='pre_representations')

                    # TDOD
                    if ('dis_emb_avg' in args.mtd and args.mtd['dis_emb_avg'] and
                            "embedding_avg" in old_outputs):
                        pre_representations = old_outputs["embedding_avg"]

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

                    y_log_ = F.log_softmax(y_[:, self.pre_minclass: self.pre_maxclass + 1]
                                           / args.mtd['tau'], dim=1)
                    pre_output = F.softmax(
                        old_outputs['logits'].detach()[:, self.pre_minclass: self.pre_maxclass + 1]
                        / args.mtd['tau'], dim=1)
                    loss_kl = F.kl_div(y_log_, pre_output, reduction='batchmean') * args.mtd['lambd']

                    TBWriter.add_scalar('Loss_CE', loss.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_EB', loss_embed.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_MR', loss_mr.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_KL', loss_kl.detach().cpu().data.numpy(), memory.iteration)
                    memory.iteration += 1

                    loss = loss + loss_embed + loss_mr + loss_kl

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
                    y_pre = memory.pre_model(x)['logits'].detach()
                    for taskid in range(1, self.taskid):
                        minclass, maxclass = get_minmax_class(taskid)
                        pre_minclass, pre_maxclass = get_minmax_class(taskid - 1)
                        y_log_ = F.log_softmax(y_[:, pre_maxclass+1: maxclass+1] / args.tau, dim=1)
                        pre_output = F.softmax(y_pre[:, pre_maxclass+1: maxclass+1] / args.tau, dim=1)
                        loss_div += F.kl_div(y_log_, pre_output, reduction='batchmean') * args.mtd['lambd']

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

    def lwf_train(self):
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
                    loss = F.cross_entropy(y_[:, self.pre_maxclass+1:self.maxclass+1], y-(self.pre_maxclass+1))
                    y_log_ = F.log_softmax(y_[:, self.pre_minclass: self.pre_maxclass + 1] / args.tau, dim=1)
                    pre_output = F.softmax(memory.pre_model(x)['logits'].detach()[:, self.pre_minclass:
                                                                                     self.pre_maxclass + 1]
                                           / args.tau, dim=1)  # TODO
                    loss_div = F.kl_div(y_log_, pre_output, reduction='batchmean') * args.lambd

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

    def lucir_cbf_finetune(self):
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
        if args.mode in ('DP', 'DDP'):
            self.model.module.branch_training(flag=True)
            self.model.module.forward_with_branch = True
        else:
            self.model.branch_training(flag=True)  # TODO
            self.model.forward_with_branch = True

        # *********** overwrite self.optimizer and self.scheduler *********** #
        # both the old and novel embeddings are updated with the feature extractor fixed
        ignored_params = list(map(id, self.model.module.fc.parameters() if args.mode in ('DP', 'DDP')
                              else self.model.fc.parameters()))
        param_branch = self.model.module.branch_parameters() \
            if args.mode in ('DP', 'DDP') else self.model.branch_parameters()
        base_params = []
        for n, p in self.model.named_parameters():
            if id(p) not in ignored_params and 'branch' not in n:
                base_params.append(p)
        params = [{'params': self.model.module.fc.fc1.parameters() if args.mode in ('DP', 'DDP')
                  else self.model.fc.fc1.parameters(), 'lr': args.finetune['lr'], 'weight_decay': args.decay},
                  {'params': self.model.module.fc.fc2.parameters() if args.mode in ('DP', 'DDP')
                  else self.model.fc.fc2.parameters(), 'lr': args.finetune['lr'], 'weight_decay': args.decay},
                  {"params": param_branch, "lr": args.finetune['blr'], 'weight_decay': args.decay},
                  {'params': base_params, 'lr': 0, 'weight_decay': 0}]
        self.optimizer = torch.optim.SGD(params, lr=args.finetune['lr'], momentum=args.momentum,
                                         weight_decay=args.decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.finetune['steps'],
                                                              gamma=args.finetune['gamma'])
        # ************************** overwrite configs ************************** #
        args_copy = deepcopy(args)
        args.epochs = args.finetune['epochs']
        # ********************** multiplexing self.lucir_train ********************** #
        self.destroy_hook()
        self.create_hook()
        self.lucir_finetune()  # change to lucir_finetune
        self.destroy_hook()
        args.__dict__.update(args_copy.__dict__)
        self.model.eval()
        if args.mode in ('DP', 'DDP'):
            self.model.module.freeze_branch()
        else:
            self.model.freeze_branch()
        if args.debug:
            self.evaluate_()

    def ssil_cbf_finetune(self):
        if self.taskid == 1 or not hasattr(args, 'finetune'):
            return
        printlog('class-balanced finetuning', delay=1.0)
        # ****************** load memory data ****************** #
        trsf = self.traindata.trsf
        if args.dataset in ('CIFAR100', 'SVHN', 'MNIST'):
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
        if args.mode in ('DP', 'DDP'):
            self.model.module.branch_training(flag=True)
            self.model.module.forward_with_branch = True
        else:
            self.model.branch_training(flag=True)  # TODO
            self.model.forward_with_branch = True

        # *********** overwrite self.optimizer and self.scheduler *********** #
        # both the old and novel embeddings are updated with the feature extractor fixed
        ignored_params = list(map(id, self.model.module.fc.parameters() if args.mode in ('DP', 'DDP')
        else self.model.fc.parameters()))
        param_branch = self.model.module.branch_parameters() \
            if args.mode in ('DP', 'DDP') else self.model.branch_parameters()
        base_params = []
        for n, p in self.model.named_parameters():
            if id(p) not in ignored_params and 'branch' not in n:
                base_params.append(p)
        params = [{'params': self.model.module.fc.parameters() if args.mode in ('DP', 'DDP')
                  else self.model.fc.parameters(), 'lr': 0, 'weight_decay': 0},
                  {"params": param_branch, "lr": args.finetune['blr'], 'weight_decay': args.decay},
                  {'params': base_params, 'lr': 0, 'weight_decay': 0}]
        self.optimizer = torch.optim.SGD(params, lr=args.finetune['blr'], momentum=args.momentum,
                                         weight_decay=args.decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.finetune['steps'],
                                                              gamma=args.finetune['gamma'])
        # ************************** overwrite configs ************************** #
        args_copy = deepcopy(args)
        args.epochs = args.finetune['epochs']
        self.ssil_finetune()  # change to ssil_finetune
        args.__dict__.update(args_copy.__dict__)
        self.model.eval()
        if args.mode in ('DP', 'DDP'):
            self.model.module.freeze_branch()
        else:
            self.model.freeze_branch()
        if args.debug:
            self.evaluate_()

    def lwf_cbf_finetune(self):
        if self.taskid == 1 or not hasattr(args, 'finetune'):
            return
        printlog('class-balanced finetuning', delay=1.0)
        # ****************** load memory data ****************** #
        trsf = self.traindata.trsf
        if args.dataset in ('CIFAR100', 'SVHN', 'MNIST', 'FIVE'):
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
        if args.mode in ('DP', 'DDP'):
            self.model.module.branch_training(flag=True)
            self.model.module.forward_with_branch = True
        else:
            self.model.branch_training(flag=True)  # TODO
            self.model.forward_with_branch = True

        # *********** overwrite self.optimizer and self.scheduler *********** #
        # both the old and novel embeddings are updated with the feature extractor fixed
        ignored_params = list(map(id, self.model.module.fc.parameters() if args.mode in ('DP', 'DDP')
        else self.model.fc.parameters()))
        param_branch = self.model.module.branch_parameters() \
            if args.mode in ('DP', 'DDP') else self.model.branch_parameters()
        base_params = []
        for n, p in self.model.named_parameters():
            if id(p) not in ignored_params and 'branch' not in n:
                base_params.append(p)
        params = [{'params': self.model.module.fc.parameters() if args.mode in ('DP', 'DDP')
        else self.model.fc.parameters(), 'lr': 0, 'weight_decay': 0},
                  {"params": param_branch, "lr": args.finetune['blr'], 'weight_decay': args.decay},
                  {'params': base_params, 'lr': 0, 'weight_decay': 0}]
        self.optimizer = torch.optim.SGD(params, lr=args.finetune['blr'], momentum=args.momentum,
                                         weight_decay=args.decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.finetune['steps'],
                                                              gamma=args.finetune['gamma'])
        # ************************** overwrite configs ************************** #
        args_copy = deepcopy(args)
        args.epochs = args.finetune['epochs']
        self.lwf_finetune()  # change to ssil_finetune
        args.__dict__.update(args_copy.__dict__)
        self.model.eval()
        if args.mode in ('DP', 'DDP'):
            self.model.module.freeze_branch()
        else:
            self.model.freeze_branch()
        if args.debug:
            self.evaluate_()

    def inject_branch_wrap_model(self):
        # TODO, only support non distributed model
        assert 'permute' in args.mtd and 'scale' in args.mtd

        if isinstance(self.model, torch.nn.DataParallel) or \
                isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_ = self.model.module
        else:
            model_ = self.model
        if 'ridge' in args.mtd and args.mtd['ridge'] > 0:
            num = args.mtd['num'] + 1
        else:
            num = args.mtd['num']
        if args.base in ('PODNet', 'AFC', 'ANCL'):
            if args.dataset == 'CIFAR100':
                for index in range(1, num + 1):
                    # ************** stage 3 ************** #
                    if 'stage_3' in args.mtd['stages']:
                        stage_3 = []
                        for module in model_.convnet.stage_3.blocks:
                            # default not to scale
                            block = PermutableResidualBlockCifarScalable(module,
                                                                         iconv_requires_grad=True,
                                                                         random_permute=args.mtd['permute'],
                                                                         random_scale=args.mtd['scale']).to(self.device)
                            block.permute(block.gen_permutation())
                            stage_3.append(block)

                        stage_3 = torch.nn.ModuleList(stage_3)
                        stage_3 = Stage(stage_3, block_relu=model_.convnet.stage_3.block_relu)
                        model_.register_module('branch_stage_3_{}'.format(index), stage_3.to(self.device))
                    # ************** stage 4 ************** #
                    if 'stage_4' in args.mtd['stages']:
                        stage_4 = PermutableResidualBlockCifarScalable(model_.convnet.stage_4,
                                                                       iconv_requires_grad=True,
                                                                       random_permute=args.mtd['permute'],
                                                                       random_scale=args.mtd['scale']).to(self.device)
                        stage_4.permute(stage_4.gen_permutation())

                        classifier = CosineClassifier(features_dim=model_.classifier.features_dim)
                        initialize_instanceA_from_instanceB(classifier, model_.classifier)
                        post_processor = FactorScalar()
                        initialize_instanceA_from_instanceB(post_processor, model_.post_processor)
                        model_.register_module('branch_stage_4_{}'.format(index), stage_4)
                        model_.register_module('branch_classifier_{}'.format(index), classifier.to(self.device))
                        model_.register_module('branch_post_processor_{}'.format(index), post_processor.to(self.device))

            elif args.dataset in ('ImageNet100', 'ImageNet1000', 'DTD', 'CUB'):
                for index in range(1, num + 1):
                    # ************** stage 3 ************** #
                    if 'layer3' in args.mtd['stages']:
                        layer3 = []
                        for module in model_.convnet.layer3:
                            block = PermutableBasicBlockImageNetScalable(module,
                                                                         iconv_requires_grad=True,
                                                                         random_permute=args.mtd['permute'],
                                                                         random_scale=args.mtd['scale']).to(self.device)
                            block.permute(block.gen_permutation())
                            layer3.append(block)

                        layer3 = torch.nn.Sequential(*layer3)
                        model_.register_module('branch_layer3_{}'.format(index), layer3.to(self.device))
                    # ************** stage 4 ************** #
                    if 'layer4' in args.mtd['stages']:
                        layer4 = []
                        for module in model_.convnet.layer4:
                            block = PermutableBasicBlockImageNetScalable(module,
                                                                         iconv_requires_grad=True,
                                                                         random_permute=args.mtd['permute'],
                                                                         random_scale=args.mtd['scale']).to(self.device)
                            block.permute(block.gen_permutation())
                            layer4.append(block)

                        layer4 = torch.nn.Sequential(*layer4)
                        model_.register_module('branch_layer4_{}'.format(index), layer4.to(self.device))
                        classifier = CosineClassifier(features_dim=model_.classifier.features_dim)
                        initialize_instanceA_from_instanceB(classifier, model_.classifier)
                        post_processor = FactorScalar()
                        initialize_instanceA_from_instanceB(post_processor, model_.post_processor)
                        model_.register_module('branch_classifier_{}'.format(index), classifier.to(self.device))
                        model_.register_module('branch_post_processor_{}'.format(index), post_processor.to(self.device))
            else:
                raise NotImplementedError

        elif args.base == 'LUCIR':
            self.destroy_hook()
            if args.dataset in ('ImageNet100', 'ImageNet1000'):
                for index in range(1, num + 1):
                    # ************** stage 3 ************** #
                    if 'layer3' in args.mtd['stages']:
                        layer3 = []
                        for module in model_.layer3:
                            block = PermutableBasicBlockLUCIRImageNet(module,
                                                                      iconv_requires_grad=True,
                                                                      random_permute=args.mtd['permute']).to(self.device)
                            block.permute(block.gen_permutation())
                            layer3.append(block)

                        layer3 = torch.nn.Sequential(*layer3)
                        model_.register_module('branch_layer3_{}'.format(index), layer3.to(self.device))
                    # ************** stage 4 ************** #
                    if 'layer4' in args.mtd['stages']:
                        layer4 = []
                        for module in model_.layer4:
                            block = PermutableBasicBlockLUCIRImageNet(module,
                                                                      iconv_requires_grad=True,
                                                                      random_permute=args.mtd['permute']).to(self.device)
                            block.permute(block.gen_permutation())
                            layer4.append(block)

                        layer4 = torch.nn.Sequential(*layer4)
                        model_.register_module('branch_layer4_{}'.format(index), layer4.to(self.device))
                        model_.register_module('branch_fc_{}'.format(index), deepcopy(model_.fc))
            else:
                raise NotImplementedError

        elif args.base in ('SSIL', 'LwF'):
            self.destroy_hook()
            if args.dataset in ('CIFAR100', 'SVHN', 'FIVE'):
                for index in range(1, num + 1):
                    # ************** stage 3 ************** #
                    if 'layer2' in args.mtd['stages']:
                        layer2 = []
                        for module in model_.layer2:
                            block = PermutableBasicBlockSSILCifar(module,
                                                                  iconv_requires_grad=True,
                                                                  random_permute=args.mtd['permute']).to(self.device)
                            block.permute(block.gen_permutation())
                            layer2.append(block)

                        layer2 = torch.nn.Sequential(*layer2)
                        model_.register_module('branch_layer2_{}'.format(index), layer2.to(self.device))
                    # ************** stage 4 ************** #
                    if 'layer3' in args.mtd['stages']:
                        layer3 = []
                        for module in model_.layer3:
                            block = PermutableBasicBlockSSILCifar(module,
                                                                  iconv_requires_grad=True,
                                                                  random_permute=args.mtd['permute']).to(self.device)
                            block.permute(block.gen_permutation())
                            layer3.append(block)

                        layer3 = torch.nn.Sequential(*layer3)
                        model_.register_module('branch_layer3_{}'.format(index), layer3.to(self.device))
                        model_.register_module('branch_fc_{}'.format(index), deepcopy(model_.fc))
            elif 'MNIST' in args.dataset:
                assert args.mtd['stages'] == ['layer1', 'fc']
                for index in range(1, num + 1):
                    linear = self.model.body.layer_1.linear
                    fc = self.model.fc

                    linear_b = torch.nn.Linear(linear.in_features, linear.out_features, bias=False).to(self.device)
                    index_ = torch.randperm(linear_b.out_features, requires_grad=False, device=self.device)
                    P = torch.eye(linear_b.out_features)[:, index_].to(self.device)
                    linear_b.weight.data = P @ linear.weight.data
                    fc_b = torch.nn.Linear(fc.in_features, fc.out_features, bias=False).to(self.device)
                    fc_b.weight.data = fc.weight.data @ P.T

                    model_.register_module('branch_layer1_{}'.format(index), linear_b.to(self.device))
                    model_.register_module('branch_fc_{}'.format(index), fc_b)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        if args.base in ('PODNet', 'AFC', 'ANCL'):
            model_ = ModelWrapper(model_)
            self.model = model_
            self.allocate_model()
        elif args.base == 'LUCIR':
            model_ = LUCIRModelWraaper(model_, args)
            self.model = model_
            self.allocate_model()
        elif args.base in ('SSIL', 'LwF'):
            model_ = SSILModelWraaper(model_, args)
            self.model = model_
            self.allocate_model()
        else:
            raise NotImplementedError

    @staticmethod
    def remove_branch_recover_forward(model_):
        # remove branch in the current model (optional)
        # and recover forward without branch (normal training)
        if isinstance(model_, torch.nn.DataParallel) or \
                isinstance(model_, torch.nn.parallel.DistributedDataParallel):
            model__ = model_.module
        else:
            model__ = model_
        branch_names = []
        for name, module in model__.named_modules():
            if 'branch' not in name:
                continue
            branch_names.append(name)
        for name in branch_names:
            if hasattr(model__, name):
                exec('del model__.{}'.format(name))
        model__.forward_with_branch = False

    def cbf_finetune(self):
        if self.taskid == 1 or not hasattr(args, 'finetuning_config'):
            # skip the first task, but will additionally
            # use the KL-divergence to train the model
            # (all branches outputs be the same)
            return
        printlog('class-balanced finetuning', delay=1.0)
        # ****************** load memory data ****************** #
        trsf = self.traindata.trsf
        if args.dataset == 'CIFAR100':
            memdata = Memorydata(memory, trsf)
        elif 'ImageNet' in args.dataset or args.dataset in ('DTD', 'CUB'):
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

        # *********** overwrite self.optimizer and self.scheduler *********** #
        if 'separate' in args.mtd and args.mtd['separate']:
            param_classifier = []
        else:
            param_classifier = list(self.model.module.classifier.parameters()
                                    if args.mode in ('DP', 'DDP') else self.model.classifier.parameters())
        param_branch = self.model.module.branch_parameters() \
            if args.mode in ('DP', 'DDP') else self.model.branch_parameters()
        params = [{"params": param_classifier, "lr": args.finetuning_config['lr']},
                  {"params": param_branch, "lr": args.finetuning_config['blr']}]
        self.optimizer = torch.optim.SGD(params,
                                         lr=args.finetuning_config['lr'],
                                         weight_decay=args.decay, momentum=args.momentum)
        self.scheduler = None
        # ************************** overwrite configs ************************** #
        args_copy = deepcopy(args)
        args.epochs = args.finetuning_config['epochs']
        # ************************** adjust model state ************************** #
        # it is consistent with the original implementation
        if 'separate' in args.mtd and args.mtd['separate']:
            self.model.eval()
        else:
            self.model.train()
        if args.mode in ('DP', 'DDP'):
            self.model.module.branch_training(flag=True)
            self.model.module.forward_with_branch = True
        else:
            self.model.branch_training(flag=True)  # TODO
            self.model.forward_with_branch = True
        # ********************** multiplexing self.lucir_train ********************** #
        self.finetune()
        args.__dict__.update(args_copy.__dict__)
        self.model.eval()
        if args.mode in ('DP', 'DDP'):
            self.model.module.freeze_branch()
        else:
            self.model.freeze_branch()
        if args.debug:
            self.evaluate_()

    @staticmethod
    def save_finetune_model(model, taskid, epoch):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) \
                or isinstance(model, torch.nn.DataParallel):
            model = model.module
        state = model.state_dict()
        for key, value in state.items():
            state[key] = value.cpu()
        dir_ = os.path.join(args.logdir + args.name, 'pkl/finetune')
        if os.path.exists(dir_) is False:
            os.makedirs(dir_)
        torch.save(state, os.path.join(dir_, 'task{}_epoch{}.pkl'.format(taskid, epoch)))

    def finetune(self):
        if args.base in ('PODNet', 'AFC', 'ANCL'):
            self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                                 disable='slient' in args.opt)
            self.progress.set_description('train')
            model_cur = copy.deepcopy(self.model)
            self.remove_branch_recover_forward(model_cur)
            model_cur = self.freeze_model(model_cur)
            # self.save_finetune_model(self.model, self.taskid, 0)
            for epoch in range(1, args.epochs + 1):
                for x, y, t in self.trainloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    outputs = self.model(x)

                    outputs_cur = model_cur(x)

                    loss = self.nca_loss(outputs["logits_0"], y,
                                         scale=self.model.module.post_processor.factor
                                         if args.mode in ('DP', 'DDP') else self.model.post_processor.factor,
                                         margin=args.nca['margin'],
                                         exclude_pos_denominator=args.nca['exclude_pos_denominator'])
                    for index in range(1, args.mtd['num'] + 1):
                        if 'mixup' in args.mtd:
                            loss = loss + (1 - outputs['mixup_lambd_{}'.format(index)]) * \
                                   self.nca_loss(outputs["logits_{}".format(index)], y,
                                                 scale=eval('self.model.module.branch_post_processor_{}'.format(index)).factor
                                                 if args.mode in ('DP', 'DDP') else
                                                 eval('self.model.branch_post_processor_{}'.format(index)).factor,
                                                 margin=args.nca['margin'],
                                                 exclude_pos_denominator=args.nca['exclude_pos_denominator']) + \
                                   outputs['mixup_lambd_{}'.format(index)] * \
                                   self.nca_loss(outputs["logits_{}".format(index)], y[outputs['mixup_index_{}'.format(index)]],
                                                 scale=eval('self.model.module.branch_post_processor_{}'.format(index)).factor
                                                 if args.mode in ('DP', 'DDP') else
                                                 eval('self.model.branch_post_processor_{}'.format(index)).factor,
                                                 margin=args.nca['margin'],
                                                 exclude_pos_denominator=args.nca['exclude_pos_denominator'])

                        else:
                            loss = loss + self.nca_loss(outputs["logits_{}".format(index)], y,
                                                        scale=eval('self.model.module.branch_post_processor_{}'.format(index)).factor
                                                        if args.mode in ('DP', 'DDP') else
                                                        eval('self.model.branch_post_processor_{}'.format(index)).factor,
                                                        margin=args.nca['margin'],
                                                        exclude_pos_denominator=args.nca['exclude_pos_denominator'])
                    if 'dissimilar' in args.mtd:
                        loss_sim = self.branch_similarity(outputs) * args.mtd['dissimilar']
                        loss = loss + loss_sim

                    if self.taskid > 1:
                        old_outputs = memory.pre_model(x)

                        y_log_ = F.log_softmax(outputs['logits'][:, self.pre_minclass: self.pre_maxclass + 1]
                                               / args.mtd['tau'], dim=1)
                        pre_output = F.softmax(
                            old_outputs['logits'].detach()[:, self.pre_minclass: self.pre_maxclass + 1]
                            / args.mtd['tau'], dim=1)
                        loss_kl = F.kl_div(y_log_, pre_output, reduction='batchmean') * args.mtd['kl']['p']

                        y_log_ = F.log_softmax(outputs['logits'][:, self.pre_maxclass + 1:]
                                               / args.mtd['tau'], dim=1)
                        cur_output = F.softmax(
                            outputs_cur['logits'].detach()[:, self.pre_maxclass + 1:]
                            / args.mtd['tau'], dim=1)
                        loss_kl_cur = F.kl_div(y_log_, cur_output, reduction='batchmean') * args.mtd['kl']['c']

                        TBWriter.add_scalar('Loss_NCA', loss.detach().cpu().data.numpy(), memory.iteration)
                        TBWriter.add_scalar('Loss_KL', loss_kl.detach().cpu().data.numpy(), memory.iteration)
                        memory.iteration += 1

                        loss = loss + loss_kl + loss_kl_cur

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.loss = loss.item()
                    self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                               'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                               'avgloss': round(self.loss, 3), 'loss': loss.item(),
                                               'loss_sim': loss_sim.item() if 'dissimilar' in args.mtd else None})
                    self.progress.update(1)
                if self.scheduler is not None:
                    self.scheduler.step()
                # self.save_finetune_model(self.model, self.taskid, epoch)
            self.progress.close()
        else:
            raise NotImplementedError

    def lucir_finetune(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt)
        self.progress.set_description('train')
        model_cur = copy.deepcopy(self.model)
        self.remove_branch_recover_forward(model_cur)
        model_cur = self.freeze_model(model_cur)
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                bs = x.shape[0]
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                y_ = outputs['logits']

                outputs_cur = model_cur(x)

                loss = 0
                for index in range(args.mtd['num'] + 1):
                    loss = loss + self.criterion(outputs['logits_{}'.format(index)], y, t)

                if 'dissimilar' in args.mtd:
                    loss_sim = self.branch_similarity(outputs) * args.mtd['dissimilar']
                    loss = loss + loss_sim

                if self.taskid > 1:
                    old_outputs = memory.pre_model(x)

                    fc1_scores = self.get_hook_result(self.model, variable_name='fc1_scores')[:bs]
                    fc2_scores = self.get_hook_result(self.model, variable_name='fc2_scores')[:bs]
                    cur_representations = self.get_hook_result(self.model, variable_name='cur_representations')[:bs]
                    pre_representations = self.get_hook_result(memory.pre_model, variable_name='pre_representations')

                    # TDOD
                    if ('dis_emb_avg' in args.mtd and args.mtd['dis_emb_avg'] and
                            "embedding_avg" in old_outputs):
                        pre_representations = old_outputs["embedding_avg"]

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

                    y_log_ = F.log_softmax(y_[:, self.pre_minclass: self.pre_maxclass + 1]
                                           / args.mtd['tau'], dim=1)
                    pre_output = F.softmax(
                        old_outputs['logits'].detach()[:, self.pre_minclass: self.pre_maxclass + 1]
                        / args.mtd['tau'], dim=1)
                    loss_kl = F.kl_div(y_log_, pre_output, reduction='batchmean') * args.mtd['kl']['p']

                    y_log_ = F.log_softmax(y_[:, self.pre_maxclass + 1:]
                                           / args.mtd['tau'], dim=1)
                    cur_output = F.softmax(
                        outputs_cur.detach()[:, self.pre_maxclass + 1:]
                        / args.mtd['tau'], dim=1)
                    loss_kl_cur = F.kl_div(y_log_, cur_output, reduction='batchmean') * args.mtd['kl']['c']

                    TBWriter.add_scalar('Loss_CE', loss.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_EB', loss_embed.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_MR', loss_mr.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_KL', loss_kl.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_KL_CUR', loss_kl_cur.detach().cpu().data.numpy(), memory.iteration)
                    memory.iteration += 1

                    loss = loss + loss_embed + loss_mr + loss_kl + loss_kl_cur

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item(),
                                           'loss_sim': loss_sim.item() if 'dissimilar' in args.mtd else None})
                self.progress.update(1)
            if self.scheduler is not None:
                self.scheduler.step()
        self.progress.close()
        self.destroy_hook()

    def ssil_finetune(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt)
        self.progress.set_description('train')
        model_cur = copy.deepcopy(self.model)
        self.remove_branch_recover_forward(model_cur)
        model_cur = self.freeze_model(model_cur)
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                y_ = outputs['logits']

                outputs_cur = model_cur(x)

                loss = 0
                for index in range(args.mtd['num'] + 1):
                    loss = loss + self.criterion(outputs['logits_{}'.format(index)], y, t)

                if 'dissimilar' in args.mtd:
                    loss_sim = self.branch_similarity(outputs) * args.mtd['dissimilar']
                    loss = loss + loss_sim

                if self.taskid > 1:
                    loss_div = 0
                    y_pre = memory.pre_model(x)['logits'].detach()
                    for taskid in range(1, self.taskid):
                        minclass, maxclass = get_minmax_class(taskid)
                        pre_minclass, pre_maxclass = get_minmax_class(taskid - 1)
                        y_log_ = F.log_softmax(y_[:, pre_maxclass + 1: maxclass + 1] / args.tau, dim=1)
                        pre_output = F.softmax(y_pre[:, pre_maxclass + 1: maxclass + 1] / args.tau, dim=1)
                        loss_div += F.kl_div(y_log_, pre_output, reduction='batchmean') * args.mtd['kl']['p']

                    y_log_ = F.log_softmax(y_[:, self.pre_maxclass + 1:]
                                           / args.mtd['tau'], dim=1)
                    cur_output = F.softmax(
                        outputs_cur.detach()[:, self.pre_maxclass + 1:]
                        / args.mtd['tau'], dim=1)
                    loss_kl_cur = F.kl_div(y_log_, cur_output, reduction='batchmean') * args.mtd['kl']['c']

                    TBWriter.add_scalar('Loss_CE', loss.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_KL_PRE', loss_div.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_KL_CUR', loss_kl_cur.detach().cpu().data.numpy(), memory.iteration)
                    memory.iteration += 1

                    loss = loss + loss_div + loss_kl_cur

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][1]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item(),
                                           'loss_sim': loss_sim.item() if 'dissimilar' in args.mtd else None})
                self.progress.update(1)
            if self.scheduler is not None:
                self.scheduler.step()
        self.progress.close()

    def lwf_finetune(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt)
        self.progress.set_description('train')
        model_cur = copy.deepcopy(self.model)
        self.remove_branch_recover_forward(model_cur)
        model_cur = self.freeze_model(model_cur)
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                y_ = outputs['logits']

                outputs_cur = model_cur(x)

                loss = 0
                for index in range(args.mtd['num'] + 1):
                    loss = loss + self.criterion(outputs['logits_{}'.format(index)], y, t)

                if 'dissimilar' in args.mtd:
                    loss_sim = self.branch_similarity(outputs) * args.mtd['dissimilar']
                    loss = loss + loss_sim

                if self.taskid > 1:
                    y_log_ = F.log_softmax(y_[:, self.pre_minclass: self.pre_maxclass + 1] / args.mtd['tau'], dim=1)
                    pre_output = F.softmax(memory.pre_model(x)['logits'].detach()[:, self.pre_minclass:
                                                                                     self.pre_maxclass + 1]
                                           / args.mtd['tau'], dim=1)
                    loss_div = F.kl_div(y_log_, pre_output, reduction='batchmean') * args.mtd['kl']['p']

                    y_log_ = F.log_softmax(y_[:, self.pre_maxclass + 1:]
                                           / args.mtd['tau'], dim=1)
                    cur_output = F.softmax(
                        outputs_cur.detach()[:, self.pre_maxclass + 1:]
                        / args.mtd['tau'], dim=1)
                    loss_kl_cur = F.kl_div(y_log_, cur_output, reduction='batchmean') * args.mtd['kl']['c']

                    TBWriter.add_scalar('Loss_CE', loss.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_KL_PRE', loss_div.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_KL_CUR', loss_kl_cur.detach().cpu().data.numpy(), memory.iteration)
                    memory.iteration += 1

                    loss = loss + loss_div + loss_kl_cur

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][1]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item(),
                                           'loss_sim': loss_sim.item() if 'dissimilar' in args.mtd else None})
                self.progress.update(1)
            if self.scheduler is not None:
                self.scheduler.step()
        self.progress.close()

    def ssil_construct_nme_classifier(self):
        self.model.eval()
        hook_handle = self.register_hook(module_name='fc', variable_name='representations',
                                         call_back=self.hook_abstract_call_back, side='input')
        if args.mode in ('DP', 'DDP'):
            self.nme = torch.nn.Linear(self.model.module.fc.in_features, len(np.unique(memory.y)),
                                       bias=False, device=self.device)
        else:
            if 'MNIST' in args.dataset:
                _, y, __ = zip(*memory.x)
                self.nme = torch.nn.Linear(self.model.fc.in_features, len(np.unique(y)), bias=False,
                                           device=self.device)
            else:
                self.nme = torch.nn.Linear(self.model.fc.in_features, len(np.unique(memory.y)), bias=False,
                                           device=self.device)
        trsf = self.traindata.trsf
        if args.dataset in ('CIFAR100', 'SVHN', 'MNIST'):
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

    def lwf_construct_train_subset(self):
        # Herding is adopted by default
        self.model.eval()
        trsf = self.traindata.trsf
        if args.dataset in ['PMNIST', 'RMNIST'] and isinstance(trsf, list):
            # This is due to dirty implementation of the new version continuum
            trsf = trsf[self.taskid - 1]
        _x, _y = self.traindata._x, self.traindata._y
        if 'sample' in args and args.sample == 'random':
            x, y, t = [], [], []
            classes = np.unique(_y)
            for c in classes:
                mask = c == _y
                indices = np.where(mask)[0]
                indices = np.random.choice(indices, size=args.memory, replace=False)
                for index in indices:
                    x.append(_x[index])
                    y.append(_y[index])
                    t.append(self.taskid - 1)
        else:
            raise NotImplementedError

        if args.dataset in ('CIFAR10', 'CIFAR100', 'SVHN', 'FIVE'):
            x = np.stack(x, axis=0)
            y = np.stack(y, axis=0)
            t = np.stack(t, axis=0)
            memory.x = np.concatenate((memory.x, x), axis=0) if memory.x is not None else x
            memory.y = np.concatenate((memory.y, y), axis=0) if memory.y is not None else y
            memory.t = np.concatenate((memory.t, t), axis=0) if memory.t is not None else t
        elif 'MNIST' in args.dataset:
            x__ = []
            for x_, y_, t_ in zip(x, y, t):
                if trsf is not None:
                    x_ = Image.fromarray(x_.astype("uint8"))
                    x_ = trsf(x_).squeeze()
                x__.append([x_, y_, t_])
            if memory.x is None:
                memory.x = x__
            else:
                memory.x.extend(x__)
        elif 'ImageNet' in args.dataset:
            if memory.x is None:
                memory.x = x
                memory.y = y
                memory.t = t
            else:
                memory.x += x
                memory.y += y
                memory.t += t
        else:
            raise ValueError('Unsupported dataset: {}'.format(args.dataset))

    def branch_similarity(self, outputs):
        similarities = []
        for i in range(args.mtd['num'] + 1):
            embedding_A = outputs['raw_features'] if i == 0 else outputs['raw_features_{}'.format(i)]
            embedding_A = F.normalize(embedding_A, p=2, dim=1)
            for j in range(i + 1, args.mtd['num'] + 1):
                embedding_B = outputs['raw_features_{}'.format(j)]
                embedding_B = F.normalize(embedding_B, p=2, dim=1)
                product = embedding_A * embedding_B
                sim = torch.abs(torch.sum(product, dim=1).mean())  # TODO
                # sim = torch.sum(product, dim=1).mean()
                similarities.append(sim)
        return sum(similarities) / len(similarities)

    def evaluate_(self):
        self.model.eval()
        if args.mode in ('DP', 'DDP'):
            self.model.module.branch_training(False)
        else:
            self.model.branch_training(False)
        scenario = memory.scenario_eval
        time.sleep(1)
        taskid = self.taskid
        progress = tqdm(range(1, taskid + 1), disable='slient' in args.opt)
        progress.set_description('eval_')
        results = {}
        DMatrix = np.zeros((args.mtd['num']+2, args.mtd['num']+2))
        task_size = OrderedDict()
        for t in progress:
            evaldata = scenario[t - 1]
            if args.dataset in ('PMNIST', 'FIVE') and hasattr(args, 'datafun'):
                evaldata = UnifiedDataset(evaldata['test']['x'],
                                          evaldata['test']['y'] + (get_minmax_class(t - 1)[1] + 1)
                                          if args.dataset == 'FIVE' else evaldata['train']['y'],
                                          [evaldata['t']] * len(evaldata['test']['x']),
                                          dataset=args.dataset)
            task_size[t] = len(evaldata)
            if args.mode == 'DDP':
                sampler = utils.data.DistributedSampler(evaldata, shuffle=False)
            else:
                sampler = None
            evalloader = DataLoader(evaldata, args.bs, shuffle=False, num_workers=args.workers, sampler=sampler)
            if args.scenario == 'class':
                minclass, maxclass = get_minmax_class(taskid)
            else:
                minclass, maxclass = get_minmax_class(t)
            targets, cnn_predicts = [], []
            with torch.no_grad():
                for x, y, _ in evalloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    targets.append(y)
                    outputs = self.model(x)
                    tmp = []
                    for index in range(args.mtd['num'] + 2):
                        if index == args.mtd['num'] + 1:
                            y_cnn = outputs['logits']
                        else:
                            y_cnn = outputs['logits_{}'.format(index)]
                        y_cnn, _ = activate_head(minclass, maxclass, y_cnn, y)
                        y_cnn = y_cnn.topk(k=1, dim=1)[1]
                        y_cnn = remap_label(y_cnn, minclass)
                        tmp.append(y_cnn)
                    cnn_predicts.append(torch.cat(tmp, dim=1))
            targets = torch.cat(targets, dim=0)
            cnn_predicts = torch.cat(cnn_predicts, dim=0)
            if args.mode == 'DDP':
                torch.distributed.barrier()
                targets = distributed_concat(targets)
                cnn_predicts = distributed_concat(cnn_predicts)
            targets = targets.unsqueeze(dim=1)
            match = targets.eq(cnn_predicts).cpu()
            cnn_top1_acc = match.sum(dim=0) / targets.shape[0]
            DMatrix += (match.T.sum(dim=1) - match.T.int() @ match.int()).numpy()
            for index, acc in enumerate(cnn_top1_acc):
                if index + 1 in results:
                    results[index + 1].append(round(acc.item(), 3))
                else:
                    results[index + 1] = [round(acc.item(), 3)]
        progress.close()
        DMatrix = DMatrix / taskid
        task_size = np.array(list(task_size.values()))
        info = '************* debug *************\n'
        for key, value in results.items():
            accs = []
            info += 'model-{} '.format(key)
            for index, value_ in enumerate(value):
                accs.append(value_)
                info += 'task{}:{} '.format(index + 1, value_)
            Acc = ((np.array(accs) * task_size).sum() / task_size.sum()).round(3)
            info += 'Acc:{}'.format(Acc)
            info += '\n'
        info += 'DMatrix:\n{}\n'.format(DMatrix)
        info += '************* debug *************'
        printlog(info)

    def evaluate(self, scenario):
        time_memory_snapshot('Task{} [end]:'.format(self.taskid), device=self.device)
        if hasattr(memory, 'AccMatrix_NME') is False:
            memory.AccMatrix_NME = np.zeros((args.tasks, args.tasks))
        if hasattr(memory, 'AccMatrix_ENS') is False:
            memory.AccMatrix_ENS = copy.deepcopy(memory.AccMatrix)
        self.model.eval()
        if args.mode in ('DP', 'DDP'):
            self.model.module.branch_training(flag=False)
            self.model.module.forward_with_branch = True
        else:
            self.model.branch_training(flag=False)
            self.model.forward_with_branch = True
        if args.base in ('PODNet', 'AFC', 'ANCL'):
            self.register_hook(module_name='classifier', variable_name='representations',
                               call_back=self.hook_abstract_call_back, side='input')
        elif args.base in ('LUCIR', 'SSIL', 'LwF'):
            self.register_hook(module_name='fc', variable_name='representations',
                               call_back=self.hook_abstract_call_back, side='input')
        time.sleep(1)
        taskid = self.taskid
        progress = tqdm(range(1, taskid + 1), disable='slient' in args.opt)
        progress.set_description('eval ')
        for t in progress:
            evaldata = scenario[t - 1]
            if args.dataset in ('PMNIST', 'FIVE') and hasattr(args, 'datafun'):
                evaldata = UnifiedDataset(evaldata['test']['x'],
                                          evaldata['test']['y'] + (get_minmax_class(t - 1)[1] + 1)
                                          if args.dataset == 'FIVE' else evaldata['train']['y'],
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
            targets, cnn_predicts, nme_predicts, ens_predicts = [], [], [], []
            with torch.no_grad():
                for x, y, _ in evalloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    outputs = self.model(x)
                    y_cnn = outputs['logits_0']
                    y_cnn, _ = activate_head(minclass, maxclass, y_cnn, y)
                    y_cnn = y_cnn.topk(k=1, dim=1)[1]
                    y_cnn = remap_label(y_cnn, minclass)
                    # NME classifier
                    representation = self.get_hook_result(variable_name='representations')
                    representation = torch.nn.functional.normalize(representation, p=2, dim=1)
                    y_nme = self.nme(representation)
                    y_nme = y_nme.topk(k=1, dim=1)[1]
                    y_nme = remap_label(y_nme, minclass)
                    # ENS
                    y_ens = outputs['logits']
                    y_ens, _ = activate_head(minclass, maxclass, y_ens, y)
                    y_ens = y_ens.topk(k=1, dim=1)[1]
                    y_ens = remap_label(y_ens, minclass)
                    targets.append(y)
                    cnn_predicts.append(y_cnn)
                    nme_predicts.append(y_nme)
                    ens_predicts.append(y_ens)
            targets = torch.cat(targets, dim=0)
            cnn_predicts = torch.cat(cnn_predicts, dim=0)
            nme_predicts = torch.cat(nme_predicts, dim=0)
            ens_predicts = torch.cat(ens_predicts, dim=0)
            if args.mode == 'DDP':
                torch.distributed.barrier()
                targets = distributed_concat(targets)
                cnn_predicts = distributed_concat(cnn_predicts)
                nme_predicts = distributed_concat(nme_predicts)
                ens_predicts = distributed_concat(ens_predicts)
            targets = targets.unsqueeze(dim=1)
            cnn_top1_acc = targets.eq(cnn_predicts[:, :1]).sum().item() / targets.shape[0]
            nme_top1_acc = targets.eq(nme_predicts[:, :1]).sum().item() / targets.shape[0]
            ens_top1_acc = targets.eq(ens_predicts[:, :1]).sum().item() / targets.shape[0]
            memory.AccMatrix[taskid - 1, t - 1] = cnn_top1_acc
            memory.AccMatrix_NME[taskid - 1, t - 1] = nme_top1_acc
            memory.AccMatrix_ENS[taskid - 1, t - 1] = ens_top1_acc
            # It is important to record sizes of tasks to compute AIAcc.
            memory.task_size[t] = targets.shape[0]
        progress.close()
        self.destroy_hook()
        if args.snapshot:
            save_memory(memory, self.taskid)
            save_model(self.model, self.taskid)
        print_log_metrics(memory.AccMatrix, taskid, remarks='CNN')
        print_log_metrics(memory.AccMatrix_NME, taskid, remarks='NME')
        print_log_metrics(memory.AccMatrix_ENS, taskid, remarks='ENS')
        time.sleep(1)
