import math
from scheme.base import *
from copy import deepcopy
import torch.nn.functional as F
from scheme.replay.podnet.podnet import PODNet


if hasattr(args, 'base') and args.scheme == 'ANCL':
    if args.base == 'PODNet':
        BASE = PODNet
    else:
        printlog('\033[1;30;41mPredetermined base {} does not exist, '
                 'use default base: PODNet\033[0m'.format(args.scheme))
        BASE = PODNet
else:
    BASE = PODNet


class ANCL(BASE):
    """
    Kim S, Noci L, Orvieto A, et al.
    Achieving a Better Stability-Plasticity Trade-off via Auxiliary Networks in Continual Learning[C]
    //Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
    2023: 11930-11939.
    """
    def __init__(self, model, traindata, taskid):
        super(ANCL, self).__init__(model, traindata, taskid)
        if self.taskid == 1:
            return
        self.aux_optimizer = None
        self.aux_scheduler = None
        self.aux_model = deepcopy(self.model)
        self.init_aux_optimizer()

    def train(self):
        if args.base == 'PODNet':
            if self.taskid > 1:
                self.aux_model_train()
                self.aux_model = self.freeze_model(self.aux_model)
            self.ancl_train()
            self.construct_exemplar_set()
            self.cbf_finetune()
            self.construct_nme_classifier()
            pre_model = deepcopy(self.model)
            setattr(memory, 'pre_model', self.freeze_model(pre_model))
            return self.model
        else:
            raise NotImplementedError

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

                    TBWriter.add_scalar('Loss_NCA', loss.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_FLAT', loss_flat.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_SPATIAL', loss_spatial.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_FLAT_AUX', loss_flat_aux.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_SPATIAL_AUX', loss_spatial_aux.detach().cpu().data.numpy(), memory.iteration)
                    memory.iteration += 1

                    loss = loss + (loss_flat + loss_spatial) * args.ANCL['lambda'] \
                                + (loss_flat_aux + loss_spatial_aux) * args.ANCL['lambda_a']

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

    def aux_model_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt)
        self.progress.set_description('aux  ')
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.aux_model(x)
                loss = self.nca_loss(outputs["logits"], y,
                                     scale=self.aux_model.module.post_processor.factor
                                     if args.mode in ('DP', 'DDP') else self.aux_model.post_processor.factor,
                                     margin=args.nca['margin'],
                                     exclude_pos_denominator=args.nca['exclude_pos_denominator'])
                self.aux_optimizer.zero_grad()
                loss.backward()
                self.aux_optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.aux_optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item()})
                self.progress.update(1)
            if self.aux_scheduler is not None:
                self.aux_scheduler.step()
        self.progress.close()

    def init_aux_optimizer(self):
        if args.groupwise_factors and isinstance(args.groupwise_factors, dict):
            groupwise_factor = args.groupwise_factors

            params = []
            for group_name, group_params in self.aux_model.get_group_parameters().items():
                if group_params is None or group_name == "last_block":
                    continue
                factor = groupwise_factor.get(group_name, 1.0)
                if factor == 0.:
                    continue
                params.append({"params": group_params, "lr": args.lr * factor})
                printlog(f"Group: {group_name}, lr: {args.lr * factor}.")
        else:
            params = self.aux_model.parameters()
        self.aux_optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.decay,
                                             momentum=args.momentum)
        self.aux_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.aux_optimizer, args.epochs)
        # ******************** clamp gradient ******************** #
        for p in self.aux_model.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))
