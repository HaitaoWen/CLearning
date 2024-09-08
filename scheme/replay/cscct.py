from scheme.base import *
from copy import deepcopy
from scheme.replay.lucir import LUCIR
from scheme.replay.icarl import iCaRL
from scheme.replay.podnet.podnet import PODNet
import torch.nn.functional as F


if hasattr(args, 'base') and args.scheme == 'CSCCT':
    if args.base == 'iCaRL':
        BASE = iCaRL
    elif args.base == 'LUCIR':
        BASE = LUCIR
    elif args.base == 'PODNet':
        BASE = PODNet
    else:
        printlog('\033[1;30;41mPredetermined base {} does not exist, '
                 'use default base: LUCIR\033[0m'.format(args.scheme))
        BASE = iCaRL
else:
    BASE = iCaRL


class CSCCT(BASE):
    """
    Ashok A, Joseph K J, Balasubramanian V.
    Class-Incremental Learning with Cross-Space Clustering and Controlled Transfer[J].
    arXiv preprint arXiv:2208.03767, 2022.
    accepted at ECCV 2022.

    default baseline is LUCIR.
    part of code is borrowed from https://github.com/ashok-arjun/CSCCT.git
    """
    def __init__(self, model, traindata, taskid):
        super(CSCCT, self).__init__(model, traindata, taskid)

    def train(self):
        if args.base == 'iCaRL':
            self.icarl_train()
            self.construct_exemplar_set()
            self.construct_nme_classifier()
            pre_model = deepcopy(self.model)
            setattr(memory, 'pre_model', self.freeze_model(pre_model))

        elif args.base == 'LUCIR':
            self.lucir_train()
            self.construct_exemplar_set()
            self.cbf_finetune()
            self.construct_nme_classifier()
            pre_model = deepcopy(self.model)
            setattr(memory, 'pre_model', self.freeze_model(pre_model))

        elif args.base == 'PODNet':
            self.podnet_train()
            self.construct_exemplar_set()
            self.cbf_finetune()
            self.construct_nme_classifier()
            pre_model = deepcopy(self.model)
            setattr(memory, 'pre_model', self.freeze_model(pre_model))

        return self.model

    def icarl_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt)
        self.progress.set_description('train')
        self.register_hook(self.model, module_name='fc', variable_name='cur_representations',
                           call_back=self.hook_abstract_call_back)
        self.register_hook(memory.pre_model, module_name='fc', variable_name='pre_representations',
                           call_back=self.hook_abstract_call_back)
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

                    mask = t == self.taskid
                    cur_representations = self.get_hook_result(self.model, variable_name='cur_representations')
                    pre_representations = self.get_hook_result(memory.pre_model, variable_name='pre_representations')
                    loss_csc = self.cross_space_cluster_loss(cur_representations, pre_representations, y) * args.csc
                    loss__ct = self.control_transfer_loss(cur_representations, pre_representations, mask) * args.ct

                    TBWriter.add_scalar('Loss_CE', loss.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_KL', loss_div.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_CSC', loss_csc.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_CT', loss__ct.detach().cpu().data.numpy(), memory.iteration)

                    memory.iteration += 1

                    loss = loss + loss_div + loss_csc + loss__ct

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

                    mask = t == self.taskid
                    loss_csc = self.cross_space_cluster_loss(cur_representations, pre_representations, y) * args.csc
                    loss__ct = self.control_transfer_loss(cur_representations, pre_representations, mask) * args.ct

                    TBWriter.add_scalar('Loss_CE', loss.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_EB', loss_embed.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_MR', loss_mr.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_CSC', loss_csc.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_CT', loss__ct.detach().cpu().data.numpy(), memory.iteration)

                    memory.iteration += 1

                    loss = loss + loss_embed + loss_mr + loss_csc + loss__ct

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

    def podnet_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt)
        self.progress.set_description('train')
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                loss = self.nca_loss(outputs["logits"], y, scale=self.model.post_processor.factor,
                                     margin=args.nca['margin'],
                                     exclude_pos_denominator=args.nca['exclude_pos_denominator'])
                if self.taskid > 1:
                    old_outputs = memory.pre_model(x)
                    old_features = old_outputs["raw_features"]

                    loss_embed = F.cosine_embedding_loss(outputs["raw_features"], old_features.detach(),
                                                         torch.ones(old_features.shape[0]).to(self.device)
                                                         ) * self.flat_factor

                    loss_pod = self.pod_loss(old_outputs["attention"], outputs["attention"],
                                             **args.pod_spatial) * self.pod_factor

                    TBWriter.add_scalar('Loss_NCA', loss.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_EB', loss_embed.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_POD', loss_pod.detach().cpu().data.numpy(), memory.iteration)
                    memory.iteration += 1

                    loss = loss + loss_embed + loss_pod

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

    @staticmethod
    def similarity_matrix(a, b, eps=1e-8):
        """
        Batch cosine similarity taken from https://stackoverflow.com/a/58144658/10425618
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def cross_space_cluster_loss(self, cur_representations, pre_representations, targets):
        targets_unsqueezed = targets.unsqueeze(1)
        indexes = (targets_unsqueezed == targets_unsqueezed.T).to(torch.int)
        indexes[indexes == 0] = -1
        computed_similarity = self.similarity_matrix(cur_representations, pre_representations).flatten()
        csc_loss = 1 - computed_similarity
        csc_loss *= indexes.flatten()
        csc_loss = csc_loss.mean()
        return csc_loss

    def control_transfer_loss(self, cur_representations, pre_representations, mask):
        if mask.all() or (~mask).all():
            return torch.zeros(1).to(self.device)
        ref_features_curtask = pre_representations[mask]
        ref_features_prevtask = pre_representations[~mask]
        cur_features_curtask = cur_representations[mask]
        cur_features_prevtask = cur_representations[~mask]
        previous_model_similarities = self.similarity_matrix(ref_features_curtask, ref_features_prevtask)
        current_model_similarities = self.similarity_matrix(cur_features_curtask, cur_features_prevtask)
        ct_loss = F.kl_div(F.log_softmax(current_model_similarities / args.ct_tau, dim=1),
                           F.softmax(previous_model_similarities / args.ct_tau, dim=1),
                           reduction='batchmean') / current_model_similarities.shape[1] * (args.ct_tau ** 2)
        return ct_loss
