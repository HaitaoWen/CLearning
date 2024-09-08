import math
from copy import deepcopy
from scheme.base import *
from scheme.replay.icarl import iCaRL
import torch.nn.functional as F
from .utils import add_new_weights


class PODNet(iCaRL):
    """Pooled Output Distillation Network.
    # Reference:
        * Small Task Incremental Learning
          Douillard et al. 2020
    """
    def __init__(self, model, traindata, taskid):
        super(PODNet, self).__init__(model, traindata, taskid)
        # TODO, the original toolkit is not designed
        #  for DP/DDP, so we de-parallelization first.
        self.deparallelization()
        self.modify_traindata_transform()
        add_new_weights(
            self.model, args.weight_generation if self.taskid > 1 else "basic",
            sum(args.increments[:self.taskid-1]), args.increments[self.taskid-1], traindata
        )
        self.init_optimizer_custom()
        self.allocate_model()

    def train(self):
        time_memory_snapshot('Train [begin]:', device=self.device, reset=True)
        self.podnet_train()
        time_memory_snapshot('Train [end]:', device=self.device, reset=True)
        self.construct_exemplar_set()
        time_memory_snapshot('CBF [begin]:', device=self.device)
        self.cbf_finetune()
        time_memory_snapshot('CBF [end]:', device=self.device)
        self.construct_nme_classifier()
        pre_model = deepcopy(self.model)
        setattr(memory, 'pre_model', self.freeze_model(pre_model))
        return self.model

    def podnet_train(self):
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

                    TBWriter.add_scalar('Loss_NCA', loss.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_FLAT', loss_flat.detach().cpu().data.numpy(), memory.iteration)
                    TBWriter.add_scalar('Loss_SPATIAL', loss_spatial.detach().cpu().data.numpy(), memory.iteration)
                    memory.iteration += 1

                    loss = loss + loss_flat + loss_spatial

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
    def nca_loss(similarities, targets, class_weights=None,
                 focal_gamma=None, scale=1, margin=0.,
                 exclude_pos_denominator=True, hinge_proxynca=False,
                 memory_flags=None,):
        """Compute AMS cross-entropy loss.

        Reference:
            * Goldberger et al.
              Neighbourhood components analysis.
              NeuriPS 2005.
            * Feng Wang et al.
              Additive Margin Softmax for Face Verification.
              Signal Processing Letters 2018.

        :param similarities: Result of cosine similarities between weights and features.
        :param targets: Sparse targets.
        :param scale: Multiplicative factor, can be learned.
        :param margin: Margin applied on the "right" (numerator) similarities.
        :param memory_flags: Flags indicating memory samples, although it could indicate
                             anything else.
        :return: A float scalar loss.

        Args:
            hinge_proxynca:
            hinge_proxynca:
        """
        margins = torch.zeros_like(similarities)
        margins[torch.arange(margins.shape[0]), targets] = margin
        similarities = scale * (similarities - margin)

        if exclude_pos_denominator:  # NCA-specific
            similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

            disable_pos = torch.zeros_like(similarities)
            disable_pos[torch.arange(len(similarities)),
                        targets] = similarities[torch.arange(len(similarities)), targets]

            numerator = similarities[torch.arange(similarities.shape[0]), targets]
            denominator = similarities - disable_pos

            losses = numerator - torch.log(torch.exp(denominator).sum(-1))
            if class_weights is not None:
                losses = class_weights[targets] * losses

            losses = -losses
            if hinge_proxynca:
                losses = torch.clamp(losses, min=0.)

            loss = torch.mean(losses)
            return loss

        return F.cross_entropy(similarities, targets, weight=class_weights, reduction="mean")

    @staticmethod
    def pod_loss(list_attentions_a, list_attentions_b,
                 collapse_channels="spatial", normalize=True,
                 memory_flags=None, only_old=False,
                 feature_distil_factor=None, **kwargs):
        """Pooled Output Distillation.

        Reference:
            * Douillard et al.
              Small Task Incremental Learning.
              arXiv 2020.

        :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
        :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
        :param collapse_channels: How to pool the channels.
        :param memory_flags: Integer flags denoting exemplars.
        :param only_old: Only apply loss to exemplars.
        :return: A float scalar loss.
        """
        assert len(list_attentions_a) == len(list_attentions_b)

        loss = torch.tensor(0.).to(list_attentions_a[0].device)
        for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
            # shape of (b, n, w, h)
            assert a.shape == b.shape, (a.shape, b.shape)

            if only_old:
                a = a[memory_flags]
                b = b[memory_flags]
                if len(a) == 0:
                    continue

            a = torch.pow(a, 2)
            b = torch.pow(b, 2)

            if collapse_channels == "channels":
                a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
                b = b.sum(dim=1).view(b.shape[0], -1)
            elif collapse_channels == "width":
                a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
                b = b.sum(dim=2).view(b.shape[0], -1)
            elif collapse_channels == "height":
                a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
                b = b.sum(dim=3).view(b.shape[0], -1)
            elif collapse_channels == "gap":
                a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
                b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
            elif collapse_channels == "spatial":
                a_h = a.sum(dim=3).view(a.shape[0], -1)
                b_h = b.sum(dim=3).view(b.shape[0], -1)
                a_w = a.sum(dim=2).view(a.shape[0], -1)
                b_w = b.sum(dim=2).view(b.shape[0], -1)
                a = torch.cat([a_h, a_w], dim=-1)
                b = torch.cat([b_h, b_w], dim=-1)
            elif collapse_channels == "pixel":
                a = a.view(a.shape[0], a.shape[1], -1)
                b = b.view(b.shape[0], b.shape[1], -1)
            else:
                raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

            if normalize:
                if collapse_channels == "pixel":
                    a = F.normalize(a, dim=2, p=2)
                    b = F.normalize(b, dim=2, p=2)
                else:
                    a = F.normalize(a, dim=1, p=2)
                    b = F.normalize(b, dim=1, p=2)

            if feature_distil_factor is None:
                layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
            else:
                factor = feature_distil_factor[i].reshape([1, -1])
                layer_loss = torch.mean(factor * torch.frobenius_norm(a - b, dim=-1))
            loss += layer_loss

        return loss / len(list_attentions_a)

    def init_optimizer_custom(self):
        if args.groupwise_factors and isinstance(args.groupwise_factors, dict):
            groupwise_factor = args.groupwise_factors

            params = []
            for group_name, group_params in self.model.get_group_parameters().items():
                if group_params is None or group_name == "last_block":
                    continue
                factor = groupwise_factor.get(group_name, 1.0)
                if factor == 0.:
                    continue
                params.append({"params": group_params, "lr": args.lr * factor})
                printlog(f"Group: {group_name}, lr: {args.lr * factor}.")
        else:
            params = self.model.parameters()
        self.optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.decay,
                                         momentum=args.momentum)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs)
        # ******************** clamp gradient ******************** #
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

    def modify_traindata_transform(self):
        self.traindata.trsf.transforms.insert(2, transforms.ColorJitter(brightness=63 / 255))

    def cbf_finetune(self):
        if self.taskid == 1 or not hasattr(args, 'finetuning_config'):
            return
        printlog('class-balanced finetuning', delay=1.0)
        # ****************** load memory data ****************** #
        trsf = self.traindata.trsf
        if args.dataset in ('CIFAR10', 'CIFAR100', 'SVHN'):
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
        self.optimizer = torch.optim.SGD(self.model.module.classifier.parameters()
                                         if args.mode in ('DP', 'DDP') else self.model.classifier.parameters(),
                                         lr=args.finetuning_config['lr'],
                                         weight_decay=args.decay, momentum=args.momentum)
        self.scheduler = None
        # ************************** overwrite configs ************************** #
        args_copy = deepcopy(args)
        args.epochs = args.finetuning_config['epochs']
        # ************************** adjust model state ************************** #
        # it is consistent with the original implementation
        self.model.train()
        # ********************** multiplexing self.lucir_train ********************** #
        self.podnet_train()
        args.__dict__.update(args_copy.__dict__)

    def construct_exemplar_set(self):
        # Herding is adopted by default
        self.model.eval()
        # trsf = self.traindata.trsf
        _x, _y = self.traindata._x, self.traindata._y
        # register hook to get representations
        hook_handle = self.register_hook(module_name='classifier', variable_name='representations',
                                         call_back=self.hook_abstract_call_back, side='input')
        x, y, t = self.herding_sample(_x, _y, transforms.Compose(Transforms[args.dataset]['eval']))
        self.destroy_hook(hooks=hook_handle)
        if args.dataset in ('CIFAR10', 'CIFAR100', 'SVHN'):
            x = np.stack(x, axis=0)
            y = np.stack(y, axis=0)
            t = np.stack(t, axis=0)
            memory.x = np.concatenate((memory.x, x), axis=0) if memory.x is not None else x
            memory.y = np.concatenate((memory.y, y), axis=0) if memory.y is not None else y
            memory.t = np.concatenate((memory.t, t), axis=0) if memory.t is not None else t
        elif 'ImageNet' in args.dataset or args.dataset in ('DTD', 'CUB'):
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

    def construct_nme_classifier(self):
        self.model.eval()
        hook_handle = self.register_hook(module_name='classifier', variable_name='representations',
                                         call_back=self.hook_abstract_call_back, side='input')
        if args.mode in ('DP', 'DDP'):
            self.nme = torch.nn.Linear(self.model.module.classifier.features_dim, len(np.unique(memory.y)),
                                       bias=False, device=self.device)
        else:
            self.nme = torch.nn.Linear(self.model.classifier.features_dim, len(np.unique(memory.y)), bias=False,
                                       device=self.device)
        if args.dataset in ('CIFAR10', 'CIFAR100', 'SVHN'):
            memdata = Memorydata(memory, transforms.Compose(Transforms[args.dataset]['eval']))
        elif 'ImageNet' in args.dataset or args.dataset in ('DTD', 'CUB'):
            memdata = ImageNetData(memory.x, memory.y, memory.t, transforms.Compose(Transforms[args.dataset]['eval']))
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

        # flipped
        representations_flipped = []
        memdata.transform.transforms.insert(0, transforms.RandomHorizontalFlip(p=1.))
        with torch.no_grad():
            for x, _, _ in loader:
                x = x.to(self.device)
                self.model(x)
                representations_flipped.append(self.get_hook_result(variable_name='representations'))
        representations_flipped = torch.cat(representations_flipped, dim=0)
        if args.mode == 'DDP':
            torch.distributed.barrier()
            representations_flipped = distributed_concat(representations_flipped)
        representations_flipped = torch.nn.functional.normalize(representations_flipped, p=2, dim=1)

        for label in labels.unique():
            mask = labels == label
            representation = representations[mask]
            representation_flipped = representations_flipped[mask]
            representation_mean = representation.mean(dim=0)
            representation_mean_flipped = representation_flipped.mean(dim=0)
            representation_mean = (representation_mean + representation_mean_flipped) / 2
            representation_mean = torch.nn.functional.normalize(representation_mean, p=2, dim=0)
            self.nme.weight.data[label] = representation_mean
        self.destroy_hook(hooks=hook_handle)

    def evaluate(self, scenario):
        time_memory_snapshot('Task{} [end]:'.format(self.taskid), device=self.device)
        if hasattr(memory, 'AccMatrix_NME') is False:
            memory.AccMatrix_NME = np.zeros((args.tasks, args.tasks))
        self.model.eval()
        self.register_hook(module_name='classifier', variable_name='representations',
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
                    outputs = self.model(x)
                    y_cnn = outputs['logits']
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
