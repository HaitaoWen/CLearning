from scheme.base import *
from .curve import *
from .rosen import Rosen
from copy import deepcopy
import torch.nn.functional as F
from scheme.replay.podnet.podnet import PODNet
from scheme.structure.aanet.aanet import AANet


if hasattr(args, 'base') and args.scheme == 'OPC':
    if args.base == 'PODNet':
        BASE = PODNet
    elif args.base == 'AANet':
        BASE = AANet
    else:
        printlog('\033[1;30;41mPredetermined base {} does not exist, '
                 'use default base: PODNet\033[0m'.format(args.scheme))
        BASE = PODNet
else:
    BASE = PODNet


class OPC(BASE):
    def __init__(self, model, traindata, taskid):
        super(OPC, self).__init__(model, traindata, taskid)
        self.eps = 1e-5
        self.stop = None
        self.curve = None
        self.start = None
        self.loader = None
        self.fast_state = None
        self.paramkeys = None
        if args.base in ('PODNet') and self.taskid > 1:
            self.imprinted_weights = self.model.module.classifier._weights[1].data.clone().detach() \
                if args.mode in ('DP', 'DDP') else self.model.classifier._weights[1].data.clone().detach()

    def train(self):
        if args.base == 'PODNet':
            self.podnet_train()
            self.construct_exemplar_set()
            self.cbf_finetune()
            if self.taskid > 1:
                if args.opc['curve'] in ['PolygonalChain', 'BezierCurve', 'TOrderBezierCurve']:
                    self.garipov_postprocessing_podnet()
                elif args.opc['curve'] in ['SimplicialComplex']:
                    self.simplicial_complex_postprocessing_podnet()
                else:
                    self.opc_postprocessing_podnet()
                self.ensembling()
                self.drop_curtask_exemplar()
                self.construct_exemplar_set()
            self.construct_nme_classifier()
            pre_model = deepcopy(self.model)
            setattr(memory, 'pre_model', self.freeze_model(pre_model))

        elif args.base == 'AANet':
            self.podnet_train()
            self.construct_exemplar_set()
            self.cbf_finetune()
            if self.taskid > 2:
                self.opc_postprocessing_podnet()
                self.ensembling()
                self.drop_curtask_exemplar()
                self.construct_exemplar_set()
            self.construct_nme_classifier()
            pre_model = deepcopy(self.model)
            setattr(memory, 'pre_model', self.freeze_model(pre_model))

        return self.model

    def opc_postprocessing_podnet(self):
        printlog('{} with EOPC'.format(args.base))
        # ************************ prepare data ************************ #
        trsf = self.traindata.trsf
        if args.dataset == 'CIFAR100':
            memdata = Memorydata(memory, trsf)
        elif 'ImageNet' in args.dataset:
            memdata = ImageNetData(memory.x, memory.y, memory.t, trsf)
        else:
            raise ValueError
        if args.mode == 'DDP':
            sampler = utils.data.DistributedSampler(memdata, shuffle=True)
            self.loader = DataLoader(memdata, args.bs, num_workers=args.workers, sampler=sampler)
        else:
            self.loader = DataLoader(memdata, args.bs, num_workers=args.workers, shuffle=True)

        # ************************ optimize curve ************************ #
        self.model.eval()
        self.paramkeys = self.param_keys()
        self.fast_state = self.detach_state()
        self.start = self.state_detach(memory.pre_model.state_dict())
        self.stop = self.state_detach(self.model.state_dict())
        self.start = self.podnet_add_parameters()

        if args.opc['curve'] == 'Line':
            self.curve = Line(self.filter_params(self.start),
                              self.filter_params(self.stop)).to(self.device)
        else:
            # assert args.opc['lr'] > 0  # learning rate of curve
            # assert args.opc['order'] > 0
            # assert args.opc['boundary'] > 0
            self.model.eval()
            hook_handle = self.register_hook(module_name='classifier', variable_name='representations',
                                             call_back=self.hook_abstract_call_back, side='input')
            self.curve = eval(args.opc['curve'])(self.filter_params(self.start),
                                                 self.filter_params(self.stop),
                                                 args.opc['order'],
                                                 args.opc['boundary']).to(self.device)
            if args.mode == 'DDP' and not isinstance(self.curve, nn.parallel.DistributedDataParallel):
                self.curve = nn.parallel.DistributedDataParallel(self.curve, device_ids=[args.local_rank],
                                                                 output_device=args.local_rank)
            optimizer = Rosen(list(self.curve.parameters()), lr=args.opc['lr'],
                              boundary=args.opc['boundary'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.opc['steps'],
                                                             gamma=args.opc['gamma'])

            progress = tqdm(range(1, len(self.loader) * args.opc['epochs'] + 1),
                            disable='slient' in args.opt)
            progress.set_description('OPC  ')
            losses = []
            for epoch in range(int(args.opc['epochs'])):
                for x, y, t in self.loader:
                    t = t + 1
                    x = x.to(self.device)
                    y = y.to(self.device)
                    mask = t == self.taskid

                    losssum = 0
                    optimizer.zero_grad()
                    factors = np.linspace(0.1, 0.95, args.opc['points'])
                    if 'add' in args and args.add > 0:
                        factors_ = np.linspace(args.opc['cut_off']-args.ens['gap']/2,
                                               args.opc['cut_off']+args.ens['gap']/2, args.add)
                        factors = np.append(factors, factors_, axis=0)
                    for factor in factors:
                        index = 0
                        tmp_parameters = self.curve.forward(factor)
                        tmp_tangent = [p.data.view(-1) for p in self.curve.tangent(factor)]
                        tmp_tangent = torch.cat(tmp_tangent, dim=0)
                        noise = torch.randn(tmp_tangent.numel(), requires_grad=False).to(self.device)
                        noise = noise - (noise * tmp_tangent).sum() / (
                                    torch.norm(tmp_tangent).pow(2) + self.eps) * tmp_tangent
                        noise = F.normalize(noise, p=2, dim=0) * args.opc['radius']
                        begin, end = 0, 0
                        for key, value in self.fast_state.items():
                            if key in self.paramkeys:
                                begin = end
                                value_ = tmp_parameters[index]
                                end += value_.numel()
                                self.fast_state[key] = value_ + noise[begin: end].reshape(value.shape)
                                index += 1
                            else:
                                self.fast_state[key] = self.start[key] * (1 - factor) + self.stop[key] * factor
                        self.model.zero_grad()

                        # current
                        with torch.no_grad():
                            if factor < args.opc['cut_off']:
                                # previous
                                self.model.load_state_dict(self.start)
                                pre_output = self.model(x)
                                pre_representations = self.get_hook_result(model=self.model,
                                                                           variable_name='representations').detach().clone()
                            else:
                                self.model.load_state_dict(self.stop)
                                cur_output = self.model(x)
                                cur_representations = self.get_hook_result(model=self.model,
                                                                           variable_name='representations').detach().clone()

                        self.model.load_state_dict(self.fast_state)
                        tmp_output = self.model(x)
                        tmp_representations = self.get_hook_result(model=self.model,
                                                                   variable_name='representations').clone()

                        loss = 0
                        if factor < args.opc['cut_off']:
                            # classification loss
                            if 'ce' in args.opc['loss']:
                                loss = loss + torch.nn.functional.cross_entropy(tmp_output['logits'][~mask], y[~mask])

                            if 'nca' in args.opc['loss']:
                                loss = loss + self.nca_loss(tmp_output['logits'][~mask], y[~mask],
                                                            scale=self.model.module.post_processor.factor
                                                            if args.mode in (
                                                            'DP', 'DDP') else self.model.post_processor.factor,
                                                            margin=args.nca['margin'],
                                                            exclude_pos_denominator=args.nca['exclude_pos_denominator'])

                            if 'balance' in args.opc and args.opc['balance']:
                                loss = loss * (abs(factor - 0.5) + args.opc['balance'])

                            # embedding loss
                            if 'embed' in args.opc['loss']:
                                loss = loss + F.cosine_embedding_loss(tmp_representations, pre_representations,
                                                                      torch.ones(x.shape[0]).to(self.device))

                            if 'pod' in args.opc['loss']:
                                if args.base == 'PODNet':
                                    pod_factor = args.pod_spatial["scheduled_factor"] * math.sqrt(
                                        sum(args.increments[:self.taskid])
                                        / args.increments[self.taskid - 1])
                                    loss = loss + self.pod_loss(pre_output["attention"], tmp_output["attention"],
                                                                **args.pod_spatial) * pod_factor
                        else:
                            # classification loss
                            if 'ce' in args.opc['loss']:
                                loss = loss + torch.nn.functional.cross_entropy(tmp_output['logits'], y)

                            if 'nca' in args.opc['loss']:
                                loss = loss + self.nca_loss(tmp_output['logits'], y,
                                                            scale=self.model.module.post_processor.factor
                                                            if args.mode in (
                                                                'DP', 'DDP') else self.model.post_processor.factor,
                                                            margin=args.nca['margin'],
                                                            exclude_pos_denominator=args.nca['exclude_pos_denominator'])

                            if 'balance' in args.opc and args.opc['balance']:
                                loss = loss * (abs(factor - 0.5) + args.opc['balance'])

                            # embedding loss
                            if 'embed' in args.opc['loss']:
                                loss = loss + F.cosine_embedding_loss(tmp_representations, cur_representations,
                                                                      torch.ones(x.shape[0]).to(self.device))

                            if 'pod' in args.opc['loss']:
                                if args.base == 'PODNet':
                                    pod_factor = args.pod_spatial["scheduled_factor"] * math.sqrt(
                                        sum(args.increments[:self.taskid])
                                        / args.increments[self.taskid - 1])
                                    loss = loss + self.pod_loss(cur_output["attention"], tmp_output["attention"],
                                                                **args.pod_spatial) * pod_factor

                        loss.backward()
                        losssum += loss.item()
                        for p, (n, p_) in zip(tmp_parameters, self.model.named_parameters()):
                            if p_.grad is not None:
                                p.backward(p_.grad / args.opc['points'], retain_graph=True)

                    optimizer.step()
                    losssum = losssum / args.opc['points']
                    losses.append(losssum)
                    progress.set_postfix({'epoch': epoch,
                                          'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                                          'loss': losssum})
                    progress.update(1)
                scheduler.step()
            progress.close()
            self.destroy_hook(hooks=hook_handle)

            if 'VIS' in args.opc and args.opc['VIS']:
                png_root = args.logdir + args.name + '/png'
                if not os.path.exists(png_root):
                    os.makedirs(png_root)
                import matplotlib.pyplot as plt
                plt.plot(losses)
                plt.savefig(png_root + '/loss{}.png'.format(self.taskid))
                plt.close()

    def garipov_postprocessing_podnet(self):
        printlog('{} with {}'.format(args.base, args.opc['curve']))
        # ************************ prepare data ************************ #
        trsf = self.traindata.trsf
        if args.dataset == 'CIFAR100':
            memdata = Memorydata(memory, trsf)
        elif 'ImageNet' in args.dataset:
            memdata = ImageNetData(memory.x, memory.y, memory.t, trsf)
        else:
            raise ValueError
        if args.mode == 'DDP':
            sampler = utils.data.DistributedSampler(memdata, shuffle=True)
            self.loader = DataLoader(memdata, args.bs, num_workers=args.workers, sampler=sampler)
        else:
            self.loader = DataLoader(memdata, args.bs, num_workers=args.workers, shuffle=True)

        # ************************ optimize curve ************************ #
        self.model.eval()
        self.paramkeys = self.param_keys()
        self.fast_state = self.detach_state()
        self.start = self.state_detach(memory.pre_model.state_dict())
        self.stop = self.state_detach(self.model.state_dict())
        self.start = self.podnet_add_parameters()

        # assert args.opc['lr'] > 0  # learning rate of curve
        # assert args.opc['order'] > 0
        # assert args.opc['boundary'] > 0
        self.model.eval()
        hook_handle = self.register_hook(module_name='classifier', variable_name='representations',
                                         call_back=self.hook_abstract_call_back, side='input')
        self.curve = eval(args.opc['curve'])(self.filter_params(self.start),
                                             self.filter_params(self.stop),
                                             lambd=args.opc['cut_off']).to(self.device)
        if args.mode == 'DDP' and not isinstance(self.curve, nn.parallel.DistributedDataParallel):
            self.curve = nn.parallel.DistributedDataParallel(self.curve, device_ids=[args.local_rank],
                                                             output_device=args.local_rank)
        optimizer = torch.optim.SGD(self.curve.bend_parameters, lr=args.opc['lr'], weight_decay=args.opc['decay'],
                                    momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.opc['steps'],
                                                         gamma=args.opc['gamma'])

        progress = tqdm(range(1, len(self.loader) * args.opc['epochs'] + 1),
                        disable='slient' in args.opt)
        progress.set_description('OPC  ')
        losses = []
        for epoch in range(int(args.opc['epochs'])):
            for x, y, t in self.loader:
                t = t + 1
                x = x.to(self.device)
                y = y.to(self.device)
                mask = t == self.taskid

                losssum = 0
                optimizer.zero_grad()
                factors = np.linspace(0.1, 0.95, args.opc['points'])
                for factor in factors:
                    index = 0
                    tmp_parameters = self.curve.forward(factor)
                    # tmp_tangent = [p.data.view(-1) for p in self.curve.tangent(factor)]
                    # tmp_tangent = torch.cat(tmp_tangent, dim=0)
                    # noise = torch.randn(tmp_tangent.numel(), requires_grad=False).to(self.device)
                    # noise = noise - (noise * tmp_tangent).sum() / (torch.norm(tmp_tangent).pow(2) + self.eps) * tmp_tangent
                    # noise = F.normalize(noise, p=2, dim=0) * args.opc['radius']
                    begin, end = 0, 0
                    for key, value in self.fast_state.items():
                        if key in self.paramkeys:
                            begin = end
                            value_ = tmp_parameters[index]
                            end += value_.numel()
                            self.fast_state[key] = value_  # + noise[begin: end].reshape(value.shape)
                            index += 1
                        else:
                            self.fast_state[key] = self.start[key] * (1 - factor) + self.stop[key] * factor
                    self.model.zero_grad()

                    # current
                    with torch.no_grad():
                        if factor < args.opc['cut_off']:
                            # previous
                            self.model.load_state_dict(self.start)
                            pre_output = self.model(x)
                            pre_representations = self.get_hook_result(model=self.model,
                                                                       variable_name='representations').detach().clone()
                        else:
                            self.model.load_state_dict(self.stop)
                            cur_output = self.model(x)
                            cur_representations = self.get_hook_result(model=self.model,
                                                                       variable_name='representations').detach().clone()

                    self.model.load_state_dict(self.fast_state)
                    tmp_output = self.model(x)
                    tmp_representations = self.get_hook_result(model=self.model,
                                                               variable_name='representations').clone()

                    loss = 0
                    if factor < args.opc['cut_off']:
                        # classification loss
                        if 'ce' in args.opc['loss']:
                            loss = loss + torch.nn.functional.cross_entropy(tmp_output['logits'][~mask], y[~mask])

                        if 'nca' in args.opc['loss']:
                            loss = loss + self.nca_loss(tmp_output['logits'][~mask], y[~mask],
                                                        scale=self.model.module.post_processor.factor
                                                        if args.mode in (
                                                        'DP', 'DDP') else self.model.post_processor.factor,
                                                        margin=args.nca['margin'],
                                                        exclude_pos_denominator=args.nca['exclude_pos_denominator'])

                        if 'balance' in args.opc and args.opc['balance']:
                            loss = loss * (abs(factor - 0.5) + args.opc['balance'])

                        # embedding loss
                        if 'embed' in args.opc['loss']:
                            loss = loss + F.cosine_embedding_loss(tmp_representations, pre_representations,
                                                                  torch.ones(x.shape[0]).to(self.device))

                        if 'pod' in args.opc['loss']:
                            if args.base == 'PODNet':
                                pod_factor = args.pod_spatial["scheduled_factor"] * math.sqrt(
                                    sum(args.increments[:self.taskid])
                                    / args.increments[self.taskid - 1])
                                loss = loss + self.pod_loss(pre_output["attention"], tmp_output["attention"],
                                                            **args.pod_spatial) * pod_factor
                    else:
                        # classification loss
                        if 'ce' in args.opc['loss']:
                            loss = loss + torch.nn.functional.cross_entropy(tmp_output['logits'], y)

                        if 'nca' in args.opc['loss']:
                            loss = loss + self.nca_loss(tmp_output['logits'], y,
                                                        scale=self.model.module.post_processor.factor
                                                        if args.mode in (
                                                            'DP', 'DDP') else self.model.post_processor.factor,
                                                        margin=args.nca['margin'],
                                                        exclude_pos_denominator=args.nca['exclude_pos_denominator'])

                        if 'balance' in args.opc and args.opc['balance']:
                            loss = loss * (abs(factor - 0.5) + args.opc['balance'])

                        # embedding loss
                        if 'embed' in args.opc['loss']:
                            loss = loss + F.cosine_embedding_loss(tmp_representations, cur_representations,
                                                                  torch.ones(x.shape[0]).to(self.device))

                        if 'pod' in args.opc['loss']:
                            if args.base == 'PODNet':
                                pod_factor = args.pod_spatial["scheduled_factor"] * math.sqrt(
                                    sum(args.increments[:self.taskid])
                                    / args.increments[self.taskid - 1])
                                loss = loss + self.pod_loss(cur_output["attention"], tmp_output["attention"],
                                                            **args.pod_spatial) * pod_factor
                    loss.backward()
                    losssum += loss.item()
                    for p, (n, p_) in zip(tmp_parameters, self.model.named_parameters()):
                        if p_.grad is not None:
                            p.backward(p_.grad / args.opc['points'], retain_graph=True)

                optimizer.step()
                losssum = losssum / args.opc['points']
                losses.append(losssum)
                progress.set_postfix({'epoch': epoch,
                                      'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                                      'loss': losssum})
                progress.update(1)
            scheduler.step()
        progress.close()
        self.destroy_hook(hooks=hook_handle)

        if 'VIS' in args.opc and args.opc['VIS']:
            png_root = args.logdir + args.name + '/png'
            if not os.path.exists(png_root):
                os.makedirs(png_root)
            import matplotlib.pyplot as plt
            plt.plot(losses)
            plt.savefig(png_root + '/loss{}.png'.format(self.taskid))
            plt.close()

    def simplicial_complex_postprocessing_podnet(self):
        printlog('{} with {}'.format(args.base, args.opc['curve']))
        # ************************ prepare data ************************ #
        trsf = self.traindata.trsf
        if args.dataset == 'CIFAR100':
            memdata = Memorydata(memory, trsf)
        elif 'ImageNet' in args.dataset:
            memdata = ImageNetData(memory.x, memory.y, memory.t, trsf)
        else:
            raise ValueError
        if args.mode == 'DDP':
            sampler = utils.data.DistributedSampler(memdata, shuffle=True)
            self.loader = DataLoader(memdata, args.bs, num_workers=args.workers, sampler=sampler)
        else:
            self.loader = DataLoader(memdata, args.bs, num_workers=args.workers, shuffle=True)

        # ************************ optimize curve ************************ #
        self.model.eval()
        self.paramkeys = self.param_keys()
        self.fast_state = self.detach_state()
        self.start = self.state_detach(memory.pre_model.state_dict())
        self.stop = self.state_detach(self.model.state_dict())
        self.start = self.podnet_add_parameters()

        # assert args.opc['lr'] > 0  # learning rate of curve
        # assert args.opc['order'] > 0
        # assert args.opc['boundary'] > 0
        self.model.eval()
        hook_handle = self.register_hook(module_name='classifier', variable_name='representations',
                                         call_back=self.hook_abstract_call_back, side='input')
        self.curve = eval(args.opc['curve'])(self.filter_params(self.start),
                                             self.filter_params(self.stop),
                                             lambd=args.opc['cut_off']).to(self.device)
        if args.mode == 'DDP' and not isinstance(self.curve, nn.parallel.DistributedDataParallel):
            self.curve = nn.parallel.DistributedDataParallel(self.curve, device_ids=[args.local_rank],
                                                             output_device=args.local_rank)
        optimizer = torch.optim.SGD(self.curve.bend_parameters, lr=args.opc['lr'], weight_decay=args.opc['decay'],
                                    momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.opc['steps'],
                                                         gamma=args.opc['gamma'])

        progress = tqdm(range(1, len(self.loader) * args.opc['epochs'] * args.opc['connectors'] + 1),
                        disable='slient' in args.opt)
        progress.set_description('OPC  ')
        losses = []
        for connector_id in range(args.opc['connectors']):
            for epoch in range(int(args.opc['epochs'])):
                for x, y, t in self.loader:
                    t = t + 1
                    x = x.to(self.device)
                    y = y.to(self.device)
                    mask = t == self.taskid

                    losssum = 0
                    optimizer.zero_grad()
                    factors = np.linspace(0.1, 0.95, args.opc['points'])
                    for factor in factors:
                        index = 0
                        tmp_parameters = self.curve.forward(factor)
                        # tmp_tangent = [p.data.view(-1) for p in self.curve.tangent(factor)]
                        # tmp_tangent = torch.cat(tmp_tangent, dim=0)
                        # noise = torch.randn(tmp_tangent.numel(), requires_grad=False).to(self.device)
                        # noise = noise - (noise * tmp_tangent).sum() / (torch.norm(tmp_tangent).pow(2) + self.eps) * tmp_tangent
                        # noise = F.normalize(noise, p=2, dim=0) * args.opc['radius']
                        begin, end = 0, 0
                        for key, value in self.fast_state.items():
                            if key in self.paramkeys:
                                begin = end
                                value_ = tmp_parameters[index]
                                end += value_.numel()
                                self.fast_state[key] = value_  # + noise[begin: end].reshape(value.shape)
                                index += 1
                            else:
                                self.fast_state[key] = self.start[key] * (1 - factor) + self.stop[key] * factor
                        self.model.zero_grad()

                        # current
                        with torch.no_grad():
                            if factor < args.opc['cut_off']:
                                # previous
                                self.model.load_state_dict(self.start)
                                pre_output = self.model(x)
                                pre_representations = self.get_hook_result(model=self.model,
                                                                           variable_name='representations').detach().clone()
                            else:
                                self.model.load_state_dict(self.stop)
                                cur_output = self.model(x)
                                cur_representations = self.get_hook_result(model=self.model,
                                                                           variable_name='representations').detach().clone()

                        self.model.load_state_dict(self.fast_state)
                        tmp_output = self.model(x)
                        tmp_representations = self.get_hook_result(model=self.model,
                                                                   variable_name='representations').clone()

                        loss = 0
                        if factor < args.opc['cut_off']:
                            # classification loss
                            if 'ce' in args.opc['loss']:
                                loss = loss + torch.nn.functional.cross_entropy(tmp_output['logits'][~mask], y[~mask])

                            if 'nca' in args.opc['loss']:
                                loss = loss + self.nca_loss(tmp_output['logits'][~mask], y[~mask],
                                                            scale=self.model.module.post_processor.factor
                                                            if args.mode in (
                                                            'DP', 'DDP') else self.model.post_processor.factor,
                                                            margin=args.nca['margin'],
                                                            exclude_pos_denominator=args.nca['exclude_pos_denominator'])

                            if 'balance' in args.opc and args.opc['balance']:
                                loss = loss * (abs(factor - 0.5) + args.opc['balance'])

                            # embedding loss
                            if 'embed' in args.opc['loss']:
                                loss = loss + F.cosine_embedding_loss(tmp_representations, pre_representations,
                                                                      torch.ones(x.shape[0]).to(self.device))

                            if 'pod' in args.opc['loss']:
                                if args.base == 'PODNet':
                                    pod_factor = args.pod_spatial["scheduled_factor"] * math.sqrt(
                                        sum(args.increments[:self.taskid])
                                        / args.increments[self.taskid - 1])
                                    loss = loss + self.pod_loss(pre_output["attention"], tmp_output["attention"],
                                                                **args.pod_spatial) * pod_factor
                        else:
                            # classification loss
                            if 'ce' in args.opc['loss']:
                                loss = loss + torch.nn.functional.cross_entropy(tmp_output['logits'], y)

                            if 'nca' in args.opc['loss']:
                                loss = loss + self.nca_loss(tmp_output['logits'], y,
                                                            scale=self.model.module.post_processor.factor
                                                            if args.mode in (
                                                                'DP', 'DDP') else self.model.post_processor.factor,
                                                            margin=args.nca['margin'],
                                                            exclude_pos_denominator=args.nca['exclude_pos_denominator'])

                            if 'balance' in args.opc and args.opc['balance']:
                                loss = loss * (abs(factor - 0.5) + args.opc['balance'])

                            # embedding loss
                            if 'embed' in args.opc['loss']:
                                loss = loss + F.cosine_embedding_loss(tmp_representations, cur_representations,
                                                                      torch.ones(x.shape[0]).to(self.device))

                            if 'pod' in args.opc['loss']:
                                if args.base == 'PODNet':
                                    pod_factor = args.pod_spatial["scheduled_factor"] * math.sqrt(
                                        sum(args.increments[:self.taskid])
                                        / args.increments[self.taskid - 1])
                                    loss = loss + self.pod_loss(cur_output["attention"], tmp_output["attention"],
                                                                **args.pod_spatial) * pod_factor

                        loss.backward()
                        losssum += loss.item()
                        for p, (n, p_) in zip(tmp_parameters, self.model.named_parameters()):
                            if p_.grad is not None:
                                p.backward(p_.grad / args.opc['points'], retain_graph=True)

                    # maximize the volume of complex
                    complex_volume = self.curve.complex_volume() * (-args.opc['lambd'])
                    if isinstance(complex_volume, torch.Tensor):
                        complex_volume.backward()

                    optimizer.step()
                    losssum = losssum / args.opc['points']
                    losses.append(losssum)
                    progress.set_postfix({'connect': connector_id,
                                          'epoch': epoch,
                                          'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                                          'loss': losssum,
                                          'volume': complex_volume.item() / (-args.opc['lambd']) \
                                              if isinstance(complex_volume, torch.Tensor) else 0})
                    progress.update(1)
                scheduler.step()
            self.curve.add_vertex()
        progress.close()
        self.destroy_hook(hooks=hook_handle)

        if 'VIS' in args.opc and args.opc['VIS']:
            png_root = args.logdir + args.name + '/png'
            if not os.path.exists(png_root):
                os.makedirs(png_root)
            import matplotlib.pyplot as plt
            plt.plot(losses)
            plt.savefig(png_root + '/loss{}.png'.format(self.taskid))
            plt.close()
    
    def ensembling(self):

        if (args.ens['type'] == 'sample') or ('sample' in args.ens['type']):
            factors = np.linspace(args.ens['center'] - args.ens['gap'] / 2,
                                  args.ens['center'] + args.ens['gap'] / 2, args.ens['num'])
            for key, value in self.fast_state.items():
                self.fast_state[key] = value.zero_()

            for factor in factors:
                index = 0
                tmp_parameters = self.curve.forward(factor)
                tmp_tangent = [p.data.view(-1) for p in self.curve.tangent(factor)]
                tmp_tangent = torch.cat(tmp_tangent, dim=0)
                noise = torch.randn(tmp_tangent.numel(), requires_grad=False).to(self.device)
                noise = noise - (noise * tmp_tangent).sum() / (torch.norm(tmp_tangent).pow(2) + 1e-5) * tmp_tangent
                noise = F.normalize(noise, p=2, dim=0) * np.random.uniform(0, args.ens['radius'])
                begin, end = 0, 0
                for key, value in self.fast_state.items():
                    if key in self.paramkeys:
                        begin = end
                        value_ = tmp_parameters[index]
                        end += value_.numel()
                        if key in self.fast_state:
                            self.fast_state[key] += value_ + noise[begin: end].reshape(value.shape)
                        else:
                            self.fast_state[key] = value_ + noise[begin: end].reshape(value.shape)
                        index += 1
                    else:
                        if 'num_batches_tracked' in key:
                            self.fast_state[key] = self.stop[key]
                        else:
                            if key in self.fast_state:
                                self.fast_state[key] += self.start[key] * (1 - factor) + self.stop[key] * factor
                            else:
                                self.fast_state[key] = self.start[key] * (1 - factor) + self.stop[key] * factor

            for key, value in self.fast_state.items():
                if 'num_batches_tracked' not in key:
                    if 'running_var' in key:
                        self.fast_state[key] = F.relu(value / args.ens['num'])
                    else:
                        self.fast_state[key] = value / args.ens['num']

        self.model.load_state_dict(self.fast_state)

    def filter_params(self, state, keys=None):
        if keys is None:
            keys = self.param_keys()
        params = []
        for n, p in state.items():
            if n in keys:
                params.append(p)
        return params

    def podnet_add_parameters(self):
        start_ = OrderedDict()
        new_keys = list(set(self.stop.keys()) - set(self.start.keys()))
        for key in self.stop.keys():
            if key in new_keys:
                if 'bias' in key:
                    start_[key] = torch.zeros(self.stop[key].shape).to(self.device)
                else:
                    if 'init_new' in args.opc:
                        if args.opc['init_new'] == 'kaiming':
                            start_[key] = torch.randn(self.stop[key].shape).to(self.device)
                            nn.init.kaiming_normal_(start_[key], nonlinearity="linear", mode="fan_out")
                        elif args.opc['init_new'] == 'imprint':
                            start_[key] = self.imprinted_weights
                        else:
                            start_[key] = torch.randn(self.stop[key].shape).to(self.device)
                            nn.init.kaiming_normal_(start_[key], nonlinearity="linear", mode="fan_out")
                    else:
                        start_[key] = torch.randn(self.stop[key].shape).to(self.device)
                        nn.init.kaiming_normal_(start_[key], nonlinearity="linear", mode="fan_out")
            else:
                start_[key] = self.start[key]
        return start_
