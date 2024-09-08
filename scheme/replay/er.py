from scheme.base import *
from scheme.finetune import FineTune
from tool.sampler import MemoryBatchSampler_extra
from tool.sampler import MemoryBatchSampler_intra


class ER(FineTune):
    """
    Chaudhry A, Rohrbach M, Elhoseiny M, et al.
    On tiny episodic memories in continual learning[J].
    arXiv preprint arXiv:1902.10486, 2019.

    Args:
        memory: the number of samples sampled for a class
        opt: list type, default sampling mode is MoF(offline);
             if 'random_sample' is in this list, sampling mode is random(offline);
    """

    def __init__(self, model, traindata, taskid):
        super(ER, self).__init__(model, traindata, taskid)
        assert args.memory > 0, 'ER based scheme needs memory to store samples of old tasks!'
        if taskid == 1:
            return
        if args.mcat:
            cur_data_num = len(self.traindata)
            sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=[i for i in range(cur_data_num)])
            if 'catp' in args:
                if args.catp == 'extra':
                    batch_sampler = MemoryBatchSampler_extra(sampler, batch_size=args.bs, mem_size=len(memory.x),
                                                             mem_batch_size=args.mbs, drop_last=False)
                elif args.catp == 'intra':
                    batch_sampler = MemoryBatchSampler_intra(sampler, batch_size=args.bs, mem_size=len(memory.x),
                                                             mem_batch_size=args.mbs, drop_last=False)
                else:
                    raise ValueError('Unrecognized concatenate pattern {}'.format(args.catp))
            else:
                batch_sampler = MemoryBatchSampler_intra(sampler, batch_size=args.bs, mem_size=len(memory.x),
                                                         mem_batch_size=args.mbs, drop_last=False)
        if args.dataset == 'CIFAR100':
            dataset = CIFARData(self.traindata, memory)
        elif 'MNIST' in args.dataset:
            dataset = []
            trans = self.traindata.trsf
            if isinstance(trans, list):
                # deal with problem caused by the new version continuum
                trans = trans[self.taskid - 1]
            _x, _y, _t = self.traindata._x, self.traindata._y, self.traindata._t
            for x, y, t in zip(_x, _y, _t):
                x = Image.fromarray(x.astype("uint8"))
                x_t = trans(x).squeeze()
                dataset.append([x_t, y, int(t)])
            dataset.extend(memory.x)
            dataset = MNISTData(dataset)
        elif 'ImageNet' in args.dataset:
            x = traindata._x.tolist() + memory.x
            y = traindata._y.tolist() + memory.y
            t = traindata._t.tolist() + memory.t
            dataset = ImageNetData(x, y, t, traindata.trsf)
        else:
            raise ValueError('Unsupported dataset: {}'.format(args.dataset))
        if args.mode == 'DDP':
            sampler_ = utils.data.DistributedSampler(dataset, shuffle=True)
        else:
            sampler_ = None
        if args.mcat:
            self.trainloader = DataLoader(dataset, batch_sampler=batch_sampler,
                                          num_workers=args.workers, sampler=sampler_, pin_memory=args.pin_memory)
        else:
            self.trainloader = DataLoader(dataset, batch_size=args.bs, shuffle=True if sampler_ is None else False,
                                          num_workers=args.workers, sampler=sampler_, pin_memory=args.pin_memory)

    def train(self):
        super(ER, self).train()
        self.construct_exemplar_set()
        return self.model

    def random_sample(self, _x, _y):
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
        return x, y, t

    def herding_sample(self, _x, _y, trsf):
        x, y, t = [], [], []
        classes = np.unique(_y)
        for c in classes:
            mask = c == _y
            images = _x[mask]
            imagesdata = Images(images, trsf)
            if args.mode == 'DDP':
                sampler = utils.data.DistributedSampler(imagesdata, shuffle=False)
                loader = DataLoader(imagesdata, args.bs, num_workers=args.workers, sampler=sampler)
            else:
                loader = DataLoader(imagesdata, args.bs, num_workers=args.workers, shuffle=False)
            representations = []
            for x_, index in loader:
                x_ = x_.to(self.device)
                with torch.no_grad():
                    self.model(x_)
                representations.append(self.get_hook_result(variable_name='representations'))
            representations = torch.cat(representations, dim=0)
            if args.mode == 'DDP':
                # TODO, warning, indices of samples may be disorder
                torch.distributed.barrier()
                representations = distributed_concat(representations)

            D = representations.T
            D = D / (torch.linalg.norm(D, dim=0) + 1e-8)
            mu = torch.mean(D, dim=1)
            herding_matrix = torch.zeros((representations.shape[0],))
            w_t = mu
            iter_herding, iter_herding_eff = 0, 0
            while not (
                    torch.sum(herding_matrix != 0) == min(args.memory, representations.shape[0])
            ) and iter_herding_eff < 1000:
                tmp_t = w_t[None, :] @ D
                ind_max = torch.argmax(tmp_t)
                iter_herding_eff += 1
                if herding_matrix[ind_max] == 0:
                    herding_matrix[ind_max] = 1 + iter_herding
                    iter_herding += 1
                w_t = w_t + mu - D[:, ind_max]
            herding_matrix[torch.where(herding_matrix == 0)[0]] = 10000

            indices = herding_matrix.argsort()[:args.memory]
            for i in indices:
                x.append(images[i]), y.append(c), t.append(self.taskid - 1)
        return x, y, t

    def construct_exemplar_set(self):
        # Herding is adopted by default
        self.model.eval()
        trsf = self.traindata.trsf
        if args.dataset in ['perMNIST', 'rotMNIST'] and isinstance(trsf, list):
            # deal with problem caused by the new version continuum
            trsf = trsf[self.taskid - 1]
        _x, _y = self.traindata._x, self.traindata._y
        if 'sample' in args and args.sample == 'random':
            x, y, t = self.random_sample(_x, _y)
        elif ('sample' in args and args.sample == 'herding') or ('sample' not in args):
            # register hook to get representations
            hook_handle = self.register_hook(module_name='fc', variable_name='representations',
                                             call_back=self.hook_abstract_call_back, side='input')
            x, y, t = self.herding_sample(_x, _y, trsf)
            self.destroy_hook(hooks=hook_handle)
        else:
            raise NotImplementedError

        if args.dataset == 'CIFAR100':
            x = np.stack(x, axis=0)
            y = np.stack(y, axis=0)
            t = np.stack(t, axis=0)
            memory.x = np.concatenate((memory.x, x), axis=0) if memory.x is not None else x
            memory.y = np.concatenate((memory.y, y), axis=0) if memory.y is not None else y
            memory.t = np.concatenate((memory.t, t), axis=0) if memory.t is not None else t
        elif 'MNIST' in args.dataset:
            x__ = []
            for x_, y_, t_ in zip(x, y, t):
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

    def drop_curtask_exemplar(self):
        if args.dataset == 'CIFAR100':
            mask = memory.t != self.taskid - 1
            memory.x = memory.x[mask]
            memory.y = memory.y[mask]
            memory.t = memory.t[mask]
        elif 'ImageNet' in args.dataset:
            x, y, t = np.array(memory.x), np.array(memory.y), np.array(memory.t)
            mask = t != self.taskid - 1
            memory.x = x[mask].tolist()
            memory.y = y[mask].tolist()
            memory.t = t[mask].tolist()
        else:
            raise ValueError('Unsupported dataset: {}'.format(args.dataset))
