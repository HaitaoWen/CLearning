from .tool import *
from scheme.base import *
import torch.optim as optim
import torch.nn.functional as F
from .dataset.dataset import UnifiedDataset


class GPM(Base):
    """
    Saha G, Garg I, Roy K.
    Gradient projection memory for continual learning[J].
    arXiv preprint arXiv:2103.09762, 2021.
    """
    def __init__(self, model, data, taskid):
        self.traindata = UnifiedDataset(data['train']['x'], data['train']['y'], [data['t']] * len(data['train']['x']),
                                        dataset=args.dataset)
        self.validdata = UnifiedDataset(data['valid']['x'], data['valid']['y'], [data['t']] * len(data['valid']['x']),
                                        dataset=args.dataset)
        self.validloader = DataLoader(self.validdata, args.bs, shuffle=False, num_workers=args.workers,
                                      pin_memory=args.pin_memory)
        super(GPM, self).__init__(model, self.traindata, taskid)
        if 'step' in args.GPM:
            self.threshold = np.array(args.GPM['th']) + (taskid - 1) * np.array(args.GPM['step'])
        else:
            self.threshold = np.array(args.GPM['th'])
        self.optimizer = optim.SGD(model.parameters(), lr=args.lr)
        self.criterion = nn.CrossEntropyLoss()
        # validate
        self.projector = None
        self.best_loss = np.inf
        self.patience = args.lr_patience
        self.best_model = get_model(self.model)

    def train(self):
        self.compute_projector()
        self.gpm_train()
        self.update_GradientMemory()
        return self.model

    def gpm_train(self):
        self.progress = tqdm(range(1, len(self.trainloader) * args.epochs + 1),
                             disable='slient' in args.opt)
        self.progress.set_description('train')
        for epoch in range(1, args.epochs + 1):
            for x, y, t in self.trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                if args.scenario == 'domain':
                    y_ = self.model(x)
                elif args.scenario == 'task':
                    y_ = self.model(x)[self.taskid - 1]
                else:
                    raise NotImplementedError

                loss = self.criterion(y_, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.project_gradient()
                self.optimizer.step()
                self.loss = loss.item()
                self.progress.set_postfix({'task': self.taskid, 'epoch': epoch,
                                           'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                                           'avgloss': round(self.loss, 3), 'loss': loss.item()})
                self.progress.update(1)
            if self.validate(epoch) < 0:
                break
        self.progress.close()

    def validate(self, epoch):
        if args.dataset == 'PMNIST':
            # don't validate
            return 1
        self.model.eval()
        correct = 0
        total_num = 0
        total_loss = 0
        with torch.no_grad():
            for x, y, t in self.validloader:
                x = x.to(self.device)
                y = y.to(self.device)

                if args.scenario == 'domain':
                    y_ = self.model(x)
                elif args.scenario == 'task':
                    y_ = self.model(x)[self.taskid - 1]
                else:
                    raise NotImplementedError
                loss = self.criterion(y_, y)
                pred = y_.argmax(dim=1, keepdim=True)

                correct += pred.eq(y.view_as(pred)).sum().item()
                total_loss += loss.data.cpu().numpy().item() * len(x)
                total_num += len(x)

        valid_acc = correct / total_num
        valid_loss = total_loss / total_num
        if args.debug:
            printlog('Epoch {} | Valid: loss={:.3f}, acc={:5.1f}% |'.format(epoch, valid_loss, valid_acc * 100),
                     printed=False)
        # Adapt lr
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_model = get_model(self.model)
            self.patience = args.lr_patience
        else:
            self.patience -= 1
            if self.patience <= 0:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                lr *= args.gamma
                if args.debug:
                    printlog(' lr={:.1e}'.format(lr), printed=False)
                if lr < args.lr_min:
                    set_model_(self.model, self.best_model)
                    self.model.train()
                    return -1
                self.patience = args.lr_patience
                adjust_learning_rate(self.optimizer, epoch, args)
        if epoch == args.epochs:
            set_model_(self.model, self.best_model)
        self.model.train()
        return 1

    def compute_projector(self):
        if self.taskid == 1:
            return
        projector = []
        bias = memory.bias
        # Projection Matrix Precomputation
        for i in range(len(bias)):
            Uf = torch.Tensor(np.dot(bias[i], bias[i].transpose())).to(self.device)
            printlog('Layer {} - Projection Matrix shape: {}'.format(i + 1, Uf.shape))
            projector.append(Uf)
        printlog('-' * 40)
        self.projector = projector

    def project_gradient(self):
        if self.taskid == 1:
            return
        # Gradient Projections
        if args.dataset == 'PMNIST':
            for k, (m, params) in enumerate(self.model.named_parameters()):
                sz = params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1),
                                                               self.projector[k]).view(params.size())
        elif args.dataset == 'CIFAR100':
            kk = 0
            for k, (m, params) in enumerate(self.model.named_parameters()):
                if k < 15 and len(params.size()) != 1:
                    sz = params.grad.data.size(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1),
                                                                   self.projector[kk]).view(params.size())
                    kk += 1
                elif (k < 15 and len(params.size()) == 1) and self.taskid > 1:
                    params.grad.data.fill_(0)

        elif args.dataset == 'SUPER':
            kk = 0
            for k, (m, params) in enumerate(self.model.named_parameters()):
                if k < 4 and len(params.size()) != 1:
                    sz = params.grad.data.size(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1),
                                                                   self.projector[kk]).view(params.size())
                    kk += 1
                elif (k < 4 and len(params.size()) == 1) and self.taskid > 1:
                    params.grad.data.fill_(0)

        elif args.dataset in ('FIVE', 'miniImageNet'):
            kk = 0
            for k, (m, params) in enumerate(self.model.named_parameters()):
                if len(params.size()) == 4:
                    sz = params.grad.data.size(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1),
                                                                   self.projector[kk]).view(params.size())
                    kk += 1
                elif len(params.size()) == 1 and self.taskid > 1:
                    params.grad.data.fill_(0)
        else:
            raise NotImplementedError

    def get_representation_matrix(self):
        # Collect activations by forward pass
        self.model.eval()
        indices = np.arange(self.traindata._x.size(0))
        np.random.shuffle(indices)
        if args.dataset == 'PMNIST':
            indices = indices[0: 300]  # Take random training samples
            example_data = self.traindata._x[indices].view(-1, 28 * 28)
            example_data = example_data.to(self.device)
            self.model(example_data)

            batch_list = [300, 300, 300]
            mat_list = []  # list contains representation matrix of each layer
            act_key = list(self.model.act.keys())

            for i in range(len(act_key)):
                bsz = batch_list[i]
                act = self.model.act[act_key[i]].detach().cpu().numpy()
                activation = act[0:bsz].transpose()
                mat_list.append(activation)

        elif args.dataset == 'CIFAR100':
            indices = indices[0: 125]  # Take 125 random samples
            example_data = self.traindata._x[indices]
            example_data = example_data.to(self.device)
            self.model(example_data)

            batch_list = [2 * 12, 100, 100, 125, 125]
            mat_list = []
            act_key = list(self.model.act.keys())
            for i in range(len(self.model.map)):
                bsz = batch_list[i]
                k = 0
                if i < 3:
                    ksz = self.model.ksize[i]
                    s = compute_conv_output_size(self.model.map[i], self.model.ksize[i])
                    mat = np.zeros((self.model.ksize[i] * self.model.ksize[i] * self.model.in_channel[i], s * s * bsz))
                    act = self.model.act[act_key[i]].detach().cpu().numpy()
                    for kk in range(bsz):
                        for ii in range(s):
                            for jj in range(s):
                                mat[:, k] = act[kk, :, ii:ksz + ii, jj:ksz + jj].reshape(-1)
                                k += 1
                    mat_list.append(mat)
                else:
                    act = self.model.act[act_key[i]].detach().cpu().numpy()
                    activation = act[0: bsz].transpose()
                    mat_list.append(activation)

        elif args.dataset == 'SUPER':
            indices = indices[0: 125]  # Take 125 random samples
            example_data = self.traindata._x[indices]
            example_data = example_data.to(self.device)
            self.model(example_data)

            batch_list = [2 * 12, 100, 125, 125]
            pad = 2
            p1d = (2, 2, 2, 2)
            mat_list = []
            act_key = list(self.model.act.keys())
            for i in range(len(self.model.map)):
                bsz = batch_list[i]
                k = 0
                if i < 2:
                    ksz = self.model.ksize[i]
                    s = compute_conv_output_size(self.model.map[i], self.model.ksize[i], 1, pad)
                    mat = np.zeros((self.model.ksize[i] * self.model.ksize[i] * self.model.in_channel[i], s * s * bsz))
                    act = F.pad(self.model.act[act_key[i]], p1d, "constant", 0).detach().cpu().numpy()

                    for kk in range(bsz):
                        for ii in range(s):
                            for jj in range(s):
                                mat[:, k] = act[kk, :, ii:ksz + ii, jj:ksz + jj].reshape(-1)
                                k += 1
                    mat_list.append(mat)
                else:
                    act = self.model.act[act_key[i]].detach().cpu().numpy()
                    activation = act[0: bsz].transpose()
                    mat_list.append(activation)

        elif args.dataset in ('FIVE', 'miniImageNet'):
            indices = indices[0: 100]  # Take random training samples
            example_data = self.traindata._x[indices]
            example_data = example_data.to(self.device)
            self.model(example_data)

            act_list = []
            act_list.extend([self.model.act['conv_in'],
                             self.model.layer1[0].act['conv_0'],
                             self.model.layer1[0].act['conv_1'],
                             self.model.layer1[1].act['conv_0'],
                             self.model.layer1[1].act['conv_1'],
                             self.model.layer2[0].act['conv_0'],
                             self.model.layer2[0].act['conv_1'],
                             self.model.layer2[1].act['conv_0'],
                             self.model.layer2[1].act['conv_1'],
                             self.model.layer3[0].act['conv_0'],
                             self.model.layer3[0].act['conv_1'],
                             self.model.layer3[1].act['conv_0'],
                             self.model.layer3[1].act['conv_1'],
                             self.model.layer4[0].act['conv_0'],
                             self.model.layer4[0].act['conv_1'],
                             self.model.layer4[1].act['conv_0'],
                             self.model.layer4[1].act['conv_1']])

            batch_list = [10, 10, 10, 10, 10, 10, 10, 10, 50, 50, 50, 100, 100, 100, 100, 100, 100]  # scaled
            # network arch
            if args.dataset == 'FIVE':
                stride_list = [1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
                map_list = [32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8, 4, 4, 4]
            elif args.dataset == 'miniImageNet':
                stride_list = [2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
                map_list = [84, 42, 42, 42, 42, 42, 21, 21, 21, 21, 11, 11, 11, 11, 6, 6, 6]
            else:
                raise NotImplementedError

            in_channel = [3, 20, 20, 20, 20, 20, 40, 40, 40, 40, 80, 80, 80, 80, 160, 160, 160]

            pad = 1
            sc_list = [5, 9, 13]
            p1d = (1, 1, 1, 1)
            mat_final = []  # list containing GPM Matrices
            mat_list = []
            mat_sc_list = []
            for i in range(len(stride_list)):
                if i == 0:
                    ksz = 3
                else:
                    ksz = 3
                bsz = batch_list[i]
                st = stride_list[i]
                k = 0
                s = compute_conv_output_size(map_list[i], ksz, stride_list[i], pad)
                mat = np.zeros((ksz * ksz * in_channel[i], s * s * bsz))
                act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()
                for kk in range(bsz):
                    for ii in range(s):
                        for jj in range(s):
                            mat[:, k] = act[kk, :, st * ii:ksz + st * ii, st * jj:ksz + st * jj].reshape(-1)
                            k += 1
                mat_list.append(mat)
                # For Shortcut Connection
                if i in sc_list:
                    k = 0
                    s = compute_conv_output_size(map_list[i], 1, stride_list[i])
                    mat = np.zeros((1 * 1 * in_channel[i], s * s * bsz))
                    act = act_list[i].detach().cpu().numpy()
                    for kk in range(bsz):
                        for ii in range(s):
                            for jj in range(s):
                                mat[:, k] = act[kk, :, st * ii:1 + st * ii, st * jj:1 + st * jj].reshape(-1)
                                k += 1
                    mat_sc_list.append(mat)

            ik = 0
            for i in range(len(mat_list)):
                mat_final.append(mat_list[i])
                if i in [6, 10, 14]:
                    mat_final.append(mat_sc_list[ik])
                    ik += 1
            mat_list = mat_final
        else:
            raise NotImplementedError

        printlog('-' * 30)
        printlog('Representation Matrix')
        printlog('-' * 30)
        for i in range(len(mat_list)):
            printlog('Layer {} : {}'.format(i + 1, mat_list[i].shape))
        printlog('-' * 30)
        return mat_list

    def update_GradientMemory(self):
        mat_list = self.get_representation_matrix()
        if not hasattr(memory, 'bias'):
            setattr(memory, 'bias', [])
        bias = memory.bias
        printlog('Threshold: ', self.threshold)
        if not bias:
            # After First Task
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U, S, Vh = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                r = np.sum(np.cumsum(sval_ratio) < self.threshold[i])  # +1
                bias.append(U[:, 0:r])
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1 ** 2).sum()
                act_hat = activation - np.dot(np.dot(bias[i], bias[i].transpose()), activation)
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                # criteria (Eq-9)
                sval_hat = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                accumulated_sval = (sval_total - sval_hat) / sval_total

                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < self.threshold[i]:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    printlog('Skip Updating GPM for layer: {}'.format(i + 1))
                    continue
                Ui = np.hstack((bias[i], U[:, 0:r]))
                if Ui.shape[1] > Ui.shape[0]:
                    bias[i] = Ui[:, 0:Ui.shape[0]]
                else:
                    bias[i] = Ui

        printlog('-' * 40)
        printlog('Gradient Constraints Summary')
        printlog('-' * 40)
        for i in range(len(bias)):
            printlog('Layer {} : {}x{}'.format(i + 1, bias[i].shape[0], bias[i].shape[1]))
        printlog('-' * 40)

    def evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            for t in range(self.taskid):
                total_num, correct = 0, 0
                testdata = UnifiedDataset(data[t]['test']['x'], data[t]['test']['y'],
                                          [data[t]['t']] * len(data[t]['test']['x']), dataset=args.dataset)
                testloader = DataLoader(testdata, args.bs, shuffle=False, num_workers=args.workers,
                                        pin_memory=args.pin_memory)
                for x, y, _ in testloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    if args.scenario == 'domain':
                        y_ = self.model(x)
                    elif args.scenario == 'task':
                        y_ = self.model(x)[t]
                    else:
                        raise NotImplementedError
                    pred = y_.argmax(dim=1, keepdim=True)
                    correct += pred.eq(y.view_as(pred)).sum().item()
                    total_num += len(x)
                memory.AccMatrix[self.taskid-1, t] = correct / total_num
        print_log_metrics(memory.AccMatrix, self.taskid)
