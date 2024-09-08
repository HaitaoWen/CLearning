import abc
import torch.nn
from tool import *
import torch.nn as nn
from tqdm import tqdm
from torch import utils
from torch import distributed
from collections import OrderedDict
from torch.utils.data import DataLoader


class Base(abc.ABC):
    def __init__(self, model, traindata, taskid):
        super(Base, self).__init__()
        self.loss_ = 0
        self.device = None
        self.model = model
        self.hooks = dict()
        self.iterations = 0
        self.progress = None
        self.minclass = None
        self.maxclass = None
        self.taskid = taskid
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.trainloader = None
        self.traindata = traindata

        self.init_device()
        self.allocate_model()
        self.init_memory()
        self.allocate_data()
        self.init_optimizer()
        self.init_minmax_class()
        self.define_loss()
        time_memory_snapshot('Task{} [begin]:'.format(taskid), device=self.device)

    @property
    def loss(self):
        return self.loss_

    @loss.setter
    def loss(self, tmp):
        loss = self.loss_ * self.iterations + tmp
        self.iterations += 1
        self.loss_ = loss / self.iterations

    def init_device(self):
        if args.mode == 'GPU':
            self.device = torch.device('cuda', args.gpuid)
        elif args.mode == 'DP':
            self.device = torch.device('cuda')
        elif args.mode == 'DDP':
            torch.cuda.set_device(args.local_rank)
            self.device = torch.device('cuda', args.local_rank)

    def allocate_model(self):
        self.model = self.model.to(self.device)
        if args.mode == 'DP' and not isinstance(self.model, nn.DataParallel):
            # just need only once parallelization
            self.model = nn.DataParallel(self.model)
        elif args.mode == 'DDP' and not isinstance(self.model, nn.parallel.DistributedDataParallel):
            if not distributed.is_initialized():
                distributed.init_process_group(backend='nccl', init_method='env://')
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank)
        self.model.train()

    def init_memory(self):
        for attr, attr_v in memory.__dict__.items():
            if isinstance(attr_v, nn.Module):
                model = attr_v
                state = model.training
                model = model.to(self.device)
                if args.mode == 'DP' and not isinstance(model, nn.DataParallel):
                    # just need only once parallelization
                    model = nn.DataParallel(model)
                elif args.mode == 'DDP' and not isinstance(model, nn.parallel.DistributedDataParallel):
                    if not distributed.is_initialized():
                        distributed.init_process_group(backend='nccl', init_method='env://')
                    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                                output_device=args.local_rank)
                model.training = state
                memory.__dict__[attr] = model

    def deparallelization(self):
        if args.mode in ('DP', 'DDP'):
            self.model = self.model.module

    def allocate_data(self):
        self.trainloader = DataLoader(self.traindata, args.bs, shuffle=True, num_workers=args.workers,
                                      pin_memory=args.pin_memory)
        if args.mode == 'DDP':
            sampler = utils.data.DistributedSampler(self.traindata, shuffle=True)
            self.trainloader = DataLoader(self.traindata, args.bs, num_workers=args.workers, sampler=sampler,
                                          pin_memory=args.pin_memory)

    def init_optimizer(self):
        if args.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.decay,
                                             momentum=args.momentum)
        else:
            self.optimizer = None
        if args.steps is not None and self.optimizer is not None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.steps,
                                                                  gamma=args.gamma)
        else:
            self.scheduler = None

    # single head loss
    def single_head_loss(self, predict, target, taskids):
        """
        1) activate corresponding head
        """
        y_, y = activate_head(self.minclass, self.maxclass, predict, target)
        loss = torch.nn.functional.cross_entropy(y_, y)
        return loss

    @staticmethod
    def multi_head_loss(predict, target, taskids):
        """
        1) unique task indices
        2) iterate each head and activate it
        3) compute specific head loss
        4) average multi heads' loss
        """
        loss = 0
        taskids_ = torch.unique(taskids)
        for t in taskids_:
            mask = (t == taskids)
            minclass, maxclass = get_minmax_class(t)
            y_, y = activate_head(minclass, maxclass, predict[mask], target[mask])
            loss = loss + torch.nn.functional.cross_entropy(y_, y, reduction='sum')
        loss = loss / predict.shape[0]
        return loss

    def define_loss(self):
        if args.scenario == 'task':
            if args.memory > 0:
                loss_function = self.multi_head_loss
            else:
                loss_function = self.single_head_loss
        elif args.scenario == 'domain':
            loss_function = self.single_head_loss
        elif args.scenario == 'class':
            loss_function = self.single_head_loss
        else:
            raise ValueError("Unsupported {} scenario".format(args.scenario))
        self.criterion = loss_function

    def init_minmax_class(self):
        self.minclass, self.maxclass = get_minmax_class(self.taskid)

    @staticmethod
    def freeze_model(model):
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model

    def detach_state(self):
        state = OrderedDict()
        for key, value in self.model.state_dict().items():
            state[key] = value.clone().detach()
        return state

    def clone_parameters(self):
        parameters = OrderedDict()
        for key, value in self.model.named_parameters():
            parameters[key] = value.clone()
        return parameters

    def detach_parameters(self, model=None):
        if model is None:
            model = self.model
        parameters = OrderedDict()
        for key, value in model.named_parameters():
            parameters[key] = value.clone().detach()
        return parameters

    @staticmethod
    def state_detach(state):
        state_ = OrderedDict()
        for key, value in state.items():
            state_[key] = value.clone().detach()
        return state_

    def load_param(self, named_param: OrderedDict, model=None, detach=True, requires_grad=False):
        if model is None:
            model = self.model
        # printlog('\033[1;30;41mWarning! Using \'load_param\' with \'requires_grad=True\', '
        #          'this will unbind optimizer with model\033[0m')
        for key, value in named_param.items():
            # TODO, requires_grad is unrealized!
            # ********* Warning! this will unbind optimizer with model ********* #
            # this operation will change the memory address of parameters in model!
            # if you need to optimize the model,
            # it must need to recreate the optimizer for the model!
            module_path, _, param_name = key.rpartition(".")
            mod: torch.nn.Module = model.get_submodule(module_path)
            assert hasattr(mod, param_name)
            # assert isinstance(named_param[key], nn.Parameter)
            if not detach:
                value = nn.Parameter(value)
            try:
                if detach:
                    exec('mod.{}.data = value.data.clone().detach()'.format(param_name))
                else:
                    exec('mod.{}.data = value.data'.format(param_name))
            except SyntaxError:
                if isinstance(mod, torch.nn.ParameterList):
                    if detach:
                        exec('mod[{}].data = value.data.clone().detach()'.format(eval(param_name)))
                    else:
                        exec('mod[{}].data = value.data'.format(eval(param_name)))
                else:
                    raise NotImplementedError('This ERROR is caused by special customized structure:{}, '
                                              'not implemented in \'load_param\' function!'.format(mod))

    def param_keys(self):
        keys = []
        for key, _ in self.model.named_parameters():
            keys.append(key)
        return keys

    @staticmethod
    def hook_abstract_call_back(model, variable_name, side):
        def hook(module, input, output):
            if side == 'input':
                if args.mode == 'DP':
                    exec('model.{}[threading.get_native_id()] = input[0]'.format(variable_name),
                         {'threading': threading, 'model': model, 'input': input})
                else:
                    exec('model.{} = input[0]'.format(variable_name), {'model': model, 'input': input})
            else:
                if args.mode == 'DP':
                    exec('model.{}[threading.get_native_id()] = output'.format(variable_name),
                         {'threading': threading, 'model': model, 'output': output})
                else:
                    exec('model.{} = output'.format(variable_name), {'model': model, 'output': output})
        return hook
    
    def register_hook(self, model=None, module_name='fc', variable_name='representations',
                      call_back=None, direction='forward', side='input'):
        # ****** NOTE: variable_name must be unique compared with all registered hooks ***** #
        
        # each operation of registering hook will be recorded in self.hooks, 
        # no matter for current model or previous model
        
        if call_back is None:
            raise ValueError('call_back is None.')
        assert side in ('input', 'output')
        assert direction in ('forward', 'backward')

        if model is None:
            model = self.model

        if isinstance(model, torch.nn.DataParallel):
            model_ = model.module
            setattr(model_, variable_name, defaultdict())
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_ = model.module
            setattr(model_, variable_name, None)
        else:
            model_ = model
            setattr(model_, variable_name, None)

        count = 0
        hook_handle = None
        for name, module in model_.named_modules():
            if name == module_name:
                if direction == 'forward':
                    hook_handle = module.register_forward_hook(call_back(model_, variable_name, side))
                else:
                    hook_handle = module.register_backward_hook(call_back(model_, variable_name, side))
                self.hooks[variable_name] = hook_handle
                count += 1
        if count == 0:
            raise ValueError('{} is not in model.'.format(module_name))
        if count > 1:
            raise ValueError('{} is not unique.'.format(module_name))
        return hook_handle

    def get_hook_result(self, model=None, variable_name='representations', target_device=0):
        """
        target_device works only in DP mode.
        """
        if model is None:
            model = self.model
        if args.mode == 'GPU':
            result = eval('model.{}'.format(variable_name))
        elif args.mode == 'DP':
            result = dp_gather_and_reinit(model.module, variable_name, target_device=target_device)
        elif args.mode == 'DDP':
            result = eval('model.module.{}'.format(variable_name))
        return result

    def create_hook(self):
        pass

    def destroy_hook(self, model=None, hooks=None, variable_name=None):
        if model is None:
            model = self.model
         
        if args.mode in ('DP', 'DDP'):
            model_ = model.module
        else:
            model_ = model
        
        hooks_buffer_names = list(self.hooks.keys())
        hooks_buffer_values = list(self.hooks.values())
        if hooks is not None:
            if isinstance(hooks, list):
                if len(hooks) > 0:
                    for hook in hooks:
                        if hook in hooks_buffer_values:
                            # self.hooks.remove(hook)
                            index = hooks_buffer_values.index(hook)
                            variable_name_ = hooks_buffer_names[index]
                            self.hooks.pop(variable_name_)
                            if hasattr(model_, variable_name_):
                                exec('del model_.{}'.format(variable_name_))
                        hook.remove()
                    hooks.clear()
            else:
                if hooks in hooks_buffer_values:
                    index = hooks_buffer_values.index(hooks)
                    variable_name_ = hooks_buffer_names[index]
                    self.hooks.pop(variable_name_)
                    if hasattr(model_, variable_name_):
                        exec('del model_.{}'.format(variable_name_))
                hooks.remove()
        else:
            if len(self.hooks) > 0:
                for name, hook in self.hooks.items():
                    hook.remove()
                    if hasattr(model_, name):
                        exec('del model_.{}'.format(name))
                self.hooks.clear()
        
        if variable_name is not None:
            if isinstance(variable_name, list):
                for name in variable_name:
                    if name in self.hooks:
                        self.hooks[name].remove()
                        self.hooks.pop(name)
                    if hasattr(model_, name):
                        exec('del model_.{}'.format(name))
            elif isinstance(variable_name, str):
                if variable_name in self.hooks:
                    self.hooks[variable_name].remove()
                    self.hooks.pop(variable_name)
                if hasattr(model_, variable_name):
                    exec('del model_.{}'.format(variable_name))

    def evaluate(self, scenario):
        time_memory_snapshot('Task{} [end]:'.format(self.taskid), device=self.device)
        self.model.eval()
        time.sleep(1)
        if args.scheme == 'MultiTask':
            taskid = args.tasks
        else:
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
            targets, predicts = [], []
            with torch.no_grad():
                for x, y, _ in evalloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    y_ = self.model(x)
                    y_, _ = activate_head(minclass, maxclass, y_, y)
                    y_ = y_.topk(k=1, dim=1)[1]
                    y_ = remap_label(y_, minclass)
                    targets.append(y)
                    predicts.append(y_)
            targets = torch.cat(targets, dim=0)
            predicts = torch.cat(predicts, dim=0)
            if args.mode == 'DDP':
                torch.distributed.barrier()
                targets = distributed_concat(targets)
                predicts = distributed_concat(predicts)
            targets = targets.unsqueeze(dim=1)
            top1_acc = targets.eq(predicts[:, :1]).sum().item() / targets.shape[0]
            memory.AccMatrix[taskid - 1, t - 1] = top1_acc
            memory.task_size[t] = targets.shape[0]
        progress.close()
        if args.snapshot:
            save_memory(memory, self.taskid)
            save_model(self.model, self.taskid)
        print_log_metrics(memory.AccMatrix, taskid)
        time.sleep(1)
