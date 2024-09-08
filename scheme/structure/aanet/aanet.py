from scheme.base import *
from copy import deepcopy
from .basenet import BasicNet
from .resnet_mtl import resnet18
from .my_resnet_mtl import resnet_rebuffi
from scheme.replay.podnet.podnet import PODNet


if hasattr(args, 'base') and args.scheme == 'AANet':
    if args.base == 'PODNet':
        BASE = PODNet
    else:
        printlog('\033[1;30;41mPredetermined base {} does not exist, '
                 'use default base: PODNet\033[0m'.format(args.scheme))
        BASE = PODNet
else:
    BASE = PODNet


class AANet(BASE):
    """
    Liu Y, Schiele B, Sun Q.
    Adaptive aggregation networks for class-incremental learning[C]
    //Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
    2021: 2544-2553.
    """
    # TODO, only support PODNet baseline currently.
    def __init__(self, model, traindata, taskid):
        Base.__init__(self, model, traindata, taskid)
        self.wrap_model()
        super(AANet, self).__init__(self.model, traindata, taskid)

    def train(self):
        self.podnet_train()
        self.construct_exemplar_set()
        self.cbf_finetune()
        self.construct_nme_classifier()
        pre_model = deepcopy(self.model)
        setattr(memory, 'pre_model', self.freeze_model(pre_model))
        return self.model

    def wrap_model(self):
        """
        model1 is plastic, model2 is stable. 
        """
        if self.taskid == 2:
            printlog("Using weight transferring operation")
            model1 = self.model.module.copy().to(self.device) if args.mode in ('DP', 'DDP') else \
                self.model.copy().to(self.device)
            model2 = self.model.module.copy().to(self.device) if args.mode in ('DP', 'DDP') else \
                self.model.copy().to(self.device)
            convnet_kwargs = args.__dict__.get("convnet_config", {})
            if args.dataset == 'CIFAR100':
                model1.convnet = resnet_rebuffi(**convnet_kwargs)
            elif args.dataset in ('ImageNet100', 'ImageNet1000'):
                model1.convnet = resnet18(**convnet_kwargs)
            else:
                raise ValueError('Unsupported Dataset!')
            temp_dict = model2.convnet.state_dict()
            network_dict = model1.convnet.state_dict()
            network_dict.update(temp_dict)
            model1.convnet.load_state_dict(network_dict)
            model1.convnet.to(self.device)
            self.model = BasicNet(
                model1.convnet,
                model2.convnet,
                classifier_kwargs=args.__dict__.get("classifier_config", {}),
                postprocessor_kwargs=args.__dict__.get("postprocessor_config", {}),
                device=self.device,
                return_features=True,
                extract_no_act=True,
                classifier_no_act=args.__dict__.get("classifier_no_act", True),
                attention_hook=True,
                gradcam_hook=bool(args.__dict__.get("gradcam_distil", {})),
                dataset=args.dataset
            )
            self.model.classifier = model1.classifier

    def init_optimizer_custom(self):
        if args.groupwise_factors and isinstance(args.groupwise_factors, dict):
            groupwise_factor = args.groupwise_factors

            params = []
            for group_name, group_params in self.model.get_group_parameters().items():
                if group_params is None or group_name == "last_block" or group_name == "last_block_1" \
                        or group_name == "last_block_2":
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
