from tool import *
from .finetune import FineTune

# ************** regularization ************** #
from .regularization.lwf import LwF
from .regularization.ewc import EWC
from .regularization.gpm.gpm import GPM

# **************     replay     ************** #
from .replay.er import ER
from .replay.icarl import iCaRL
from .replay.lucir import LUCIR
from .replay.podnet.podnet import PODNet
from .replay.ssil import SSIL
from .replay.afc.afc import AFC
from .replay.cscct import CSCCT
from .replay.opc.opc import OPC
from .replay.ancl import ANCL
from .replay.mtd.mtd import MTD

# *************   structure     ************** #
from .structure.aanet.aanet import AANet


def init_scheme(model, scenario):
    start = 1
    Scheme = eval(args.scheme)
    if args.resume:
        memory.load_memory()
        start = eval(args.resume[args.resume.rfind('task') + 4: args.resume.rfind('.')]) + 1
        for taskid, traindata in enumerate(scenario, start=1):
            if taskid == start:
                break
            scheme = Scheme(model, traindata, taskid)
            model = scheme.model
        model = load_model(model)
    return Scheme, model, start
