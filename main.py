from tool import *
from scheme import *


def main():
    backup()
    print_args()
    init_state()
    model = build_model()
    scenario, scenario_eval = load_scenario()
    Scheme, model, start = init_scheme(model, scenario)
    for taskid, traindata in enumerate(scenario, start=1):
        if taskid < start:
            continue
        scheme = Scheme(model, traindata, taskid)
        model = scheme.train()
        scheme.evaluate(scenario_eval)


if __name__ == '__main__':
    main()
