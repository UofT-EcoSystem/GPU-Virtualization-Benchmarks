
def predictor_option(alloc, stage_1, stage_2, at_least_once):
    combo_name = "{}-{}-{}-{}".format(alloc.name,
                                      stage_1.name,
                                      stage_2.name,
                                      at_least_once)
    return combo_name


def get_ws(alloc, stage_1, stage_2, at_least_once):
    return predictor_option(alloc, stage_1, stage_2, at_least_once) + '-ws'


def get_perf(alloc, stage_1, stage_2, at_least_once):
    return predictor_option(alloc, stage_1, stage_2, at_least_once) + '-perf'


def get_option(alloc, stage_1, stage_2, at_least_once):
    return predictor_option(alloc, stage_1, stage_2, at_least_once) + '-option'
