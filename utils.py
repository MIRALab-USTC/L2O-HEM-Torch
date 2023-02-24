import math 
import time

import numpy as np
import random 
import torch
import os
import gtimer as gt 
import datetime
import dateutil.tz

from os.path import join
import os.path as osp
from collections import OrderedDict
from numbers import Number

from logger import logger
from global_const import *

_project_dir = join(os.getcwd(), os.pardir)
_LOCAL_LOG_DIR = join(_project_dir, 'data')

################
# cut feature constructor utils
def _get_integral_support(cut, model):
    nonz_coeff_cut = cut.getNNonz()
    # cols = cut.getCols()
    # cols_cur_lp = [col for col in cols if col.getLPPos() != -1]
    # nonz_coeff_integer = sum([1 for col in cols_cur_lp if col.isIntegral()])
    nonz_coeff_integer = model.getRowNumIntCols(cut)
    
    return float(nonz_coeff_integer / (nonz_coeff_cut + 1e-3))

def _get_cut_coeff_stats(cut):
    coeffs = cut.getVals()
    
    return np.mean(coeffs), np.max(coeffs), np.min(coeffs), np.std(coeffs)

def _get_obj_coeff_stats(scip_cutsel_env):
    vars = scip_cutsel_env.getVars()
    obj_coeffs = [var.getObj() for var in vars]
    
    return np.mean(obj_coeffs), np.max(obj_coeffs), np.min(obj_coeffs), np.std(obj_coeffs)

def compute_normalized_violation_scores(cut):
    lhs = cut.getLhs()
    rhs = cut.getRhs()
    cons = cut.getConstant()
    coeffs = cut.getVals()
    cols = cut.getCols()
    col_solution_value = [col.getPrimsol() for col in cols]
    lp_cut_value = np.dot(coeffs, col_solution_value) + cons
    if lp_cut_value < lhs:
        violation = (lhs - lp_cut_value) / abs(lhs+1e-1+1e-2)
        # violation = (lhs - lp_cut_value)
    elif lp_cut_value > rhs:
        violation = (lp_cut_value - rhs) / abs(rhs+1e-1+1e-2)
        # violation = (lp_cut_value - rhs)
    else:
        violation = 0.

    violation = max(0, violation)
    return violation

def advanced_cut_feature_generator(scip_cutsel_env, cuts):
    # add normalized violation feature
    cut_features = np.zeros((len(cuts), AdvancedCutFeatureNum))
    mean_coeff_obj, max_coeff_obj, min_coeff_obj, std_coeff_obj = _get_obj_coeff_stats(scip_cutsel_env)
    for i, cut in enumerate(cuts):
        obj_parall = scip_cutsel_env.getRowObjParallelism(cut)
        eff = scip_cutsel_env.getCutEfficacy(cut)
        # directed_cut_off_distance = scip_cutsel_env.getCutLPSolCutoffDistance(cut, best_primal_sol)
        # nonz_coeff_cut = cut.getNLPNonz()
        nonz_coeff_cut = cut.getNNonz()
        num_vars = scip_cutsel_env.getNVars()
        support = float(nonz_coeff_cut / (num_vars + 1e-3))
        # st_int_support = time.time()
        integral_support = _get_integral_support(cut, scip_cutsel_env)
        # time test debug
        # st_nv_end_int_support = time.time()
        normalized_violation = compute_normalized_violation_scores(cut)
        # et_nv = time.time()
        
        mean_coeff_cut, max_coeff_cut, min_coeff_cut, std_coeff_cut = _get_cut_coeff_stats(cut) # 统计量包不包括0 系数呢？
        # et_cut_coeff = time.time()
        # mean_obj_coeff = time.time()
        # print(f"steps: {i}, time_int_support: {st_nv_end_int_support-st_int_support}")
        # print(f"steps: {i}, time_nv: {et_nv-st_nv_end_int_support}")
        # print(f"steps: {i}, time cut coeff: {et_cut_coeff - et_nv}")
        # print(f"steps: {i}, time obj coeff: {mean_obj_coeff - et_cut_coeff}")
        
        cut_feature = [
            obj_parall,
            eff,
            support,
            integral_support,
            normalized_violation,
            mean_coeff_cut,
            max_coeff_cut,
            min_coeff_cut,
            std_coeff_cut,
            mean_coeff_obj,
            max_coeff_obj,
            min_coeff_obj,
            std_coeff_obj
        ]
        cut_features[i,:] = np.array(cut_feature)    
    return cut_features

def cut_feature_generator(scip_cutsel_env, cuts):
    """
    Input: scip_model static information + cuts dynamic information
    Output: the sequence of cut features
    """
    cut_features = np.zeros((len(cuts), CutFeatureNum))
    # scip_cutsel_env.getLPSol()
    # best_primal_sol = scip_cutsel_env.getBestSol()
    mean_coeff_obj, max_coeff_obj, min_coeff_obj, std_coeff_obj = _get_obj_coeff_stats(scip_cutsel_env)
    for i, cut in enumerate(cuts):
        obj_parall = scip_cutsel_env.getRowObjParallelism(cut)
        eff = scip_cutsel_env.getCutEfficacy(cut)
        # directed_cut_off_distance = scip_cutsel_env.getCutLPSolCutoffDistance(cut, best_primal_sol)
        # nonz_coeff_cut = cut.getNLPNonz()
        nonz_coeff_cut = cut.getNNonz()
        num_vars = scip_cutsel_env.getNVars()
        support = float(nonz_coeff_cut / (num_vars + 1e-3))
        integral_support = _get_integral_support(cut, scip_cutsel_env)
        mean_coeff_cut, max_coeff_cut, min_coeff_cut, std_coeff_cut = _get_cut_coeff_stats(cut) # 统计量包不包括0 系数呢？
        
        cut_feature = [
            obj_parall,
            eff,
            support,
            integral_support,
            mean_coeff_cut,
            max_coeff_cut,
            min_coeff_cut,
            std_coeff_cut,
            mean_coeff_obj,
            max_coeff_obj,
            min_coeff_obj,
            std_coeff_obj
        ]
        cut_features[i,:] =np.array(cut_feature)    
    return cut_features
################

################
# set seed utils
def set_global_seed(seed=None):
    if seed is None:
        seed = int(time.time())%4096
    np.random.seed(seed)    
    random.seed(seed)    
    torch.manual_seed(seed) #cpu    
    torch.cuda.manual_seed_all(seed)  #并行gpu    
    # torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致    
    # torch.backends.cudnn.benchmark = True 
    return seed
################

################
# logger utils
def create_exp_name(exp_prefix, exp_id=0, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return "%s_%s_%04d--s-%d" % (exp_prefix, timestamp, exp_id, seed)

def create_log_dir(
        exp_prefix,
        exp_id=0,
        seed=0,
        base_log_dir=None,
):
    """
    Creates and returns a unique log directory.
    :param exp_prefix: All experiments with this prefix will have log directories be under this directory.
    :param exp_id: The number of the specific experiment run within this experiment.
    :param base_log_dir: The directory where all log should be saved.
    :return:
    """
    exp_name = create_exp_name(exp_prefix, exp_id, seed)

    if base_log_dir is None:
        base_log_dir = _LOCAL_LOG_DIR

    log_dir = join(base_log_dir, exp_prefix, exp_name)

    if osp.exists(log_dir):
        logger.log("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)

    return log_dir

def setup_logger(
        exp_prefix="default",
        variant=None,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        snapshot_gap=1,
        log_tabular_only=False,
        log_dir=None,
        script_name=None,
        **create_log_dir_kwargs
):
    """
    Set up logger to have some reasonable default settings.
    Will save log output to

        base_log_dir/exp_prefix/exp_name.

    exp_name will be auto-generated to be unique.
    If log_dir is specified, then that directory is used as the output dir.

    :param exp_prefix: The sub-directory for this specific experiment.
    :param variant: 实验参数字典
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :param log_dir:
    :param script_name: If set, save the script name to this.
    :return:
    """

    first_time = log_dir is None
    if first_time:
        log_dir = create_log_dir(exp_prefix, **create_log_dir_kwargs)

    if variant is not None:
        logger.log("Variant:")
        # logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
        variant_log_path = join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    tabular_log_path = join(log_dir, tabular_log_file)
    text_log_path = join(log_dir, text_log_file)
    logger.add_text_output(text_log_path)

    if first_time:
        logger.add_tabular_output(tabular_log_path)

    else:
        logger._add_output(tabular_log_path, logger._tabular_outputs, logger._tabular_fds, mode='a')
        for tabular_fd in logger._tabular_fds:
            logger._tabular_header_written.add(tabular_fd)

    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)

    if script_name is not None:
        with open(join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    return log_dir

def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats
################

# get average weight of multiple models
def get_average_models(models_state_dict):
    average_model_state_dict = OrderedDict()
    for key in models_state_dict[0].keys():
        if 'net' in key:
            # pointer_net and cut_percent_policy
            average_model_state_dict[key] = OrderedDict()
            for model_key in models_state_dict[0][key].keys():
                weight_sum = 0
                for i in range(len(models_state_dict)):
                    weight_sum += models_state_dict[i][key][model_key]
                average_model_state_dict[key][model_key] = weight_sum / len(models_state_dict)
        else:
            # w_sum = 0
            # for i in range(len(models_state_dict)):
            #     w_sum += models_state_dict[i][key] 
            # average_model_state_dict[key] = w_sum / len(models_state_dict)
            average_model_state_dict[key] = models_state_dict[-1][key]
    return average_model_state_dict

# if __name__ == '__main__':
#     from ipdb import set_trace
#     data_base_path = "/datasets/code_run_experiments/code_220422_norstate_rewardscale_valid/data/parallel_reinforce_with_baseline_fix_logprobs_clamp_logp_all_anonymous/parallel_reinforce_with_baseline_fix_logprobs_clamp_logp_all_anonymous_2022_04_22_19_44_44_0000--s-1324"
#     test_model =  ["itr_70.pkl", "itr_112.pkl", "itr_140.pkl"]
#     models_state_dict = [torch.load(os.path.join(data_base_path, model_file)) for model_file in test_model]

#     state_dict = get_average_models(models_state_dict)
#     set_trace()