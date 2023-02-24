import argparse
import os
from random import random
from re import L
from unittest import result
from sqlalchemy import all_
from tqdm import tqdm 
import torch
import numpy as np
import json 
import copy 
import os.path as osp
import math
import gtimer as gt
from collections import OrderedDict

import torch.optim as optim

import torch.multiprocessing as mp
import multiprocessing as python_mp

from environments import SCIPCutSelEnv
from cutsel_agent_parallel import CutSelectAgent, HierarchyCutSelectAgent
from logger import logger
from algorithms import ReinforceBaselineAlg, HRLReinforceAlg

from utils import setup_logger, create_stats_ordered_dict, set_global_seed, get_average_models
from utilss.mean_std import RunningMeanStd

# for debug
# from ipdb import set_trace

def generate_samples(return_queue,env,policy,value,epoch,samples_per_worker,sel_cuts_percent,device,train_decode_type,reward_type,seed,mean_std,policy_type,random_seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    device = 'cuda:0'
    policy = policy.to(device)
    log_prefix = os.getpid()
    _ = set_global_seed(seed%4096)
    logger.log(f"{log_prefix}: debug log random seed {seed%4096}")
    logger.log(f"{log_prefix}: sampling data ...")
    env_step_infos = {
        "solving_time": [],
        "ntotal_nodes": [],
        "primal_dual_gap": [],
        "primaldualintegral": []
    } # dict of list
    training_datasets = {
        "state": [],
        "action": [],
        "sel_cuts_num": [],
        "neg_reward": []
    } # list of numpy/list/int
    # cuts_infos = {
    #     "length_cuts": [],
    #     "length_forced_cuts": [],
    #     "cut_features": []
    # }
    env.set_seed(seed)
    for step in range(samples_per_worker):
        logger.log(f"{log_prefix}: training...  epoch: {epoch}...  steps: {step+1}")
        logger.log(f"{log_prefix}: cuda memory: {torch.cuda.memory_allocated(0)/1024**3} GB")
        logger.log(f"{log_prefix}: cuda cached: {torch.cuda.memory_cached(0)/1024**3} GB")
        env.reset()
        # reset action agent
        cutsel_agent = CutSelectAgent(
            env.m,
            policy,
            value,
            sel_cuts_percent,
            device,
            train_decode_type,
            mean_std,
            policy_type
        )
        env_step_info = env.step(cutsel_agent)
        state_action_dict = cutsel_agent.get_data()
        lp_info = cutsel_agent.get_lp_info()
        # cuts_info = cutsel_agent.get_cuts_info()
        if not state_action_dict:
            logger.log(f"{log_prefix}: warning!!! current instance cuts len <= 1")
            continue
        if reward_type == "lp_solution_value":
            if len(lp_info["lp_solution_value"]) < 2:
                continue
            else:
                neg_reward = lp_info["lp_solution_value"][0] - lp_info["lp_solution_value"][1]
        for key in env_step_info.keys():
            assert key in env_step_infos.keys()
            env_step_infos[key].append(env_step_info[key])
        # for key in cuts_info.keys():
        #     cuts_infos[key].append(cuts_info[key])
        for key in state_action_dict.keys():
            training_datasets[key].append(state_action_dict[key])
        if reward_type == 'lp_solution_value':
            training_datasets['neg_reward'].append(neg_reward)
        else:
            training_datasets['neg_reward'].append(env_step_info[reward_type])
        cutsel_agent.free_problem()

    # list dict numpy cuda tensor 都可以传，cpu tensor 传不了，带梯度信息的cuda tensor 传不了
    return_queue.put((env_step_infos, training_datasets)) 

def generate_hierarchy_samples(return_queue,env,policy,cutsel_policy,value,epoch,samples_per_worker,sel_cuts_percent,device,train_decode_type,reward_type,seed,mean_std,policy_type,random_seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    device = 'cuda:0'
    policy = policy.to(device)
    cutsel_policy = cutsel_policy.to(device)
    log_prefix = os.getpid()
    _ = set_global_seed(seed%4096)
    logger.log(f"{log_prefix}: debug log random seed {seed%4096}")
    logger.log(f"{log_prefix}: sampling data ...")
    env_step_infos = {
        "solving_time": [],
        "ntotal_nodes": [],
        "primal_dual_gap": [],
        "primaldualintegral": []
    } # dict of list
    training_datasets = {
        "state": [],
        "action": [],
        "sel_cuts_num": [],
        "neg_reward": []
    } # list of numpy/list/int
    training_high_level_datasets = {
        "state": [],
        "action": [],
        "neg_reward": []
    }
    env.set_seed(seed)
    for step in range(samples_per_worker):
        logger.log(f"{log_prefix}: training...  epoch: {epoch}...  steps: {step+1}")
        logger.log(f"{log_prefix}: cuda memory: {torch.cuda.memory_allocated(0)/1024**3} GB")
        logger.log(f"{log_prefix}: cuda cached: {torch.cuda.memory_cached(0)/1024**3} GB")
        env.reset()
        # reset action agent
        cutsel_agent = HierarchyCutSelectAgent(
            env.m,
            policy,
            cutsel_policy,
            value,
            sel_cuts_percent,
            device,
            train_decode_type,
            mean_std,
            policy_type
        )
        env_step_info = env.step(cutsel_agent)
        state_action_dict = cutsel_agent.get_data()
        lp_info = cutsel_agent.get_lp_info()
        high_level_state_action_dict = cutsel_agent.get_high_level_data()
        if (not state_action_dict) or (not high_level_state_action_dict):
            logger.log(f"{log_prefix}: warning!!! current instance cuts len <= 1")
            continue
        if reward_type == "lp_solution_value":
            if len(lp_info["lp_solution_value"]) < 2:
                continue
            else:
                neg_reward = lp_info["lp_solution_value"][0] - lp_info["lp_solution_value"][1]
        for key in env_step_info.keys():
            assert key in env_step_infos.keys()
            env_step_infos[key].append(env_step_info[key])
        
        for key in state_action_dict.keys():
            training_datasets[key].append(state_action_dict[key])
        for key in high_level_state_action_dict.keys():
            training_high_level_datasets[key].append(high_level_state_action_dict[key])

        if reward_type == 'lp_solution_value':
            training_datasets['neg_reward'].append(neg_reward)
            training_high_level_datasets['neg_reward'].append(neg_reward)
        else:
            training_datasets['neg_reward'].append(env_step_info[reward_type])
            training_high_level_datasets['neg_reward'].append(env_step_info[reward_type])
        cutsel_agent.free_problem()

    # list dict numpy cuda tensor 都可以传，cpu tensor 传不了，带梯度信息的cuda tensor 传不了
    return_queue.put((env_step_infos, training_datasets, training_high_level_datasets)) 

def evaluate(
    return_queue,
    env,
    policy,
    value,
    epoch,
    evaluate_samples_per_worker,
    sel_cuts_percent,
    device,
    evaluate_decode_type,
    seed,
    mean_std,
    policy_type,
    random_seed
):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    device = 'cuda:0'
    policy = policy.to(device)
    log_prefix = os.getpid()
    _ = set_global_seed(random_seed)
    logger.log(f"{log_prefix}: debug log random seed {random_seed}")
    logger.log(f"{log_prefix}: evaluating...  epoch: {epoch}")
    neg_solving_time = np.zeros((evaluate_samples_per_worker, 1))
    neg_total_nodes = np.zeros((evaluate_samples_per_worker, 1))
    primaldualintegral = np.zeros((evaluate_samples_per_worker, 1))
    primal_dual_gap = np.zeros((evaluate_samples_per_worker,1))
    lp_solution_value = []
    env.set_seed(seed)
    for i in range(evaluate_samples_per_worker):
        env.reset()
        cutsel_agent = CutSelectAgent(
            env.m,
            policy,
            value,
            sel_cuts_percent,
            device,
            evaluate_decode_type,
            mean_std,
            policy_type
        )
        env_step_info = env.step(cutsel_agent)
        lp_info = cutsel_agent.get_lp_info()
        neg_solving_time[i,:] = env_step_info['solving_time']
        neg_total_nodes[i,:] = env_step_info['ntotal_nodes']
        primaldualintegral[i,:] = env_step_info['primaldualintegral']
        primal_dual_gap[i,:] = env_step_info['primal_dual_gap']
        if len(lp_info['lp_solution_value']) >= 2:
            lp_solution_value.append(lp_info['lp_solution_value'][0] - lp_info['lp_solution_value'][1])
    return_queue.put(
        (neg_solving_time, neg_total_nodes, primaldualintegral, lp_solution_value,primal_dual_gap)
    )

def evaluate_hierarchy(
    return_queue,
    env,
    policy,
    cutsel_percent_policy,
    value,
    epoch,
    evaluate_samples_per_worker,
    sel_cuts_percent,
    device,
    evaluate_decode_type,
    seed,
    mean_std,
    policy_type,
    random_seed
):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    device = 'cuda:0'
    policy = policy.to(device)
    cutsel_percent_policy = cutsel_percent_policy.to(device)
    _ = set_global_seed(random_seed)
    log_prefix = os.getpid()
    logger.log(f"{log_prefix}: debug log random seed {random_seed}")
    logger.log(f"{log_prefix}: evaluating...  epoch: {epoch}")
    neg_solving_time = np.zeros((evaluate_samples_per_worker, 1))
    neg_total_nodes = np.zeros((evaluate_samples_per_worker, 1))
    primaldualintegral = np.zeros((evaluate_samples_per_worker, 1))
    primal_dual_gap = np.zeros((evaluate_samples_per_worker,1))
    lp_solution_value = []
    env.set_seed(seed)
    for i in range(evaluate_samples_per_worker):
        env.reset()
        cutsel_agent = HierarchyCutSelectAgent(
            env.m,
            policy,
            cutsel_percent_policy,
            value,
            sel_cuts_percent,
            device,
            evaluate_decode_type,
            mean_std,
            policy_type
        )
        env_step_info = env.step(cutsel_agent)
        lp_info = cutsel_agent.get_lp_info()
        neg_solving_time[i,:] = env_step_info['solving_time']
        neg_total_nodes[i,:] = env_step_info['ntotal_nodes']
        primaldualintegral[i,:] = env_step_info['primaldualintegral']
        primal_dual_gap[i,:] = env_step_info['primal_dual_gap']
        if len(lp_info['lp_solution_value']) >= 2:
            lp_solution_value.append(lp_info['lp_solution_value'][0] - lp_info['lp_solution_value'][1])
    return_queue.put(
        (neg_solving_time, neg_total_nodes, primaldualintegral, lp_solution_value,primal_dual_gap)
    )

def test(
    return_queue,
    instance_path,
    instance_file_list,
    policy,
    sel_cuts_percent,
    device,
    test_decode_type,
    seed,
    mean_std,
    policy_type,
    scip_seed,
    **env_kwargs
):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    device = 'cuda:0'
    policy = policy.to(device)
    _ = set_global_seed(seed)
    print(f"pid: {os.getpid()} debug log random seed {seed}")
    print(f"pid: {os.getpid()}, instance_files: {instance_file_list}")
    neg_solving_time = np.zeros((len(instance_file_list), 1))
    neg_total_nodes = np.zeros((len(instance_file_list), 1))
    primaldualintegral = np.zeros((len(instance_file_list), 1))
    primal_dual_gap = np.zeros((len(instance_file_list), 1))
    sel_cuts_info = {
        'sel_cuts_num': [],
        'cuts_total_num': []
    }
    f_name_list = []
    for i, f_name in enumerate(instance_file_list):
        env_kwargs['single_instance_file'] = f_name
        env = SCIPCutSelEnv(
            instance_path,
            scip_seed,
            seed,
            **env_kwargs
        )
        env.reset()
        cutsel_agent = CutSelectAgent(
            env.m,
            policy,
            None,
            sel_cuts_percent,
            device,
            test_decode_type,
            mean_std,
            policy_type
        )
        env_step_info = env.step(cutsel_agent)
        state_action_dict = cutsel_agent.get_data()

        neg_solving_time[i,:] = env_step_info['solving_time']
        neg_total_nodes[i,:] = env_step_info['ntotal_nodes']
        primaldualintegral[i,:] = env_step_info['primaldualintegral']
        primal_dual_gap[i,:] = env_step_info['primal_dual_gap']
        f_name_list.append(f_name)
        if not state_action_dict:
            sel_cuts_info['sel_cuts_num'].append(1)
            sel_cuts_info['cuts_total_num'].append(1)
        else:
            sel_cuts_info['sel_cuts_num'].append(state_action_dict['sel_cuts_num'])
            sel_cuts_info['cuts_total_num'].append(len(state_action_dict['state']))

    return_queue.put(
        (neg_solving_time, neg_total_nodes,primaldualintegral,primal_dual_gap,f_name_list,sel_cuts_info)
    )

def test_hierarchy(
    return_queue,
    instance_path,
    instance_file_list,
    policy,
    cutsel_percent_policy,
    sel_cuts_percent,
    device,
    test_decode_type,
    seed,
    mean_std,
    policy_type,
    scip_seed,
    **env_kwargs
):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    device = 'cuda:0'
    policy = policy.to(device)
    cutsel_percent_policy = cutsel_percent_policy.to(device)
    _ = set_global_seed(seed)
    print(f"pid: {os.getpid()} debug log random seed {seed}")
    print(f"pid: {os.getpid()}, instance_files: {instance_file_list}")
    neg_solving_time = np.zeros((len(instance_file_list), 1))
    neg_total_nodes = np.zeros((len(instance_file_list), 1))
    primaldualintegral = np.zeros((len(instance_file_list), 1))
    primal_dual_gap = np.zeros((len(instance_file_list), 1))
    sel_cuts_info = {
        'sel_cuts_num': [],
        'cuts_total_num': []
    }
    f_name_list = []
    for i, f_name in enumerate(instance_file_list):
        env_kwargs['single_instance_file'] = f_name
        env = SCIPCutSelEnv(
            instance_path,
            scip_seed,
            seed,
            **env_kwargs
        )
        env.reset()
        cutsel_agent = HierarchyCutSelectAgent(
            env.m,
            policy,
            cutsel_percent_policy,
            None,
            sel_cuts_percent,
            device,
            test_decode_type,
            mean_std,
            policy_type
        )
        env_step_info = env.step(cutsel_agent)
        state_action_dict = cutsel_agent.get_data()

        neg_solving_time[i,:] = env_step_info['solving_time']
        neg_total_nodes[i,:] = env_step_info['ntotal_nodes']
        primaldualintegral[i,:] = env_step_info['primaldualintegral']
        primal_dual_gap[i,:] = env_step_info['primal_dual_gap']
        if not state_action_dict:
            sel_cuts_info['sel_cuts_num'].append(1)
            sel_cuts_info['cuts_total_num'].append(1)
        else:
            sel_cuts_info['sel_cuts_num'].append(state_action_dict['sel_cuts_num'])
            sel_cuts_info['cuts_total_num'].append(len(state_action_dict['state']))
        f_name_list.append(f_name)
    return_queue.put(
        (neg_solving_time, neg_total_nodes,primaldualintegral,primal_dual_gap,f_name_list,sel_cuts_info)
    )    
    
def online_test(
    return_queue,
    test_instance_path,
    instance_file_list,
    policy,
    sel_cuts_percent,
    device,
    test_decode_type,
    seed,
    mean_std,
    policy_type,
    scip_seed,
    random_seed,
    **test_env_kwargs
):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    device = 'cuda:0'
    policy = policy.to(device)
    _ = set_global_seed(random_seed)
    pid_num = os.getpid()
    logger.log(f"{pid_num}: debug log random seed {random_seed}")
    neg_solving_time = np.zeros((len(instance_file_list), 1))
    neg_total_nodes = np.zeros((len(instance_file_list), 1))
    primaldualintegral = np.zeros((len(instance_file_list), 1))
    primal_dual_gap = np.zeros((len(instance_file_list), 1))
    f_name_list = []
    for i, f_name in enumerate(instance_file_list):
        logger.log(f"pid: {pid_num}, instance: {f_name}")
        test_env_kwargs['single_instance_file'] = f_name
        env = SCIPCutSelEnv(
            test_instance_path,
            scip_seed,
            seed,
            **test_env_kwargs
        )
        env.reset()
        cutsel_agent = CutSelectAgent(
            env.m,
            policy,
            None,
            sel_cuts_percent,
            device,
            test_decode_type,
            mean_std,
            policy_type
        )
        env_step_info = env.step(cutsel_agent)
        neg_solving_time[i,:] = env_step_info['solving_time']
        neg_total_nodes[i,:] = env_step_info['ntotal_nodes']
        primaldualintegral[i,:] = env_step_info['primaldualintegral']
        primal_dual_gap[i,:] = env_step_info['primal_dual_gap']
        f_name_list.append(f_name)
    return_queue.put(
        (neg_solving_time, neg_total_nodes,primaldualintegral,primal_dual_gap,f_name_list)
    )    

def online_test_hierarchy(
    return_queue,
    test_instance_path,
    instance_file_list,
    policy,
    cutsel_percent_policy,
    sel_cuts_percent,
    device,
    test_decode_type,
    seed,
    mean_std,
    policy_type,
    scip_seed,
    random_seed,
    **test_env_kwargs
):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    device = 'cuda:0'
    policy = policy.to(device)
    cutsel_percent_policy = cutsel_percent_policy.to(device)
    _ = set_global_seed(random_seed)
    pid_num = os.getpid()
    logger.log(f"{pid_num}: debug log random seed {random_seed}")
    neg_solving_time = np.zeros((len(instance_file_list), 1))
    neg_total_nodes = np.zeros((len(instance_file_list), 1))
    primaldualintegral = np.zeros((len(instance_file_list), 1))
    primal_dual_gap = np.zeros((len(instance_file_list), 1))
    f_name_list = []
    for i, f_name in enumerate(instance_file_list):
        logger.log(f"pid: {pid_num}, instance: {f_name}")
        test_env_kwargs['single_instance_file'] = f_name
        env = SCIPCutSelEnv(
            test_instance_path,
            scip_seed,
            seed,
            **test_env_kwargs
        )
        env.reset()
        cutsel_agent = HierarchyCutSelectAgent(
            env.m,
            policy,
            cutsel_percent_policy,
            None,
            sel_cuts_percent,
            device,
            test_decode_type,
            mean_std,
            policy_type
        )
        env_step_info = env.step(cutsel_agent)
        neg_solving_time[i,:] = env_step_info['solving_time']
        neg_total_nodes[i,:] = env_step_info['ntotal_nodes']
        primaldualintegral[i,:] = env_step_info['primaldualintegral']
        primal_dual_gap[i,:] = env_step_info['primal_dual_gap']
        f_name_list.append(f_name)
    return_queue.put(
        (neg_solving_time, neg_total_nodes,primaldualintegral,primal_dual_gap,f_name_list)
    )    

def process_and_log_results(raw_results,instance_type,out_dir,cutsel_rule,sel_cuts_percent,log_prefix):
    neg_solving_time = np.vstack([result[0] for result in raw_results])
    neg_total_nodes = np.vstack([result[1] for result in raw_results])
    primaldualintegral = np.vstack([result[2] for result in raw_results])
    primal_dual_gap = np.vstack([result[3] for result in raw_results])
    f_name_list = []
    sel_cuts_num = []
    cuts_total_num = []

    for result in raw_results:
        f_name_list.extend(result[4])
    for result in raw_results:
        sel_cuts_num.extend(result[5]['sel_cuts_num'])
        cuts_total_num.extend(result[5]['cuts_total_num'])
    
    new_results = {
        'solving_time': neg_solving_time,
        'neg_total_nodes': neg_total_nodes,
        "primaldualintegral": primaldualintegral,
        "primal_dual_gap": primal_dual_gap,
        "f_name_list": f_name_list,
        "sel_cuts_num": sel_cuts_num,
        "cuts_total_num": cuts_total_num
    }
    print(f"solving time mean: {np.mean(neg_solving_time)}, and std: {np.std(neg_solving_time)}")
    print(f"total nodes mean: {np.mean(neg_total_nodes)}, and total nodes std: {np.std(neg_total_nodes)}")
    print(f"primal dual integral mean: {np.mean(primaldualintegral)}, and std: {np.std(primaldualintegral)}")
    print(f"primal dual gap mean: {np.mean(primal_dual_gap)}, and std: {np.std(primal_dual_gap)}")
    print(f"sel_cuts_num mean: {np.mean(sel_cuts_num)}, and std: {np.std(sel_cuts_num)}")
    print(f"cuts_total_num mean: {np.mean(cuts_total_num)}, and std: {np.std(cuts_total_num)}")
    
    out_dir = instance_type + out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_npy = f"{out_dir}/{log_prefix}_{cutsel_rule}_max_cuts_root_{sel_cuts_percent}.npy"
    np.save(save_npy, new_results)

def online_process_and_log_results(raw_results, epoch, test_type):
    neg_solving_time = np.vstack([result[0] for result in raw_results])
    neg_total_nodes = np.vstack([result[1] for result in raw_results])
    primaldualintegral = np.vstack([result[2] for result in raw_results])
    primal_dual_gap = np.vstack([result[3] for result in raw_results])
    f_name_list = []
    for result in raw_results:
        f_name_list.extend(result[4])
    stats = {}
    if test_type == 'online_test':
        prefix = 'testing'
    else:
        prefix = 'evaluating'
    stats.update(
        create_stats_ordered_dict(f'{prefix}/solving time', neg_solving_time)
    )
    stats.update(
        create_stats_ordered_dict(f'{prefix}/neg_total_nodes', neg_total_nodes)
    )
    stats.update(
        create_stats_ordered_dict(f'{prefix}/primaldualintegral', primaldualintegral)
    )
    stats.update(
        create_stats_ordered_dict(f'{prefix}/primal_dual_gap', primal_dual_gap)
    )
    new_results = {
        'solving_time': neg_solving_time,
        'neg_total_nodes': neg_total_nodes,
        "primaldualintegral": primaldualintegral,
        "primal_dual_gap": primal_dual_gap,
        "f_name_list": f_name_list
    }
    if test_type == 'online_test':
        logger.save_npy(epoch, new_results)
    logger.log(f"solving time mean: {np.mean(neg_solving_time)}, and std: {np.std(neg_solving_time)}")
    logger.log(f"total nodes mean: {np.mean(neg_total_nodes)}, and total nodes std: {np.std(neg_total_nodes)}")
    logger.log(f"primal dual integral mean: {np.mean(primaldualintegral)}, and std: {np.std(primaldualintegral)}")
    logger.log(f"primal_dual_gap mean: {np.mean(primal_dual_gap)}, and std: {np.std(primal_dual_gap)}")

    return stats

def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times

def main():
    torch.multiprocessing.set_start_method('spawn', force=True)
    from pointer_net import PointerNetwork, CutsPercentPolicy
    from pointer_net_end_token import PointerNetworkEndToken
    from value_net import CriticNetwork
    # 参数配置：固定参数json 文件；调试参数命令行
    parser = argparse.ArgumentParser(description="RL for learning to cut")
    parser.add_argument('--config_file', type=str, default='/datasets/learning_to_cut_via_rl/configs/easy_max_independent_set_config.json', help="base config json dir")
    parser.add_argument('--sel_cuts_percent', type=float, default=0.1)
    parser.add_argument('--single_instance_file', type=str, default="all")  
    parser.add_argument('--reward_type', type=str, default="lp_solution_value")
    parser.add_argument('--baseline_type', type=str, default="simple")
    parser.add_argument('--train_type', type=str, default="train")
    parser.add_argument('--instance_type', type=str, default="item_placement") # for log file name 
    parser.add_argument('--time_limit', type=int, default=10) # for log file name 
    parser.add_argument('--use_cutsel_percent_policy', type=str, default='False')
    parser.add_argument('--policy_type', type=str, default='with_token')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--scip_seed', type=int, default=1)

    args = parser.parse_args()
    all_kwargs = json.load(open(args.config_file, 'r'))
    if args.use_cutsel_percent_policy == 'True':
        all_kwargs['cutsel_percent_policy']['use_cutsel_percent_policy'] = True
    else:
        all_kwargs['cutsel_percent_policy']['use_cutsel_percent_policy'] = False
    all_kwargs['policy_type'] = args.policy_type
    if args.policy_type == 'with_token':
        Pointer = PointerNetworkEndToken
    else:
        Pointer = PointerNetwork
    # test 
    if args.train_type == 'test':
        test_kwargs = all_kwargs['test_kwargs']
        device_kwargs = all_kwargs['devices']
        # get instance file path
        test_instance_path = test_kwargs['test_instance_path']
        f_name_list = os.listdir(test_instance_path)
        # assert len(f_name_list) % test_kwargs['n_jobs'] == 0
        file_num_each_worker = math.ceil(len(f_name_list) / (test_kwargs['n_jobs'] * len(device_kwargs['multi_devices'])))

        env_kwargs = all_kwargs['env']
        env_kwargs.pop('instance_file_path')
        all_kwargs['experiment']['seed'] = args.seed
        seed = set_global_seed(all_kwargs['experiment']['seed'])

        # load policy model
        device = torch.device(device_kwargs['global_device'])
        # multi_devices = [torch.device(d) for d in device_kwargs['multi_devices']]
        multi_devices = device_kwargs['multi_devices']

        net_share_kwargs = all_kwargs['net_share']
        policy_kwargs = all_kwargs['policy']
        value_kwargs = all_kwargs['value']
        cutsel_percent_policy_kwargs = all_kwargs['cutsel_percent_policy']

        test_model_base_path = test_kwargs['test_model_base_path']
        test_model_file = test_kwargs['test_model']
        if len(test_model_file) == 1:
            state_dict = torch.load(os.path.join(test_model_base_path, test_model_file[0]))
        else:
            list_state_dict = [torch.load(os.path.join(test_model_base_path, cur_test_model_file)) for cur_test_model_file in test_model_file]
            state_dict = get_average_models(list_state_dict)
        # load policy
        policy = Pointer(
            embedding_dim=net_share_kwargs['embedding_dim'],
            hidden_dim=net_share_kwargs['hidden_dim'],
            n_glimpses=policy_kwargs['n_glimpses'],
            tanh_exploration=net_share_kwargs['tanh_exploration'],
            use_tanh=net_share_kwargs['use_tanh'],
            beam_size=policy_kwargs['beam_size'],
            use_cuda=torch.cuda.is_available()
        )
        # .to(device)
        policy.load_state_dict(state_dict['pointer_net'])
        policy.eval()
        # load running mean std
        feature_shape = (policy.embedding_dim,)
        if 'mean' in state_dict.keys():
            mean_std = RunningMeanStd(feature_shape)
            mean_std.set_mean_std(state_dict['mean'], state_dict['std'])
        else:
            mean_std = None
        if cutsel_percent_policy_kwargs['use_cutsel_percent_policy']:
            cutsel_percent_policy = CutsPercentPolicy(
                embedding_dim=net_share_kwargs['embedding_dim'],
                hidden_dim=net_share_kwargs['hidden_dim'],
                n_process_block_iters=value_kwargs['n_process_block_iters'],
                tanh_exploration=net_share_kwargs['tanh_exploration'],
                use_tanh=net_share_kwargs['use_tanh'],
                use_cuda=torch.cuda.is_available()
            )
            # .to(device)
            cutsel_percent_policy.load_state_dict(state_dict['cutsel_percent_net'])
            cutsel_percent_policy.eval()
        # running multiprocessing
        return_queue = mp.SimpleQueue()
        processes = []
        for i, worker_device in enumerate(multi_devices):
            st_index = i * test_kwargs['n_jobs']
            for num in range(test_kwargs['n_jobs']):
                if i == (len(multi_devices)-1) and num == (test_kwargs['n_jobs']-1):
                    cur_f_list = f_name_list[(st_index+num)*file_num_each_worker:]
                else:
                    cur_f_list = f_name_list[(st_index+num)*file_num_each_worker:(st_index+num+1)*file_num_each_worker]
                if cutsel_percent_policy_kwargs['use_cutsel_percent_policy']:
                    p = mp.Process(
                        target=test_hierarchy,
                        args=(return_queue,test_instance_path,cur_f_list,policy,cutsel_percent_policy,args.sel_cuts_percent,worker_device,'greedy',seed,mean_std,args.policy_type,args.scip_seed),
                        kwargs=env_kwargs
                    )
                else:
                    p = mp.Process(
                        target=test,
                        args=(return_queue,test_instance_path,cur_f_list,policy,args.sel_cuts_percent,worker_device,'greedy',seed,mean_std,args.policy_type,args.scip_seed),
                        kwargs=env_kwargs
                    )
                p.start()
                processes.append(p)
        raw_results = [return_queue.get() for p in processes] # list of tuple
        for p in processes:
            p.join()   
        # for p in processes:
        #     p.close()     
        log_prefix = f"seed_{args.scip_seed}_model_{all_kwargs['test_kwargs']['test_model']}"
        process_and_log_results(raw_results, args.instance_type + '_use_hrl_' + str(cutsel_percent_policy_kwargs['use_cutsel_percent_policy']), "heuristics_cutsel", "RL", args.sel_cuts_percent,log_prefix)

        # get model
        # test
        # process test results
    elif args.train_type == 'train':
        # preprocess config from cmd 
        all_kwargs['experiment']['exp_prefix'] = all_kwargs['experiment']['exp_prefix'] + '_' + args.single_instance_file + '_' + args.instance_type
        all_kwargs['env']['single_instance_file'] = args.single_instance_file
        all_kwargs['algorithm']['reward_type'] = args.reward_type
        all_kwargs['algorithm']['baseline_type'] = args.baseline_type
        if args.reward_type == "lp_solution_value":
            all_kwargs['env']['max_rounds_root'] = 2
            all_kwargs['env']['scip_time_limit'] = args.time_limit
        
        all_kwargs['parser_args'] = dict(vars(args))
        experiment_kwargs = all_kwargs['experiment']
        seed = set_global_seed(experiment_kwargs['seed'])
        all_kwargs['experiment']['seed'] = seed

        # init logger 
        logger.reset()
        trainer_kwargs = all_kwargs['trainer']
        variant = copy.deepcopy(all_kwargs)
        actual_log_dir = setup_logger(
            variant=variant,
            **experiment_kwargs
        )

        # data or environments 如何生成 
        # 离线生成一系列train instances valid instances test instances: 训练集训练，验证集调参，测试集测试
        # train: 10000 instances; valid: 2000 instances; test: 20 instances GCNN 的配置，为何test instance 数目这么小
        env_kwargs = all_kwargs['env']
        instance_file_path = env_kwargs.pop('instance_file_path')
        env = SCIPCutSelEnv(
            instance_file_path,
            args.scip_seed,
            seed,
            **env_kwargs
        )

        # cutsel agent
        device_kwargs = all_kwargs['devices']
        device = torch.device(device_kwargs['global_device'])
        # worker_devices = [torch.device(worker_device) for worker_device in device_kwargs['multi_devices']]
        worker_devices = device_kwargs['multi_devices']

        net_share_kwargs = all_kwargs['net_share']
        policy_kwargs = all_kwargs['policy']
        value_kwargs = all_kwargs['value']
        cutsel_percent_policy_kwargs = all_kwargs['cutsel_percent_policy']

        pointer_net = Pointer(
            embedding_dim=net_share_kwargs['embedding_dim'],
            hidden_dim=net_share_kwargs['hidden_dim'],
            n_glimpses=policy_kwargs['n_glimpses'],
            tanh_exploration=net_share_kwargs['tanh_exploration'],
            use_tanh=net_share_kwargs['use_tanh'],
            beam_size=policy_kwargs['beam_size'],
            use_cuda=torch.cuda.is_available()
        )
        # .to(device)

        if all_kwargs['algorithm']['baseline_type'] == 'net':
            value_net = CriticNetwork(
                embedding_dim=net_share_kwargs['embedding_dim'],
                hidden_dim=net_share_kwargs['hidden_dim'],
                n_process_block_iters=value_kwargs['n_process_block_iters'],
                tanh_exploration=net_share_kwargs['tanh_exploration'],
                use_tanh=net_share_kwargs['use_tanh'],
                use_cuda=torch.cuda.is_available()
            )
            # .to(device)
        else:
            value_net = None

        if cutsel_percent_policy_kwargs['use_cutsel_percent_policy']:
            cutsel_percent_policy = CutsPercentPolicy(
                embedding_dim=net_share_kwargs['embedding_dim'],
                hidden_dim=net_share_kwargs['hidden_dim'],
                n_process_block_iters=value_kwargs['n_process_block_iters'],
                tanh_exploration=net_share_kwargs['tanh_exploration'],
                use_tanh=net_share_kwargs['use_tanh'],
                use_cuda=torch.cuda.is_available()
            )
            # .to(device)

        # preload model for retraining
        if experiment_kwargs['base_log_dir'] is not None and 'params.pkl' in os.listdir(experiment_kwargs['base_log_dir']):
            state_dict = torch.load(os.path.join(experiment_kwargs['base_log_dir'], 'params.pkl'))
            pointer_net.load_state_dict(state_dict['pointer_net'])
            if cutsel_percent_policy_kwargs['use_cutsel_percent_policy']:
                cutsel_percent_policy.load_state_dict(state_dict['cutsel_percent_net'])
        # train 函数
        alg_kwargs = all_kwargs['algorithm']
        if cutsel_percent_policy_kwargs['use_cutsel_percent_policy']:
            algorithm = HRLReinforceAlg(
                env,
                pointer_net,
                value_net,
                cutsel_percent_policy,
                args.sel_cuts_percent,
                device,
                cutsel_percent_policy_kwargs['train_freq'],
                cutsel_percent_policy_kwargs['train_highlevel_batch_size'],
                cutsel_percent_policy_kwargs['highlevel_actor_lr'],
                **alg_kwargs
            )
        else:
            algorithm = ReinforceBaselineAlg(
                env,
                pointer_net,
                value_net,
                args.sel_cuts_percent,
                device,
                **alg_kwargs
            )
        if alg_kwargs['normalize']:
            mean_std = algorithm.mean_std
        else:
            mean_std = None
    
        gt.reset_root()
        test_stats = {}
        eva_stats = {}
        train_highlevel_stats = {}
        tmp_stats = {}
        # training loop
        for epoch in gt.timed_for(range(all_kwargs['start_epoch'], alg_kwargs['num_epochs']), save_itrs=True):
        # for epoch in range(alg_kwargs['num_epochs']):
            # samples per epoch 
            # mini_batchsize
            # n_jobs
            # multiprocessing sampling data 
            pointer_net = pointer_net.to('cpu') # policy to cpu
            if cutsel_percent_policy_kwargs['use_cutsel_percent_policy']:
                cutsel_percent_policy = cutsel_percent_policy.to('cpu')

            assert trainer_kwargs['samples_per_epoch'] % trainer_kwargs['n_jobs'] == 0
            samples_each_worker = int(trainer_kwargs['samples_per_epoch'] / trainer_kwargs['n_jobs'])

            train_multiprocess_seeds = [np.random.randint(2 ** 30) for _ in range(trainer_kwargs['n_jobs'])]
            eval_multiprocess_seeds = [np.random.randint(2 ** 30) for _ in range(trainer_kwargs['n_jobs'])]
            for i, s in enumerate(train_multiprocess_seeds):
                logger.record_tabular(f'train {i+1}th process seed', s)
            for i, s in enumerate(eval_multiprocess_seeds):
                logger.record_tabular(f'evaluate {i+1}th process seed', s)
            
            # online testing ............
            online_test_kwargs = all_kwargs['online_test_kwargs']
            if online_test_kwargs['test_freq'] > 0 and (epoch % online_test_kwargs['test_freq'] == 0):
                logger.log(f"testing... epoch: {epoch+1}")
                # get instance file path
                test_instance_path = online_test_kwargs['test_instance_path']
                f_name_list = os.listdir(test_instance_path)
                # assert len(f_name_list) % test_kwargs['n_jobs'] == 0
                file_num_each_worker = math.ceil(len(f_name_list) / (online_test_kwargs['test_n_jobs'] * len(worker_devices)))
                test_env_kwargs = online_test_kwargs['test_env_kwargs']
                return_queue = mp.SimpleQueue()
                processes = []
                
                for i, worker_device in enumerate(worker_devices):
                    st_index = i * online_test_kwargs['test_n_jobs']
                    for num in range(online_test_kwargs['test_n_jobs']):
                        if i == (len(worker_devices)-1) and num == (online_test_kwargs['test_n_jobs']-1):
                            cur_f_list = f_name_list[(st_index+num)*file_num_each_worker:]
                        else:
                            cur_f_list = f_name_list[(st_index+num)*file_num_each_worker:(st_index+num+1)*file_num_each_worker]
                        # cur_policy = policy.to(worker_device)
                        if cutsel_percent_policy_kwargs['use_cutsel_percent_policy']:
                            p = mp.Process(
                                target=online_test_hierarchy,
                                args=(return_queue,test_instance_path,cur_f_list,pointer_net,cutsel_percent_policy,args.sel_cuts_percent,worker_device,'greedy',seed,mean_std,args.policy_type,args.scip_seed,seed),
                                kwargs=test_env_kwargs
                            )
                        else:
                            p = mp.Process(
                                target=online_test,
                                args=(return_queue,test_instance_path,cur_f_list,pointer_net,args.sel_cuts_percent,worker_device,'greedy',seed,mean_std,args.policy_type,args.scip_seed,seed),
                                kwargs=test_env_kwargs
                            )
                        p.start()
                        processes.append(p)
                raw_results = [return_queue.get() for p in processes] # list of tuple
                for p in processes:
                    p.join()        
                test_stats = online_process_and_log_results(raw_results, epoch, 'online_test')
            if test_stats:
                logger.record_dict(test_stats)
            gt.stamp('online_testing', unique=False)
            # evaluating ..........
            if  (alg_kwargs['evaluate_freq'] > 0) and (epoch % alg_kwargs['evaluate_freq'] == 0):
                # logger.log(f"evaluating...  epoch: {epoch+1}")
                # assert alg_kwargs['evaluate_samples'] % trainer_kwargs['n_jobs'] == 0
                # evaluate_sample_each_worker = int(alg_kwargs['evaluate_samples'] / trainer_kwargs['n_jobs'])
                # return_queue = mp.SimpleQueue()
                # processes = []
                # for i, worker_device in enumerate(worker_devices):
                #     for num in range(trainer_kwargs['n_jobs']):
                #         s = eval_multiprocess_seeds[num] + i
                #         if cutsel_percent_policy_kwargs['use_cutsel_percent_policy']:
                #             p = mp.Process(
                #                 target=evaluate_hierarchy,
                #                 args=(return_queue,env,pointer_net,cutsel_percent_policy,value_net,epoch+1,evaluate_sample_each_worker,args.sel_cuts_percent,worker_device,alg_kwargs['evaluate_decode_type'],s,mean_std,args.policy_type,seed)
                #             )
                #         else:
                #             p = mp.Process(
                #                 target=evaluate,
                #                 args=(return_queue,env,pointer_net,value_net,epoch+1,evaluate_sample_each_worker,args.sel_cuts_percent,worker_device,alg_kwargs['evaluate_decode_type'],s,mean_std,args.policy_type,seed)
                #             )
                #         p.start()
                #         processes.append(p)
                # evaluate_results = [return_queue.get() for p in processes] # list of tuple
                # for p in processes:
                #     p.join()
                # eva_stats = algorithm.log_evaluate_stats(evaluate_results)

                evaluate_kwargs = all_kwargs['evaluate_kwargs']
                logger.log(f"evaluating...  epoch: {epoch+1}")
                # get instance file path
                test_instance_path = evaluate_kwargs['test_instance_path']
                f_name_list = os.listdir(test_instance_path)
                # assert len(f_name_list) % test_kwargs['n_jobs'] == 0
                file_num_each_worker = math.ceil(len(f_name_list) / (evaluate_kwargs['test_n_jobs'] * len(worker_devices)))
                test_env_kwargs = evaluate_kwargs['test_env_kwargs']
                return_queue = mp.SimpleQueue()
                processes = []
                
                for i, worker_device in enumerate(worker_devices):
                    st_index = i * evaluate_kwargs['test_n_jobs']
                    for num in range(evaluate_kwargs['test_n_jobs']):
                        if i == (len(worker_devices)-1) and num == (evaluate_kwargs['test_n_jobs']-1):
                            cur_f_list = f_name_list[(st_index+num)*file_num_each_worker:]
                        else:
                            cur_f_list = f_name_list[(st_index+num)*file_num_each_worker:(st_index+num+1)*file_num_each_worker]
                        # cur_policy = policy.to(worker_device)
                        if cutsel_percent_policy_kwargs['use_cutsel_percent_policy']:
                            p = mp.Process(
                                target=online_test_hierarchy,
                                args=(return_queue,test_instance_path,cur_f_list,pointer_net,cutsel_percent_policy,args.sel_cuts_percent,worker_device,'greedy',seed,mean_std,args.policy_type,args.scip_seed,seed),
                                kwargs=test_env_kwargs
                            )
                        else:
                            p = mp.Process(
                                target=online_test,
                                args=(return_queue,test_instance_path,cur_f_list,pointer_net,args.sel_cuts_percent,worker_device,'greedy',seed,mean_std,args.policy_type,args.scip_seed,seed),
                                kwargs=test_env_kwargs
                            )
                        p.start()
                        processes.append(p)
                raw_results = [return_queue.get() for p in processes] # list of tuple
                for p in processes:
                    p.join()        
                eva_stats = online_process_and_log_results(raw_results, epoch, 'evaluate')

            if eva_stats:
                logger.record_dict(eva_stats)
            gt.stamp('evaluating', unique=False)
            ####################################################
            # sampling ........
            logger.log(f"training...  epoch: {epoch+1}")
            return_queue = mp.SimpleQueue()
            processes = []
            for i, worker_device in enumerate(worker_devices):
                for num in range(trainer_kwargs['n_jobs']):
                    s = train_multiprocess_seeds[num] + i
                    if cutsel_percent_policy_kwargs['use_cutsel_percent_policy']:
                        p = mp.Process(
                            target=generate_hierarchy_samples,
                            args=(return_queue,env,pointer_net,cutsel_percent_policy,value_net,epoch+1,samples_each_worker,args.sel_cuts_percent,worker_device,alg_kwargs['train_decode_type'],alg_kwargs['reward_type'],s,mean_std,args.policy_type,seed)
                        )
                    else:     
                        p = mp.Process(
                            target=generate_samples,
                            args=(return_queue,env,pointer_net,value_net,epoch+1,samples_each_worker,args.sel_cuts_percent,worker_device,alg_kwargs['train_decode_type'],alg_kwargs['reward_type'],s,mean_std,args.policy_type,seed)
                        )
                    p.start()
                    processes.append(p)
            raw_results = [return_queue.get() for p in processes] # list of tuple
            for p in processes:
                p.join()
            gt.stamp('sampling data', unique=False)

            # training policy and value with data 
            pointer_net = pointer_net.to(device)
            if cutsel_percent_policy_kwargs['use_cutsel_percent_policy']:
                cutsel_percent_policy = cutsel_percent_policy.to(device)
                # train_highlevel_stats = algorithm.train_highlevel_policy(raw_results, epoch)
                tmp_stats = algorithm.train_highlevel_policy(raw_results, epoch)
            if tmp_stats:
                train_highlevel_stats = tmp_stats
            if train_highlevel_stats:
                logger.record_dict(train_highlevel_stats)
            algorithm.train(raw_results, epoch+1)
            # 释放torch cuda 缓存
            torch.cuda.empty_cache()    
            gt.stamp('training', unique=False)

            # log timing data
            logger.record_dict(_get_epoch_timings())
            # write log to file 
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

if __name__ == '__main__':
    main()


