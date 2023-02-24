import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import time
import pyscipopt as scip
from pyscipopt import SCIP_RESULT

# from beam_search import Beam
from utils import cut_feature_generator, advanced_cut_feature_generator
# from utils_fix_isp_bug import cut_feature_generator
from logger import logger

class CutSelectAgent(scip.Cutsel):
    def __init__(
        self,
        scip_model,
        pointer_net,
        value_net,
        sel_cuts_percent,
        device,
        decode_type,
        mean_std,
        policy_type
    ):
        super().__init__()
        self.scip_model = scip_model
        self.policy = pointer_net
        self.value = value_net
        self.sel_cuts_percent = sel_cuts_percent
        self.device = device
        self.decode_type = decode_type
        self.policy_type = policy_type

        self.data = {}
        # self.cuts_info ={}
        self.lp_info = {
            "lp_solution_value": [],
            "lp_solution_integer_var_value": []
        }
        self.mean_std = mean_std

    def _normalize(self, cuts_features):
        # print(f"debug log mean: {self.mean_std.mean}, std: {self.mean_std.std}")
        return (cuts_features-self.mean_std.mean) / (self.mean_std.std + self.mean_std.epsilon)
    
    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        if self.policy_type == 'with_token':
            cuts_dict = self._cutselselect_with_token(cuts, forcedcuts, root, maxnselectedcuts)
        else:
            cuts_dict = self._cutselselect(cuts, forcedcuts, root, maxnselectedcuts)  

        return cuts_dict 
    
    def _cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        '''first method called in each iteration in the main solving loop. '''
        # this method needs to be implemented by the user
        logger.log("cut selection policy without token!!!")
        logger.log(f"forcedcuts length: {len(forcedcuts)}")
        logger.log(f"len cuts: {len(cuts)}")
        num_cuts = len(cuts)
        # cur_lp_info = self._get_lp_info()
        # for k in cur_lp_info.keys():
        #     self.lp_info[k].append(cur_lp_info[k])
        if num_cuts <= 1:
            return {
                'cuts': cuts, # selected sorted cuts
                'nselectedcuts': 1, # num of selected cuts
                'result': SCIP_RESULT.SUCCESS
            }            
        sel_cuts_num = min(int(num_cuts * self.sel_cuts_percent), int(maxnselectedcuts))
        sel_cuts_num = max(sel_cuts_num, 2)
        st_before_input = time.time()
        cuts_features = advanced_cut_feature_generator(self.scip_model, cuts)
        et_feature_extractor = time.time()
        if self.mean_std is not None:
            # normalize cut features
            normalize_cut_features = self._normalize(cuts_features)
            input_cuts = torch.from_numpy(normalize_cut_features).to(self.device)
        else:
            input_cuts = torch.from_numpy(cuts_features).to(self.device)
        
        input_cuts = input_cuts.reshape(input_cuts.shape[0], 1, input_cuts.shape[1])
        st_end_input = time.time()
        # 只做选择动作的功能，不做计算梯度的功能
        with torch.no_grad():
            _, input_idxs =  self.policy(input_cuts.float(), sel_cuts_num, self.decode_type) # (list of tensor, list of tensor)
        st_end_inference = time.time()
        print(f"process input time: {st_end_input-st_before_input} s")
        print(f"input feature extractor time: {et_feature_extractor-st_before_input} s")
        print(f"input cpu data to gpu time: {st_end_input-et_feature_extractor} s")
        print(f"pointer net inference time: {st_end_inference - st_end_input} s")
        idxes = [input.cpu().detach().item() for input in input_idxs]
        assert len(set(idxes))==len(idxes) # 保证选择的idxes 没有重复的！
        all_idxes = list(range(num_cuts))
        not_sel_idxes = list(set(all_idxes).difference(idxes))
        sorted_cuts = [cuts[idx] for idx in idxes]
        not_sel_cuts = [cuts[n_idx] for n_idx in not_sel_idxes]
        sorted_cuts.extend(not_sel_cuts)
        # debug
        # sorted_cuts = cuts
        # 只log 第一次cut 处的state 和 action
        if not self.data:
            self.data = {
                "state": cuts_features,
                "action": idxes,
                "sel_cuts_num": sel_cuts_num,
            }
            # self.cuts_info = {
            #     "length_cuts": num_cuts,
            #     "length_forced_cuts": len(forcedcuts),
            #     "cut_features": cuts_features
            # }

        return {
            'cuts': sorted_cuts, # selected sorted cuts
            'nselectedcuts': sel_cuts_num, # num of selected cuts
            'result': SCIP_RESULT.SUCCESS
        }

    def _cutselselect_with_token(self, cuts, forcedcuts, root, maxnselectedcuts):
        '''first method called in each iteration in the main solving loop. '''
        # this method needs to be implemented by the user
        logger.log("cut selection policy with token!!!")
        logger.log(f"forcedcuts length: {len(forcedcuts)}")
        logger.log(f"len cuts: {len(cuts)}")
        num_cuts = len(cuts)
        # cur_lp_info = self._get_lp_info()
        # for k in cur_lp_info.keys():
        #     self.lp_info[k].append(cur_lp_info[k])
        if num_cuts <= 1:
            return {
                'cuts': cuts, # selected sorted cuts
                'nselectedcuts': 1, # num of selected cuts
                'result': SCIP_RESULT.SUCCESS
            }            
        max_sel_cuts_num = len(cuts) + 1 
        st_before_input = time.time()
        cuts_features = advanced_cut_feature_generator(self.scip_model, cuts)
        et_feature_extractor = time.time()
        if self.mean_std is not None:
            # normalize cut features
            normalize_cut_features = self._normalize(cuts_features)
            input_cuts = torch.from_numpy(normalize_cut_features).to(self.device)
        else:
            input_cuts = torch.from_numpy(cuts_features).to(self.device)
        
        input_cuts = input_cuts.reshape(input_cuts.shape[0], 1, input_cuts.shape[1])
        st_end_input = time.time()
        # 只做选择动作的功能，不做计算梯度的功能
        with torch.no_grad():
            _, input_idxs =  self.policy(input_cuts.float(), max_sel_cuts_num, self.decode_type) # (list of tensor, list of tensor)
        st_end_inference = time.time()
        print(f"process input time: {st_end_input-st_before_input} s")
        print(f"input feature extractor time: {et_feature_extractor-st_before_input} s")
        print(f"input cpu data to gpu time: {st_end_input-et_feature_extractor} s")
        print(f"pointer net inference time: {st_end_inference - st_end_input} s")

        idxes = [input.cpu().detach().item() for input in input_idxs]
        sel_cuts_num = len(idxes)
        if not self.data:
            self.data = {
                "state": cuts_features,
                "action": idxes,
                "sel_cuts_num": sel_cuts_num,
            }
        # select cuts 
        assert idxes[-1] == num_cuts
        true_idxes = idxes[:-1] # remove the last index which is end token
        assert len(set(true_idxes))==len(true_idxes) # 保证选择的idxes 没有重复的！
        all_idxes = list(range(num_cuts))
        not_sel_idxes = list(set(all_idxes).difference(true_idxes))
        sorted_cuts = [cuts[idx] for idx in true_idxes]
        not_sel_cuts = [cuts[n_idx] for n_idx in not_sel_idxes]
        sorted_cuts.extend(not_sel_cuts)

        return {
            'cuts': sorted_cuts, # selected sorted cuts
            'nselectedcuts': sel_cuts_num-1, # num of selected cuts
            'result': SCIP_RESULT.SUCCESS
        }

    def _get_lp_info(self):
        lp_info = {}
        lp_info['lp_solution_value'] = self.scip_model.getLPObjVal()
        cols = self.scip_model.getLPColsData()
        col_solution_value = [col.getPrimsol() for col in cols if col.isIntegral()]
        lp_info['lp_solution_integer_var_value'] = [val for val in col_solution_value if val != 0.]

        return lp_info

    def get_data(self):
        return self.data

    def get_lp_info(self):
        return self.lp_info

    # def get_cuts_info(self):
    #     return self.cuts_info
        
    def free_problem(self):
        self.scip_model.freeProb()

class HierarchyCutSelectAgent(CutSelectAgent):
    def __init__(
        self,
        scip_model,
        pointer_net,
        cutsel_percent_policy,
        value_net,
        sel_cuts_percent,
        device,
        decode_type,
        mean_std,
        policy_type
    ):
        CutSelectAgent.__init__(
            self,
            scip_model,
            pointer_net,
            value_net,
            sel_cuts_percent,
            device,
            decode_type,
            mean_std,
            policy_type
        )
        self.cutsel_percent_policy = cutsel_percent_policy
        self.high_level_data = {}

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        '''first method called in each iteration in the main solving loop. '''
        # this method needs to be implemented by the user
        logger.log(f"forcedcuts length: {len(forcedcuts)}")
        logger.log(f"len cuts: {len(cuts)}")
        num_cuts = len(cuts)
        # cur_lp_info = self._get_lp_info()
        # for k in cur_lp_info.keys():
        #     self.lp_info[k].append(cur_lp_info[k])
        if num_cuts <= 1:
            return {
                'cuts': cuts, # selected sorted cuts
                'nselectedcuts': 1, # num of selected cuts
                'result': SCIP_RESULT.SUCCESS
            }
        
        st_before_input = time.time()

        # compute states
        cuts_features = advanced_cut_feature_generator(self.scip_model, cuts)
        et_feature_extractor = time.time()
        # normalize states
        if self.mean_std is not None:
            # normalize cut features
            normalize_cut_features = self._normalize(cuts_features)
            input_cuts = torch.from_numpy(normalize_cut_features).to(self.device)
        else:
            input_cuts = torch.from_numpy(cuts_features).to(self.device)
        input_cuts = input_cuts.reshape(input_cuts.shape[0], 1, input_cuts.shape[1])

        st_end_input = time.time()

        # compute sel cuts percent
        with torch.no_grad():
            if self.decode_type == 'greedy':
                deterministic = True
            else:
                deterministic = False
            raw_sel_cuts_percent = self.cutsel_percent_policy.action(input_cuts.float(), deterministic=deterministic)
        st_end_highlevel_policy_inference = time.time()

        sel_cuts_percent = raw_sel_cuts_percent.item() * 0.5 + 0.5
        sel_cuts_num = min(int(num_cuts * sel_cuts_percent), int(maxnselectedcuts))
        sel_cuts_num = max(sel_cuts_num, 2)
        # 只做选择动作的功能，不做计算梯度的功能
        with torch.no_grad():
            _, input_idxs =  self.policy(input_cuts.float(), sel_cuts_num, self.decode_type) # (list of tensor, list of tensor)
        st_end_pointer_net_inference = time.time()

        print(f"process input time: {st_end_input-st_before_input} s")
        print(f"input feature extractor time: {et_feature_extractor-st_before_input} s")
        print(f"input cpu data to gpu time: {st_end_input-et_feature_extractor} s")
        print(f"high level policy time: {st_end_highlevel_policy_inference-st_end_input} s")
        print(f"pointer net inference time: {st_end_pointer_net_inference - st_end_highlevel_policy_inference} s")

        idxes = [input.cpu().detach().item() for input in input_idxs]
        assert len(set(idxes))==len(idxes) # 保证选择的idxes 没有重复的！
        all_idxes = list(range(num_cuts))
        not_sel_idxes = list(set(all_idxes).difference(idxes))
        sorted_cuts = [cuts[idx] for idx in idxes]
        not_sel_cuts = [cuts[n_idx] for n_idx in not_sel_idxes]
        sorted_cuts.extend(not_sel_cuts)
        # debug
        # sorted_cuts = cuts
        # 只log 第一次cut 处的state 和 action
        if not self.data:
            self.data = {
                "state": cuts_features,
                "action": idxes,
                "sel_cuts_num": sel_cuts_num,
            }
        if not self.high_level_data:
            self.high_level_data = {
                "state": cuts_features,
                "action": raw_sel_cuts_percent.item()
            }

        return {
            'cuts': sorted_cuts, # selected sorted cuts
            'nselectedcuts': sel_cuts_num, # num of selected cuts
            'result': SCIP_RESULT.SUCCESS
        }

    def get_high_level_data(self):
        return self.high_level_data

## testing code
# if __name__ == '__main__':
#     from environments import SCIPCutSelEnv
#     from pointer_net import PointerNetwork
#     instance_file_path = "/datasets/learning_to_cut/dataset/data_nips_competition/instances/2_load_balancing/train/train_mps"
#     seed = 1
#     env_kwargs = {
#         "scip_time_limit": 30,
#         "single_instance_file": "all",
#         "presolving": True,
#         "separating": True,
#         "conflict": True, 
#         "heuristics": True,
#         "max_rounds_root": 1
#     }
#     env = SCIPCutSelEnv(
#         instance_file_path,
#         seed,
#         **env_kwargs
#     )  

#     device = torch.device('cuda:1')
#     pointer_net = PointerNetwork(
#         embedding_dim=13,
#         hidden_dim=128,
#         n_glimpses=1,
#         tanh_exploration=5,
#         use_tanh=True,
#         beam_size=1,
#         use_cuda=torch.cuda.is_available()
#     ).to(device)

#     for _ in range(10):
#         env.reset()
#         cutsel_agent = CutSelectAgent(
#             env.m,
#             pointer_net,
#             None,
#             0.5,
#             device,
#             'stochastic',
#             None,
#             'no_token'
#         )
#         _ = env.step(cutsel_agent)