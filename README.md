# Learning Cut Selection for Mixed-Integer Linear Programming via Hierarchical Sequence Model

This is the code of 
**Learning Cut Selection for Mixed-Integer Linear Programming via Hierarchical Sequence Model**.
Zhihai Wang, Xijun Li, Jie Wang, Yufei Kuang, Mingxuan Yuan, Jia Zeng, Yongdong Zhang, Feng Wu. ICLR 2023. [[link](https://openreview.net/forum?id=Zob4P9bRNcK)]

## Environmental requirements

- Hardware: indicates a GPU and CPU equipped machine

- Deep learning framework: Pytorch

- Python rely on

    - Python 3.7

    - tqdm

    - gtimer

- Solver dependencies

    - SCIP 8.0.0



## Quick Start

After the environment and dataset are ready, execute the following code to begin training and periodically test the training model performance

    python parallel_reinforce_algorithm.py --config_file configs/easy_setcover_config.json --sel_cuts_percent 0.2 --reward_type solving_time --instance_type easy_setcover --time_limit 300 --train_type train --use_cutsel_percent_policy True --policy_type no_token --scip_seed 1



## Script description

- configs/*.json # Indicates the parameter configuration file

- reinforce_algorithm.py # Start a serial training experimental inlet

- parallel_reinforce_algorithm.py # Start the parallel training experimental inlet

- pointer_net.py # network structure model

- environments.py # packages the solver as a reinforcement learning environment


## Random fact Sheet

There are two sources of code randomness. One is the randomness of the algorithm inside the solver, which can be fixed by setting the scip_seed parameter. The second is the random module in Python and the random module in Pytorch, which can be uniformly set by setting the seed parameter.

## Datasets

We have released our datasets in https://drive.google.com/drive/folders/1LXLZ8vq3L7v00XH-Tx3U6hiTJ79sCzxY?usp=sharing

## Citation
If you find this code useful, please consider citing the following paper.
```
@inproceedings{
wang2023learning,
title={Learning Cut Selection for Mixed-Integer Linear Programming via Hierarchical Sequence Model},
author={Zhihai Wang and Xijun Li and Jie Wang and Yufei Kuang and Mingxuan Yuan and Jia Zeng and Yongdong Zhang and Feng Wu},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=Zob4P9bRNcK}
}
```

## Remarks
We will release our data reported in our paper soon.
