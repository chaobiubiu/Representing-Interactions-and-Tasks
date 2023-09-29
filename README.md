# Multi-Task Multi-Agent Reinforcement Learning with Interaction and Task Representations

Official code for the paper "Multi-Task Multi-Agent Reinforcement Learning with Interaction and Task Representations" submitted to TNNLS. 

This repository develops RIT algorithm in the StarCraft Multi-Agent Challenge (SMAC) benchmark, and
compares it with several baselines including UPDeT-MIX, UPDeT-VDN, REFIL, ROMA, QMIX-ATTN and VDN-ATTN.

Note that we use the task sets introduced by REFIL to conduct multi-task and zero-shot evaluations. There are usually several 
issues when installing this multi-task SMAC. Please refer to https://github.com/shariqiqbal2810/REFIL/issues/2 for some guidance.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the approach in the paper, run this command:

```train
python main.py
```

You can select tasks sets in the SMAC benchamrk with random deicison order or not by setting:

 ```
 --env-config='randomsc2custom' or 'sc2custom'
```

Also you can select the training algorithm by setting 

```
--config='scvd' or 'qmix_attn_sc2custom' or 'updet' or 'refil' or 'roma'
```

Here ```'scvd'``` refers to the approach ```'RIT'``` presented in our paper.

## Hyper-parameters

To modify the hyper-parameters of algorithms, refer to:

```
src/config/algs/xxx.yaml
src/config/default.yaml
```

If you want to use ```'RIT-VDN', 'VDN-ATTN', 'UPDeT-VDN'```, simply set ```mixer: "vdn"``` in their configuration file.

To modify the hyper-parameters of environments, refer to:

```
src/config/envs/xxx.yaml
```

## Note

This repository is developed based on PyMARL2 and the code of REFIL. Please refer to 
https://github.com/hijkzzz/pymarl2 and https://github.com/shariqiqbal2810/REFIL for more details.