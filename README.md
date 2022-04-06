# A-Comparative-Study-of-Reinforcement-Learning-based-Transferable-EMS-for-HEVs

Source code for **A Comparative Study of Reinforcement Learning-based Transferable Energy Management Strategies for Hybrid Electric Vehicles** for IV 2022. More results will be presented in the next work.
If you use our implementation in your academic work, please cite the corresponding [paper](https://arxiv.org/abs/2202.11514):

```
@misc{xu2022comparative,
      title={A Comparative Study of Deep Reinforcement Learning-based Transferable Energy Management Strategies for Hybrid Electric Vehicles}, 
      author={Jingyi Xu and Zirui Li and Li Gao and Junyi Ma and Qi Liu and Yanan Zhao},
      year={2022},
      eprint={2202.11514},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


-------------------------------------
* [Abstract](#abstract)
* [Preparation](#preparation)
* [Tutorial](#tutorial)

## Abstract
The deep reinforcement learning-based energy management strategies (EMS) has become a promising solution for hybrid electric vehicles (HEVs). When driving cycles are changed, the network will be retrained, which is a time-consuming and laborious task. A more efficient way of choosing EMS is to combine deep reinforcement learning (DRL) with transfer learning, which can transfer knowledge of one domain to the other new domain, making the network of the new domain reach convergence values quickly. Different exploration methods of RL, including adding action space noise and parameter space noise, are compared against each other in the transfer learning process in this work. Results indicate that the network added parameter space noise is more stable and faster convergent than the others. In conclusion, the best exploration method for transferable EMS is to add noise in the parameter space, while the combination of action space noise and parameter space noise generally performs poorly.

## Preparation
Before starting to carry out some relevant works on our framework, some preparations are required to be done.

### Hardware
Our framework is developed based on a laptop, and the specific configuration is as follows:
- CPU: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz
- GPU: RTX 2070s

### Dependencies
Before using our code, the following dependencies are needed:
- tensorflow 1.13.2
- numpy 1.21.2
- scipy 1.7.1
- matplotlib 3.4.3

## Tutorial
Coming soon ...

