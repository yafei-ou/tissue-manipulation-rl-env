# ⚠️ Deprecation Notice

**This repository is deprecated and is no longer actively maintained.**

The repository will be publicly archived, meaning it will remain accessible in a read-only state. Please consider using [SofaGym](https://github.com/SofaDefrost/SofaGym) and [LapGym](https://github.com/ScheiklP/lap_gym) as they have much better implementations and are actively maintained.

---

# Soft Tissue Manipulation

This repo contains the main code for the paper [Sim-to-Real Surgical Robot Learning and Autonomous Planning for Internal Tissue Points Manipulation Using Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/10065553).

## Requirements

* [SOFA Framework==v21.12.00](https://github.com/sofa-framework/sofa/releases/tag/v21.12.00) with SofaPython3 plugin
* stable-baselines3==1.6.1
* gym==0.21.0

## Usage

Please refer to `test_env.py`.

```python
env = SkinTissueEnvContinuousRandom(obs_sequence_length=1, params=params, render_mode="pyplot", randomize=True)
env.reset()

for _ in range(20):
    act = env.action_space.sample()
    env.step(act)
```

## Citation

```latex
@article{ou2023sim,
  title={Sim-to-Real Surgical Robot Learning and Autonomous Planning for Internal Tissue Points Manipulation Using Reinforcement Learning},
  author={Ou, Yafei and Tavakoli, Mahdi},
  journal={IEEE Robotics and Automation Letters},
  volume={8},
  number={5},
  pages={2502--2509},
  year={2023},
  publisher={IEEE}
}
```
