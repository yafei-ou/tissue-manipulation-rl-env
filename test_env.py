from envs import SkinTissueEnvContinuousRandom, ObservationSqueezeWrapper
from components.utils import Parameters


params = Parameters("./configs/skin_tissue.yml")

env = SkinTissueEnvContinuousRandom(obs_sequence_length=1, params=params, render_mode="pyplot", randomize=True)
env = ObservationSqueezeWrapper(env=env)
env.seed(3)
env.reset()

for _ in range(20):
    act = env.action_space.sample()
    env.step(act)

env.close()
