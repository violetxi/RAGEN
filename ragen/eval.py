import hydra
from omegaconf import DictConfig
from ragen.env.math_code.env import MathCodeEnv


@hydra.main(version_base=None, config_path="config", config_name="eval_math500"):
def main(config: DictConfig):
    env = MathCodeEnv(config.env_config)
    