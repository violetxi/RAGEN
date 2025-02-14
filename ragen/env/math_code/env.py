import gym
import random
import numpy as np
import datasets
from omegaconf import DictConfig

from .math_evaluator import MathEvaluator
from .code_execution_utils import PythonREPL


class MathCodeEnv:
    """ Takes <act> </act> & <ans> </ans>from LLM outputs and executes them 
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.python_repl = PythonREPL()
        self.math_evaluator = MathEvaluator()        

    def _load_dataset(self):
        ds = datasets.load_dataset(self.config.dataset_id, self.config.split)
        ds = ds.select(range(self.config.num_questions))
        self.problems = ds['problem']
        self.answers = ds['answer']

        self.prompt_template = "Question: {problem}\n\nSolution: "

    def reset(self, seed=None):
        """ Reset the environment to a new question """
        random.seed(seed)
        np.random.seed(seed)
        self.reward = 0


    # def compute

