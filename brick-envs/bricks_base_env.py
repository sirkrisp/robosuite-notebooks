from abc import abstractmethod, ABCMeta

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv


class BricksBaseEnv(SingleArmEnv, metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def go_to_step(self, step: int):
        pass

    @property
    @abstractmethod
    def num_steps(self) -> int:
        pass

    def _load_model(self):
        """
        Load robot and table into the world.

        Returns:

        """
        pass


