from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import yaml
import os
import numpy as np
from pyquaternion import Quaternion

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

from bricks_dataset.brick_envs.brick_assembly_instruction import BrickAssemblyInstruction
from bricks_dataset.brick_objects.brick_obj import BrickObj
from bricks_dataset.brick_objects.brick_colors import hsl_colors

config_path_default = os.path.join(os.path.dirname(__file__), "configs/bricks_base_env_config.yaml")


@dataclass
class TableConfig:
    full_size: tuple[float, float, float]
    friction: tuple[float, float, float]
    offset: tuple[float, float, float]


class BricksBaseEnv(SingleArmEnv):

    def __init__(self, config_path=config_path_default, override_defaults=None):

        with open(config_path, mode="rt", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        if override_defaults is not None:
            self.config.update(override_defaults)

        self.table_config = TableConfig(**self.config["bricks_base_env_config"]["table"])

        # we need to load bricks before we call super().__init__(...) because parent initializer calls _load_model()
        self.bricks: list[BrickObj] = self._create_bricks()
        self.num_bricks = len(self.bricks)
        assert self.num_bricks <= 14
        # brick placements is initialized when _reset_internal() is called
        self.brick_placements: dict[str, np.ndarray] | None = None

        # Create placement initializer
        table_full_size = self.table_config.full_size
        self.placement_initializer = UniformRandomSampler(
            name="BricksSampler",
            mujoco_objects=self.bricks,
            x_range=[0, table_full_size[0] / 2],
            y_range=[-table_full_size[1] / 2, table_full_size[1] / 2],
            rotation=None,
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_config.offset,
            z_offset=0.01,
        )

        super().__init__(**self.config["single_arm_env_config"])

        self.reset()

        # create instruction after bricks have been placed onto the table
        self.instructions: list[BrickAssemblyInstruction] = self._create_instructions()
        self.num_instructions = len(self.instructions)

    @abstractmethod
    def _create_bricks(self) -> list[BrickObj]:
        pass

    @abstractmethod
    def _create_instructions(self) -> list[BrickAssemblyInstruction]:
        pass

    def set_brick_colors(self, hsl_color_keys: list[str]):
        hsl_selected_colors = [hsl_colors[key] for key in hsl_color_keys]
        for i in range(self.num_bricks):
            self.bricks[i].set_hsl(hsl_selected_colors[i])

    def get_random_brick_colors(self):
        hsl_color_keys = np.array(list(hsl_colors.keys()))
        return hsl_color_keys[np.random.choice(range(hsl_color_keys.shape[0]), self.num_bricks, replace=False)].tolist()

    def _assemble_bricks(
            self,
            instruction: BrickAssemblyInstruction
    ):
        """ Assemble two bricks by placing brick_2 at pin_2 with orientation o on top of brick_1 at pin_1
        """
        brick_1 = self.bricks[instruction.brick_1_idx]
        brick_2 = self.bricks[instruction.brick_2_idx]

        # fetch data for brick 1 (we only need current pos and orientation of brick 1)
        qpos_1 = self.sim.data.get_joint_qpos(brick_1.joints[0])

        # compute new quaternion for brick 2
        # in mujoco quaternions have inverted order
        q_1 = Quaternion(qpos_1[-4:][::-1])
        q_o = Quaternion(axis=[0, 0, 1], angle=instruction.o / 4 * 2 * np.pi)
        q_2_new = q_o * q_1

        # compute new position for brick 2
        r_1_local = brick_1.get_pin_pos_local("top", instruction.pin_1)

        # TODO do we have to multiply with -1 because of inverse later?
        r_1_local[0] *= -1
        r_1_local[1] *= -1
        r_2_local = brick_2.get_pin_pos_local("bottom", instruction.pin_2)
        r_2_local[0] *= -1
        r_2_local[1] *= -1

        # TODO why inverse?
        r_1_world = q_1.inverse.rotate(r_1_local)
        r_2_world = q_2_new.inverse.rotate(r_2_local)

        p_1_world = qpos_1[:3] + r_1_world
        pos_2_new = p_1_world - r_2_world

        self.sim.data.set_joint_qpos(brick_2.joints[0], np.concatenate([pos_2_new, q_2_new.elements[::-1]]))
        self.sim.forward()

    def go_to_step(self, step: int):
        assert 0 <= step < self.num_instructions
        if self.brick_placements is None:
            self._reset_internal()
        for k in self.brick_placements.keys():
            self.sim.data.set_joint_qpos(k, self.brick_placements[k])
        for i in range(step + 1):
            self._assemble_bricks(self.instructions[i])

    def get_keyframe_data(
            self,
            img_resolution: tuple[int, int] = (300, 300),
            camera_name: str = "frontview",
            use_depth: bool = False,
    ):
        images = []
        depths = []
        trajectory_points = []

        # get image and points of scene before any instructions are executed
        # TODO use sensor, check that sensor is correct

        if use_depth:
            img, depth = self.sim.render(*img_resolution, camera_name=camera_name, depth=True)
            depths.append(depth)
        else:
            img = self.sim.render(*img_resolution, camera_name=camera_name)
        images.append(img)

        for i, instruction in enumerate(self.instructions):
            moving_brick = self.bricks[instruction.brick_2_idx]

            # get first pos of moving_brick before assembly
            trajectory_point_1 = self.sim.data.get_joint_qpos(moving_brick.joints[0])
            trajectory_point_1[2] += moving_brick.size_z_half

            self.go_to_step(i)

            # get second pos of moving_brick after assembly
            trajectory_point_2 = self.sim.data.get_joint_qpos(moving_brick.joints[0])
            trajectory_point_2[2] += moving_brick.size_z_half
            trajectory_points.append([trajectory_point_1, trajectory_point_2])

            # get image and depths of scene after assembly
            if use_depth:
                img, depth = self.sim.render(*img_resolution, camera_name=camera_name, depth=True)
                depths.append(depth)
            else:
                img = self.sim.render(*img_resolution, camera_name=camera_name)
            images.append(img)

        return {
            "images": images,
            "depths": depths,
            "trajectory_points": trajectory_points,
        }

    def _load_model(self):
        """
        Load robot and table into the world.

        Returns:

        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_config.full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table-top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_config.full_size,
            table_friction=self.table_config.friction,
            table_offset=self.table_config.offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.bricks,
        )

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # self._set_brick_colors()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            self.brick_placements = {}
            for obj_pos, obj_quat, obj in object_placements.values():
                qpos = np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                self.brick_placements[obj.joints[0]] = qpos
                self.sim.data.set_joint_qpos(obj.joints[0], qpos)

    def reward(self, action=None):
        # We do not need any reward function for this environment
        return 0

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        # object-specific ids
        # for obj in self.visual_objects + self.objects:
        #     self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
        #     self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

    def _check_success(self):
        return False
