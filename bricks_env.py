import random
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import (
    BreadObject,
    BreadVisualObject,
    CanObject,
    CanVisualObject,
    CerealObject,
    CerealVisualObject,
    MilkObject,
    MilkVisualObject,
)
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


class BricksEnv(SingleArmEnv):
    """
    This class corresponds to the pick place task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        # bin1_pos (3-tuple): Absolute cartesian coordinates of the bin initially holding the objects
        #
        # bin2_pos (3-tuple): Absolute cartesian coordinates of the goal bin

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        single_object_mode (int): specifies which version of the task to do. Note that
            the observations change accordingly.

            :`0`: corresponds to the full task with all types of objects.

            :`1`: corresponds to an easier task with only one type of object initialized
               on the table with every reset. The type is randomized on every reset.

            :`2`: corresponds to an easier task with only one type of object initialized
               on the table with every reset. The type is kept constant and will not
               change between resets.

        object_type (string): if provided, should be one of "milk", "bread", "cereal",
            or "can". Determines which type of object will be spawned on every
            environment reset. Only used if @single_object_mode is 2.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid object type specified]
        AssertionError: [Invalid number of robots specified]
    """

    # TODO way too many input args, need to refactor, create a config file, etc.
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        #table_full_size=(0.39, 0.49, 0.82),
        #table_friction=(1, 0.005, 0.0001),
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        # bin1_pos=(0.1, -0.25, 0.8),
        # bin2_pos=(0.1, 0.28, 0.8),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        single_object_mode=0,
        object_type=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):

        # task settings
        self.single_object_mode = single_object_mode
        self.object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
        self.object_id_to_sensors = {}  # Maps object id to sensor names for that object
        self.obj_names = ["Milk", "Bread", "Cereal", "Can"]
        if object_type is not None:
            assert object_type in self.object_to_id.keys(), "invalid @object_type argument - choose one of {}".format(
                list(self.object_to_id.keys())
            )
            self.object_id = self.object_to_id[object_type]  # use for convenient indexing
        self.obj_to_use = None

        # settings for table-top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.placement_initializer = None

        # # settings for bin position
        # self.bin1_pos = np.array(bin1_pos)
        # self.bin2_pos = np.array(bin2_pos)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        # We do not need any reward function for this task
        return 0

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        # self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        #
        # # can sample anywhere in bin
        # bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.05
        # bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05
        #
        # # each object should just be sampled in the bounds of the bin (with some tolerance)
        # self.placement_initializer.append_sampler(
        #     sampler=UniformRandomSampler(
        #         name="CollisionObjectSampler",
        #         mujoco_objects=self.objects,
        #         x_range=[-bin_x_half, bin_x_half],
        #         y_range=[-bin_y_half, bin_y_half],
        #         rotation=None,
        #         rotation_axis="z",
        #         ensure_object_boundary_in_range=True,
        #         ensure_valid_placement=True,
        #         reference_pos=self.bin1_pos,
        #         z_offset=0.0,
        #     )
        # )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        # TODO we overwrite self.model, but we should be able to just append to it
        # super()._load_model()
        #
        # # Adjust base pose accordingly
        # xpos = self.robots[0].robot_model.base_xpos_offset["table"]
        # self.robots[0].robot_model.set_base_xpos(xpos)
        #
        # # load model for table-top workspace
        # mujoco_arena = TableArena(
        #     table_full_size=self.table_full_size, table_friction=self.table_friction
        # )
        #
        # # Arena always gets set to zero origin
        # mujoco_arena.set_origin([0, 0, 0])
        #
        # # store some arena attributes
        #
        # self.objects = []
        # self.visual_objects = []
        #
        # # task includes arena, robot, and objects of interest
        # # TODO change naming: this is not a task, it's a model of the environment
        # self.model = ManipulationTask(
        #     mujoco_arena=mujoco_arena,
        #     mujoco_robots=[robot.robot_model for robot in self.robots],
        #     mujoco_objects=self.visual_objects + self.objects,
        # )
        #
        # # Generate placement initializer
        # self._get_placement_initializer()


        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table-top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        # redwood = CustomMaterial(
        #     texture="WoodRed",
        #     tex_name="redwood",
        #     mat_name="redwood_mat",
        #     tex_attrib=tex_attrib,
        #     mat_attrib=mat_attrib,
        # )
        # greenwood = CustomMaterial(
        #     texture="WoodGreen",
        #     tex_name="greenwood",
        #     mat_name="greenwood_mat",
        #     tex_attrib=tex_attrib,
        #     mat_attrib=mat_attrib,
        # )
        # self.cubeA = BoxObject(
        #     name="cubeA",
        #     size_min=[0.02, 0.02, 0.02],
        #     size_max=[0.02, 0.02, 0.02],
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        # )
        # self.cubeB = BoxObject(
        #     name="cubeB",
        #     size_min=[0.025, 0.025, 0.025],
        #     size_max=[0.025, 0.025, 0.025],
        #     rgba=[0, 1, 0, 1],
        #     material=greenwood,
        # )
        # cubes = [self.cubeA, self.cubeB]
        cubes = []
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

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

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

    def _check_success(self):
        return False
