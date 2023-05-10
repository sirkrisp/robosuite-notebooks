from collections import OrderedDict

from bricks_dataset.brick_envs import (
    BoxEnv,
    BridgeEnv,
    ChairEnv,
    FloorEnv,
    PickPlaceEnv,
    Pyramid2DEnv,
    Pyramid3DEnv,
    Tower2DEnv,
    Tower3DEnv,
    WallEnv,
)

ALL_BRICK_ENVIRONMENTS = OrderedDict((
    ('box', BoxEnv),
    ('bridge', BridgeEnv),
    ('chair', ChairEnv),
    ('floor', FloorEnv),
    ('pick_place', PickPlaceEnv),
    ('pyramid_2d', Pyramid2DEnv),
    ('pyramid_3d', Pyramid3DEnv),
    ('tower_2d', Tower2DEnv),
    ('tower_3d', Tower3DEnv),
    ('wall', WallEnv),
))

