import colorsys

hsl_colors = {
    'flat_dark_red': (6, 63, 46),
    # 'light_red': (6, 78, 57),
    # 'light_purple': (283, 39, 53),
    'flat_dark_purple': (282, 44, 47),
    'flat_dark_blue': (204, 64, 44),
    # 'light_blue': (204, 70, 53),
    'flat_light_turquoise': (168, 76, 42),
    # 'dark_turquoise': (168, 76, 36),
    'flat_dark_green': (145, 63, 42),
    # 'light_green': (145, 63, 49),
    'flat_yellow': (48, 89, 50),
    # 'light_orange': (37, 90, 51),
    'flat_dark_orange': (28, 80, 52),
    'flat_brown': (24, 100, 41),

    'custom_blue_grey': (204, 8, 76),

    'flat_blue_brown': (210, 15, 43),
    'material_brown': (16, 25, 38),
    'material_pink': (340, 82, 52),
    'material_purple': (291, 64, 42),
    'web_safe_green': (135, 100, 40),
    'web_safe_muddy_brown_green': (60, 33, 30),
    'web_safe_muddy_green': (90, 50, 40),
    'web_safe_muddy_orange': (20, 60, 50),


}


def hsl_to_rgba(hsl: tuple[int, int, int]):
    return list(colorsys.hls_to_rgb(hsl[0] / 360, hsl[2] / 100, hsl[1] / 100)) + [1]


def hsl_change_brightness(hsl, factor):
    return hsl[0], hsl[1], hsl[2] * factor
