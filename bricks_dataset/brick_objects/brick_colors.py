import colorsys

hsl_colors = {
    'dark_red': (6, 63, 46),
    'light_red': (6, 78, 57),
    'light_purple': (283, 39, 53),
    'dark_purple': (282, 44, 47),
    'dark_blue': (204, 64, 44),
    'light_blue': (204, 70, 53),
    'light_turquoise': (168, 76, 42),
    'dark_turquoise': (168, 76, 36),
    'dark_green': (145, 63, 42),
    'light_green': (145, 63, 49),
    'yellow': (48, 89, 50),
    'light_orange': (37, 90, 51),
    'dark_orange': (28, 80, 52),
    'brown': (24, 100, 41)
}


def hsl_to_rgba(hsl: tuple[int, int, int]):
    return list(colorsys.hls_to_rgb(hsl[0] / 360, hsl[2] / 100, hsl[1] / 100)) + [1]


def hsl_change_brightness(hsl, factor):
    return hsl[0], hsl[1], hsl[2] * factor
