import numpy as np
from bricks_dataset.brick_envs.brick_assembly_instruction import BrickAssemblyInstruction


def generate_pin_sizes(num_blocks: int, total_num_pins: int):
    pin_separators = np.random.choice(np.arange(total_num_pins - 1), size=num_blocks, replace=False)
    pin_separators.sort()
    sizes = np.diff(np.concatenate(([0], pin_separators + 1, [total_num_pins])))
    assert np.sum(sizes) == total_num_pins
    return sizes


def generate_brick_shapes(num_blocks: int, total_num_pins: tuple[int, int, int], start_id: int = 0):
    sizes_x = generate_pin_sizes(num_blocks, total_num_pins[0])
    sizes_y = generate_pin_sizes(num_blocks, total_num_pins[1])
    sizes_z = generate_pin_sizes(num_blocks, total_num_pins[2])
    return {i + start_id: (sizes_x[i], sizes_y[i], sizes_z[i]) for i in range(num_blocks)}


def check_assembly(instruction: BrickAssemblyInstruction, mask3d: np.ndarray,
                   brick_locations: dict[int, tuple[int, int, int]], brick_shapes: dict[int, tuple[int, int, int]]):
    brick_1_location = brick_locations[instruction.brick_1_idx]
    assert instruction.o == 0
    brick_1_shape = brick_shapes[instruction.brick_1_idx]
    brick_2_shape = brick_shapes[instruction.brick_2_idx]
    pin_1_x, pin_1_y = instruction.pin_1
    pin_2_x, pin_2_y = instruction.pin_2
    for x in range(brick_2_shape[0]):
        for y in range(brick_2_shape[1]):
            for z in range(brick_2_shape[2]):
                if (mask3d[
                    brick_1_location[0] + x + pin_1_x - pin_2_x,
                    brick_1_location[1] + y + pin_1_y - pin_2_y,
                    brick_1_location[2] + brick_1_shape[2] + z
                ] != 0):
                    return False
    return True


def generate_instruction(used_bricks: list[int], unused_bricks: list[int],
                         brick_shapes: dict[int, tuple[int, int, int]]):
    # select random block from added blocks
    brick_1_id = np.random.choice(used_bricks)
    brick_2_id = np.random.choice(unused_bricks)
    brick_1_shape = brick_shapes[brick_1_id]
    brick_2_shape = brick_shapes[brick_2_id]

    # select random pin from brick 1
    pin_1_x = np.random.randint(brick_1_shape[0])
    pin_1_y = np.random.randint(brick_1_shape[1])

    # select random pin from brick 2
    pin_2_x = np.random.randint(brick_2_shape[0])
    pin_2_y = np.random.randint(brick_2_shape[1])

    # TODO select random orientation
    o = 0  # np.random.randint(4)

    # create instruction
    instruction = BrickAssemblyInstruction(brick_1_id, (pin_1_x, pin_1_y), brick_2_id, (pin_2_x, pin_2_y), o)

    return instruction


def generate_instructions(brick_shapes: dict[int, tuple[int, int, int]]):
    used_bricks = [0]
    unused_bricks = list(brick_shapes.keys())
    unused_bricks.remove(0)

    mask3d = np.zeros((100, 100, 100), dtype=np.uint8)
    brick_locations = {0: (50, 50, 0)}

    instructions = []

    while len(unused_bricks) > 0:

        # generate valid instruction
        instruction = generate_instruction(used_bricks, unused_bricks, brick_shapes)
        max_tries = 100
        while not check_assembly(instruction, mask3d, brick_locations, brick_shapes) and max_tries > 0:
            instruction = generate_instruction(used_bricks, unused_bricks, brick_shapes)
            max_tries -= 1
        if max_tries == 0:
            raise Exception("Could not generate valid instruction")

        # add instruction
        instructions.append(instruction)
        used_bricks.append(instruction.brick_2_idx)
        unused_bricks.remove(instruction.brick_2_idx)
        brick_locations[instruction.brick_2_idx] = (
            brick_locations[instruction.brick_1_idx][0] + instruction.pin_1[0] - instruction.pin_2[0],
            brick_locations[instruction.brick_1_idx][1] + instruction.pin_1[1] - instruction.pin_2[1],
            brick_locations[instruction.brick_1_idx][2] + brick_shapes[instruction.brick_1_idx][2]
        )

        # update mask
        brick_2_shape = brick_shapes[instruction.brick_2_idx]
        brick_2_location = brick_locations[instruction.brick_2_idx]
        for x in range(brick_2_shape[0]):
            for y in range(brick_2_shape[1]):
                for z in range(brick_2_shape[2]):
                    mask3d[brick_2_location[0] + x, brick_2_location[1] + y, brick_2_location[2] + z] = 1

    return instructions
