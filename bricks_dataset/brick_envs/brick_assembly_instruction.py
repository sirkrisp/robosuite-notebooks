from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BrickAssemblyInstruction:
    brick_1_idx: int
    pin_1: int | tuple[int, int]
    brick_2_idx: int
    pin_2: int | tuple[int, int]
    o: int

    def to_dict(self):
        return {
            "brick_1_idx": self.brick_1_idx,
            "pin_1": self.pin_1,
            "brick_2_idx": self.brick_2_idx,
            "pin_2": self.pin_2,
            "o": self.o
        }
