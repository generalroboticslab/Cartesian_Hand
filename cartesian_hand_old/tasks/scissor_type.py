"""Not yet implemented. See DESCRIPTION for intent."""

from dataclasses import dataclass

DESCRIPTION = "Shear/scissor manipulation (stub)"


@dataclass
class Config:
    pass


def run(hand, cfg: Config):
    raise NotImplementedError(
        "scissor_type is a stub. Implement run(hand, cfg) here, or drive the hand "
        "from a policy: python -m cartesian_hand policy --help")
