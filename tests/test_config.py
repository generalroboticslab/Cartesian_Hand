"""Self-check for the hand description. `python tests/test_config.py`.

Pure data and arithmetic. Guards hand-edited tables and silent-fail conversions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from cartesian_hand.config import HAND_1, HAND_2, HandConfig, identify


# ── Tensors ────────────────────────────────────────────────────────────────────

def test_a_gain_scalar_broadcasts_and_a_vector_passes_through():
    scalar = HandConfig(name="s", port="/dev/null", torque_min_to_move=50,
                        torque_stuck=50)
    assert scalar.gain_vector("torque_min_to_move").tolist() == [50] * 7
    per_dof = HandConfig(name="v", port="/dev/null",
                         torque_min_to_move=[50, 150, 50, 50, 50, 50, 50],
                         torque_stuck=50)
    assert per_dof.gain_vector("torque_min_to_move").tolist() == [50, 150, 50, 50, 50, 50, 50]
    # int, not float: these are raw servo register values and the bus wants
    # `.tolist()` of exactly these numbers. A float rounds at the boundary.
    assert per_dof.gain_vector("torque_min_to_move").dtype == torch.int32, "gains must stay int"


def test_a_variant_does_not_inherit_the_old_hand_s_cache():
    """The cache is keyed by quantity and device, not by config, so a variant
    that kept it would answer with the travel of the hand it was copied from."""
    a = HAND_1
    a.upper()                                     # populate
    b = a.variant(travel_mm=[10.0] * 7)
    assert b.upper().tolist() == [10.0] * 7, f"stale cache: {b.upper().tolist()}"
    assert a.upper().tolist() != [10.0] * 7, "variant mutated the original"


def test_tensors_follow_the_requested_device():
    """The tick passes `tensor.device` (always indexed, "cuda:0"); call sites
    pass the string "cuda". Same device, different spelling -- keying on the raw
    argument caches two copies and never hits from the tick.
    """
    if not torch.cuda.is_available():
        return
    cfg = HAND_1.variant()                        # fresh cache
    assert cfg.lower("cuda").device.type == "cuda"
    assert cfg.lower("cpu").device.type == "cpu", "device cache collided"
    resolved = torch.zeros(1, device="cuda").device
    assert cfg.lower("cuda") is cfg.lower(resolved), "cuda and cuda:0 cached apart"


# ── Unit conversion ────────────────────────────────────────────────────────────

def test_mm_to_counts_goes_to_the_nearest_count_not_the_lower_one():
    """Rounding, not truncation. 0.6 not 0.5: torch.round breaks ties to even,
    so a half-count tests the tie rule instead of the truncation this is about.
    """
    cfg = HAND_1
    zero = torch.zeros(cfg.n_dof)
    near = torch.full((cfg.n_dof,), 0.6 / cfg.counts_per_mm)
    counts = cfg.mm_to_counts(near, zero).abs()
    assert (counts == 1).all(), f"truncated instead of rounding: {counts.tolist()}"


def test_orientation_is_applied_and_the_round_trip_cancels_it():
    """Orientation is wiring, not convention: above this layer, positive mm
    already means opening on both backends. Sign matters; round trip alone can
    close with the wrong sign.
    """
    cfg = HAND_1
    zero = torch.zeros(cfg.n_dof)
    for mm in (10.0, 50.0):
        counts = cfg.mm_to_counts(torch.full((cfg.n_dof,), mm), zero)
        signs = torch.sign(counts.to(torch.float32))
        assert torch.equal(signs, cfg.orientations()), "orientation not applied"
        assert torch.allclose(cfg.counts_to_mm(counts, zero),
                              torch.full((cfg.n_dof,), mm), atol=0.02)
        # Pinned against the bench, 2026-09-01: flipping these to all +1
        # reversed five DOFs on the real hand. The assert above compares the
        # table with itself and passes for any table, so it cannot catch that.
        assert signs.tolist() == [-1, 1, -1, -1, -1, 1, -1], \
            f"LAYOUT orientation changed: {signs.tolist()}"


# ── The normalized action space ────────────────────────────────────────────────

def test_normalize_inverts_denormalize_and_saturating_actions_clip():
    """A policy pressing a joint limit must press against it, not through it.

    Direction only. That the box ends land on travel ends lives in
    `test_sim_real_contract.py::test_config_uses_the_whole_action_range`.
    """
    cfg = HAND_1
    assert torch.allclose(cfg.normalize(cfg.lower()), torch.full((7,), -1.0))
    assert torch.allclose(cfg.normalize(cfg.upper()), torch.full((7,), +1.0))
    assert torch.allclose(cfg.denormalize(torch.full((7,), 9.0)), cfg.upper())
    assert torch.allclose(cfg.denormalize(torch.full((7,), -9.0)), cfg.lower())


def test_clamp_bounds_a_whole_pose():
    cfg = HAND_1
    out = cfg.clamp(torch.full((4, 7), 999.0))
    assert torch.allclose(out, cfg.upper().expand(4, 7))


# ── Which hand is on the bus ───────────────────────────────────────────────────

class FakeBus:
    """Answers for `present` and drops the reply for everything else."""

    def __init__(self, present):
        self.present = set(present)

    def read_all(self, sids):
        return [(0, 0, 0) if s in self.present else None for s in sids]

    def scan(self, lo, hi):
        return sorted(s for s in self.present if lo <= s <= hi)


def test_the_answering_id_block_names_the_hand():
    """And that anything other than exactly one block is refused. Both failures
    move the wrong hardware if guessed: a partial block is a hand with a dead
    servo, run as if the servo were there, and two blocks is two hands on one
    bus. The port comes back as the caller's, since an adapter that enumerated
    elsewhere is still that hand."""
    assert identify(FakeBus(HAND_1.servo_ids)).name == "hand_1"
    assert identify(FakeBus(HAND_2.servo_ids), "/dev/ttyACM3").port == "/dev/ttyACM3"
    for present, expect in ((HAND_2.servo_ids[:-1], "no known hand"),
                            (HAND_1.servo_ids + HAND_2.servo_ids, "--hand")):
        try:
            identify(FakeBus(present))
            raise AssertionError(f"guessed a hand from {present}")
        except RuntimeError as e:
            assert expect in str(e), e


# ── The policy contract ────────────────────────────────────────────────────────

def test_fingerprint_matches_kinematics_not_calibration():
    """Three rules in one: ports/IDs are not in the contract (a policy works in
    normalized units); travel and control rate *are* (they change what an action
    means); calibration is not (two units with the same travel need different
    counts_per_mm and folding it in would make calibrated hands look
    incompatible).
    """
    assert HAND_1.fingerprint() == HAND_2.fingerprint(), \
        "ports or servo_ids leaked into the contract"
    assert HAND_1.port != HAND_2.port and HAND_1.servo_ids != HAND_2.servo_ids

    narrow = HAND_1.variant(travel_mm=[25.0] * 7)
    assert narrow.fingerprint() != HAND_1.fingerprint(), "travel ignored"
    slow = HAND_1.variant(control_hz=10)
    assert slow.fingerprint() != HAND_1.fingerprint(), "control rate ignored"

    recal = HAND_1.variant(counts_per_mm=80.0)
    assert recal.fingerprint() == HAND_1.fingerprint(), \
        "calibration folded in; correctly calibrated hands now look incompatible"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except BaseException as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
