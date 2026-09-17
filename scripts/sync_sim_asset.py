"""Copy the generated sim model from a `legged_env_v2` checkout into this
repo's bundled `assets/cartesian_hand/` (`cartesian_hand.xml` + `meshes/`), so
this repo does not need the sibling checkout at runtime. Overwrites the
bundled copy unconditionally -- cheap, since the model is
`cartesian_hand_creation.py` output on the other side, never hand-edited.

Run after `cartesian_hand_creation.py` regenerates the model in
`legged_env_v2`:

    python scripts/sync_sim_asset.py
    python scripts/sync_sim_asset.py --source ~/other/legged_env_v2
"""
import os
import shutil
from pathlib import Path

import tyro

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "assets" / "cartesian_hand"


def main(source: str | None = None) -> None:
    """Args:
        source: a `legged_env_v2` checkout. Default $LEGGED_ENV_ROOT, then a
            sibling `../legged_env_v2` beside this repo.
    """
    root = Path(source or os.environ.get(
        "LEGGED_ENV_ROOT", REPO_ROOT.parent / "legged_env_v2"))
    src = root / "asset" / "cartesian_hand"
    xml = src / "cartesian_hand.xml"
    if not xml.exists():
        raise SystemExit(f"no cartesian_hand.xml under {src} -- pass --source "
                         "or set $LEGGED_ENV_ROOT to a legged_env_v2 checkout")

    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(xml, DEST / "cartesian_hand.xml")
    shutil.rmtree(DEST / "meshes", ignore_errors=True)
    shutil.copytree(src / "meshes", DEST / "meshes")

    print(f"synced {DEST} from {src}")
    print("  " + xml.read_text().splitlines()[0].strip("<!- ").rstrip("- >"))


if __name__ == "__main__":
    tyro.cli(main, prog="python scripts/sync_sim_asset.py")
