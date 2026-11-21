"""Prepare tilted-product states from saved parameters and sample tables.

Run preprocessing.py to calibrate and sample. The participant commands measure
the qubits returned by this file's main in the selected basis.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime, output
from guppylang.std.qsystem.utils import get_current_shot
from guppylang.std.quantum import qubit, ry

from preprocessing import BASES, DATA_DIR, SHOTS
from pushquantum.program import current_run, with_measurement


@guppy
def tilt_qubit(q: qubit, shot: int, low: int, high: int,
               zero_angle: angle, one_angle: angle) -> int:
    bit = 0
    if shot < 64:
        bit = (low >> shot) & 1
    else:
        bit = (high >> (shot - 64)) & 1
    if bit == 1:
        ry(q, one_angle)
    else:
        ry(q, zero_angle)
    return bit


def build_preparation(temperature=2.375, basis="Z", data_dir=DATA_DIR, shot_offset=1):
    """Return a Guppy preparation function and metadata for 128 shots.

    Each shot uses an independently sampled label; 128 shots are the total for
    this temperature/basis, not 128 per configuration. Qubits are never entangled.
    H2 shot indices start at 1; pass shot_offset=0 for local Selene execution.
    """
    if basis not in BASES:
        raise ValueError("basis must be X, Y, or Z")
    if shot_offset not in (0, 1):
        raise ValueError("shot_offset must be 1 for H2 or 0 for Selene")
    # Preprocessing is explicit: compiling never calibrates or draws new samples.
    data_dir = Path(data_dir)
    parameter_path = data_dir / f"parameters_{temperature:g}.json"
    sample_path = data_dir / f"samples_{temperature:g}.npz"
    if not parameter_path.exists() or not sample_path.exists():
        raise FileNotFoundError(f"Run preprocessing.py --temperature {temperature:g} first")
    saved = json.loads(parameter_path.read_text())
    parameters = saved["parameters"]
    with np.load(sample_path, allow_pickle=False) as samples:
        labels = samples[f"{basis}_labels"].tolist()
        masks = samples[f"{basis}_masks"].tolist()
    if saved["temperature"] != temperature or saved["shots"] != SHOTS or len(labels) != SHOTS:
        raise ValueError("Preprocessed data must match the temperature and 128-shot budget")
    theta = parameters["theta_radians"]

    @guppy.comptime
    def apply_tilts(qs: array[qubit, 16]) -> int:
        # Expand saved masks at compile time; select bits per shot on the device.
        shot = get_current_shot() - shot_offset
        label = 0
        for q in range(16):
            low, high = masks[q]
            bit = tilt_qubit(qs[q], shot, low, high, angle(theta / math.pi),
                             angle((math.pi - theta) / math.pi))
            label = label | (bit << q)
        return label

    @guppy
    def main() -> array[qubit, 16]:
        qs = array(qubit() for _ in range(16))
        label = apply_tilts(qs)
        shot = get_current_shot() - comptime(shot_offset)
        output("shot", shot)
        output("label", label)
        return qs

    metadata = {"temperature": float(temperature), "basis": basis, "seed": saved["seeds"][basis],
                "shots": SHOTS, "shot_index_offset": shot_offset,
                "parameters": parameters,
                "sampled_labels": labels}
    return main, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=2.375)
    parser.add_argument("--basis", choices=("X", "Y", "Z"), default="Z")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path("tilted_product.hugr"))
    args = parser.parse_args()
    prepare, metadata = build_preparation(args.temperature, args.basis, args.data_dir)
    program = with_measurement(prepare, args.basis.lower())
    args.output.write_bytes(program.compile().to_bytes())
    print(f"Compiled {args.output}; parameters={metadata['parameters']}")
elif (run := current_run.get()) is not None:
    if run.shots != SHOTS:
        raise SystemExit(f"The tilted-product example requires exactly {SHOTS} shots.")
    main, _ = build_preparation(basis=run.basis.upper(), shot_offset=run.shot_offset)
