from contextvars import ContextVar
from dataclasses import dataclass

from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import collect_measurements, h, measure_array, qubit, s


BASES = ("x", "y", "z")


@dataclass(frozen=True)
class RunConfig:
    basis: str
    shot_offset: int
    shots: int


current_run: ContextVar[RunConfig | None] = ContextVar("current_run", default=None)


def with_measurement(prepare, basis):
    """Measure the 16 qubits returned by prepare and emit them as bits."""
    if basis not in BASES:
        raise ValueError("Measurement basis must be x, y, or z")

    @guppy.comptime
    def rotate(qs: array[qubit, 16]) -> None:
        for i in range(16):
            if basis == "y":
                # S dagger is S cubed.
                s(qs[i])
                s(qs[i])
                s(qs[i])
            if basis in ("x", "y"):
                h(qs[i])

    @guppy
    def entry() -> None:
        qs: array[qubit, 16] = prepare()
        rotate(qs)
        output("bits", collect_measurements(measure_array(qs)))

    return entry
