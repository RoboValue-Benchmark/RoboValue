"""Pure, dependency-free VS v1 calculation; inputs are never modified."""
from dataclasses import asdict, dataclass
import math
from numbers import Real
from statistics import fmean
from typing import Iterable

VERSION = "vs_v1"


@dataclass(frozen=True)
class Score:
    er: float
    scale: float
    vs: float
    flat_time_fraction: float
    total_variation: float
    signed_net_change: float
    absolute_net_change: float
    start_mean: float
    end_mean: float
    constant_raw: bool
    constant_after_boundary: bool
    n_points: int
    n_flat_intervals: int
    duration: float
    flat_duration: float
    time_basis: str

    def to_dict(self):
        return asdict(self)


def finite_numbers(values: Iterable[float], name: str) -> list[float]:
    result = []
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ValueError(f'{name} must be an iterable of real numbers') from exc
    for value in iterator:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must contain real numbers, not {value!r}")
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} exceeds float64 range") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} contains a nonfinite number")
        result.append(number)
    return result


def score_curve(values: Iterable[float], times: Iterable[float] | None = None) -> Score:
    """Score one forward expert curve; see docs/method.md for protocol limits.

    At least seven finite values are required. Times must strictly increase;
    omitted times mean equal sampling intervals (duration is in sample units).
    Exact equality uses the input values converted to Python float (float64).
    """
    y = finite_numbers(values, "values")
    if len(y) < 7:
        raise ValueError("VS v1 requires at least 7 points")
    t = list(map(float, range(len(y)))) if times is None else finite_numbers(times, "times")
    if len(t) != len(y):
        raise ValueError("values and times must have the same length")
    dt = [b-a for a, b in zip(t, t[1:])]
    if any(not math.isfinite(d) or d <= 0 for d in dt):
        raise ValueError("times must have finite, strictly positive increments")
    try:
        # Net change and total variation must use the same boundary-processed copy.
        first, last = fmean(y[:3]), fmean(y[-3:])
        z = y.copy()
        z[:3], z[-3:] = [first]*3, [last]*3
        up = math.fsum(max(b-a, 0.0) for a, b in zip(z, z[1:]))
        down = math.fsum(max(a-b, 0.0) for a, b in zip(z, z[1:]))
        total, net = up+down, last-first
        duration = math.fsum(dt)
        equal = [a == b for a, b in zip(y, y[1:])]
        flat = math.fsum(d for d, same in zip(dt, equal) if same)
        active = math.fsum(d for d, same in zip(dt, equal) if not same)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Nonfinite intermediate value") from exc
    if not all(math.isfinite(v) for v in (first, last, total, net, duration, flat, active)):
        raise ValueError("Nonfinite intermediate value")
    er = abs(net)/total if total else 1.0
    if er > 1 and math.isclose(er, 1, rel_tol=0, abs_tol=1e-12):
        er = 1.0
    if not 0 <= er <= 1:
        raise ValueError("ER outside [0, 1]")
    scale = active/duration
    return Score(er=er, scale=scale, vs=er*scale, flat_time_fraction=flat/duration,
                 total_variation=total, signed_net_change=net, absolute_net_change=abs(net),
                 start_mean=first, end_mean=last, constant_raw=len(set(y)) == 1,
                 constant_after_boundary=total == 0, n_points=len(y),
                 n_flat_intervals=sum(equal), duration=duration, flat_duration=flat,
                 time_basis="sample_intervals" if times is None else "provided_timestamps")
