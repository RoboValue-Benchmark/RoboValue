from __future__ import annotations

from typing import Sequence
from vmbmk.errors import VMBMKError
from vmbmk.inference.queries import Query, Result


def result_scores(
    queries: Sequence[Query], results: Sequence[Result], metric: str, op: str
) -> dict[str, float]:
    expected = {query.query_id for query in queries}
    scores: dict[str, float] = {}
    for result in results:
        if result.query_id in scores:
            raise VMBMKError(f"duplicate {metric} result: {result.query_id}")
        if result.op != op or result.score is None:
            raise VMBMKError(
                f"{metric} result {result.query_id!r} must use op={op}"
            )
        scores[result.query_id] = result.score
    missing = sorted(expected - set(scores))
    extra = sorted(set(scores) - expected)
    if missing or extra:
        raise VMBMKError(
            f"{metric} result coverage mismatch; missing={missing}, extra={extra}"
        )
    return scores
