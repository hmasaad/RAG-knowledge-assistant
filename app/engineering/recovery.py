"""Detect unproductive retries and close a run when recovery is exhausted."""

from dataclasses import dataclass

from app.engineering.trajectory import TOOL_OUTCOME, Trajectory

RETRY_BUDGET = 2
RECOVERED = "recovered"
RETRY_EXHAUSTED = "retry_exhausted"
RECOVERY_LOOP = "recovery_loop"
UNPRODUCTIVE = "unproductive_retries"
REPEATED_FAILURE = "repeated_failure"
NO_RECOVERY = "none"


@dataclass(frozen=True)
class RecoveryAnalysis:
    status: str
    repeated_failures: tuple[str, ...]
    unproductive_retries: int
    loop_detected: bool
    retry_budget: int
    terminal: bool
    detail: str = ""


def analyze_recovery(
    trajectory: Trajectory | None,
    *,
    retry_budget: int = RETRY_BUDGET,
) -> RecoveryAnalysis:
    """Find repeated failures, empty retries, and A→B→A loops."""
    limit = max(1, retry_budget)
    if trajectory is None:
        return RecoveryAnalysis(NO_RECOVERY, (), 0, False, limit, False, "No trajectory.")

    tools = [event for event in trajectory.events if event.kind == TOOL_OUTCOME]
    failed_keys: dict[str, int] = {}
    unproductive = 0
    seen_ok: set[str] = set()
    sequence: list[str] = []
    for event in tools:
        key = f"{event.action}:{str(event.payload.get('query', '')).casefold()}"
        sequence.append(event.action)
        if event.ok:
            seen_ok.add(key)
            continue
        failed_keys[key] = failed_keys.get(key, 0) + 1
        if failed_keys[key] > 1:
            unproductive += 1

    repeated = tuple(key for key, count in failed_keys.items() if count >= 2)
    loop_detected = _has_loop(sequence)
    recovered = bool(repeated) and all(
        any(event.ok and f"{event.action}:{str(event.payload.get('query', '')).casefold()}" == key for event in tools)
        for key in repeated
    )

    if loop_detected:
        return RecoveryAnalysis(
            RECOVERY_LOOP,
            repeated,
            unproductive,
            True,
            limit,
            True,
            "The same actions repeated without a new outcome.",
        )
    if any(count > limit for count in failed_keys.values()):
        return RecoveryAnalysis(
            RETRY_EXHAUSTED,
            repeated,
            unproductive,
            False,
            limit,
            True,
            f"A failed action exceeded the retry budget of {limit}.",
        )
    if unproductive and not recovered:
        return RecoveryAnalysis(
            UNPRODUCTIVE,
            repeated,
            unproductive,
            False,
            limit,
            False,
            "Retries repeated a failed action without new evidence.",
        )
    if repeated and not recovered:
        return RecoveryAnalysis(
            REPEATED_FAILURE,
            repeated,
            unproductive,
            False,
            limit,
            False,
            "The same action failed more than once.",
        )
    if recovered or (tools and tools[-1].ok and any(not event.ok for event in tools)):
        return RecoveryAnalysis(RECOVERED, repeated, unproductive, False, limit, False, "A later attempt succeeded.")
    return RecoveryAnalysis(NO_RECOVERY, repeated, unproductive, False, limit, False, "No recovery pattern.")


def _has_loop(sequence: list[str]) -> bool:
    if len(sequence) < 4:
        return False
    for width in (2, 3):
        if len(sequence) < width * 2:
            continue
        tail = tuple(sequence[-width:])
        previous = tuple(sequence[-width * 2 : -width])
        if tail == previous:
            return True
    return False
