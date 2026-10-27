"""Classify a failed run without forcing a single guess."""

from dataclasses import dataclass

from app.engineering.trajectory import POLICY, TOOL_OUTCOME, Trajectory, VERIFICATION

UNKNOWN = "unknown"
MULTIPLE_CAUSES = "multiple_causes"
TOOL_ERROR = "tool_error"
POLICY_DENIED = "policy"
TIMEOUT = "timeout"
BUDGET = "budget"
VERIFICATION_FAILED = "verification"
WORKFLOW = "workflow"
TEST_FAILURE = "test_failure"


@dataclass(frozen=True)
class FailureClass:
    label: str
    causes: tuple[str, ...]
    detail: str = ""


def classify_failure(
    trajectory: Trajectory | None,
    *,
    test_results: tuple[str, ...] = (),
    policy_events: tuple[str, ...] = (),
    workflow_state: str = "",
) -> FailureClass:
    """Read tool errors, workflow state, tests, and policy events.

    Returns ``unknown`` when nothing matches and ``multiple_causes`` when more
    than one distinct cause is present.
    """
    causes: list[str] = []
    details: list[str] = []
    events = trajectory.events if trajectory is not None else ()

    for event in events:
        text = f"{event.error} {event.action} {' '.join(str(value) for value in event.payload.values())}".casefold()
        if event.kind == POLICY or "denied" in text or "permitted" in text:
            _add(causes, details, POLICY_DENIED, event.error or "A policy event blocked the run.")
        if "budget" in text:
            _add(causes, details, BUDGET, event.error or "The search or cost budget was exhausted.")
        if "timed out" in text or "timeout" in text or "deadline" in text:
            _add(causes, details, TIMEOUT, event.error or "A deadline or timeout stopped the run.")
        if event.kind == TOOL_OUTCOME and event.ok is False:
            _add(causes, details, TOOL_ERROR, event.error or f"{event.action} failed.")
        if event.kind == VERIFICATION and event.ok is False:
            _add(causes, details, VERIFICATION_FAILED, event.error or "A claim was not supported.")

    for item in policy_events:
        _add(causes, details, POLICY_DENIED, item)
    for item in test_results:
        if item and item not in {"ok", "passed"}:
            _add(causes, details, TEST_FAILURE, item)
    if workflow_state in {"failed", "halted", "cancelled"} and not causes:
        _add(causes, details, WORKFLOW, f"The workflow stopped in {workflow_state}.")
    elif workflow_state == "failed":
        _add(causes, details, WORKFLOW, "The workflow reported a failure.")

    if not causes:
        return FailureClass(UNKNOWN, (), "No matching tool, policy, test, or workflow signal.")
    if len(causes) > 1:
        return FailureClass(MULTIPLE_CAUSES, tuple(causes), "; ".join(details))
    return FailureClass(causes[0], tuple(causes), details[0])


def _add(causes: list[str], details: list[str], label: str, detail: str) -> None:
    if label not in causes:
        causes.append(label)
        details.append(detail)
