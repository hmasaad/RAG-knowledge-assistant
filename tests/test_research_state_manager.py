"""The durable state manager keeps progress across crashes, timeouts, and restarts."""

from pathlib import Path

from app.research.contracts import (
    REVERIFY,
    SEARCH,
    SYNTHESIZE,
    ActionOutcome,
    ActionRequest,
    EvidenceRef,
    ResearchState,
)
from app.research.cycle import run_cycle
from app.research.dispatcher import ActionDispatcher
from app.research.mocks import MockAdapter, mock_action_adapters
from app.research.sessions import ResearchSessionStore
from app.research.state_manager import DurableResearchStateManager


class _CountingSearch:
    action = SEARCH

    def __init__(self) -> None:
        self.calls: list[ActionRequest] = []

    def run(self, request: ActionRequest, state: ResearchState | None = None) -> ActionOutcome:
        self.calls.append(request)
        return ActionOutcome(
            action=SEARCH,
            ok=True,
            evidence=(EvidenceRef("e1", "d1", "notes.md", "The retry limit is three attempts."),),
            provenance=("e1",),
            cost=1,
        )


def _manager(tmp_path: Path, *, lease_seconds: float = 900) -> DurableResearchStateManager:
    return DurableResearchStateManager(
        ResearchSessionStore(tmp_path / "research.db"),
        lease_seconds=lease_seconds,
    )


def test_checkpoint_survives_a_new_manager_instance(tmp_path: Path) -> None:
    first = _manager(tmp_path)
    session = first.create(user_id="ada", question="what is the retry limit", top_k=3)
    request = ActionRequest(SEARCH, query="what is the retry limit")
    outcome = ActionOutcome(
        action=SEARCH,
        ok=True,
        evidence=(EvidenceRef("e1", "d1", "notes.md", "The retry limit is three attempts."),),
        provenance=("e1",),
        cost=1,
    )
    state = ResearchState()
    state.apply(request, outcome)
    state.steps.append(SEARCH)
    first.checkpoint(session.id, state, request, outcome)

    restarted = _manager(tmp_path)
    restored = restarted.restore(session.id)

    assert restarted.already_done(session.id, request) is True
    assert [item.evidence_id for item in restored.evidence] == ["e1"]
    assert restored.cost == 1
    assert SEARCH in restored.steps


def test_resume_after_a_crash_does_not_search_again(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session = manager.create(user_id="ada", question="what is the retry limit", top_k=3)
    search = _CountingSearch()
    adapters = {item.action: item for item in mock_action_adapters()}
    adapters[SEARCH] = search
    dispatcher = ActionDispatcher(list(adapters.values()), budget=16, max_iterations=8)

    def crash_after_search(state, request=None, outcome=None):
        manager.checkpoint(session.id, state, request, outcome)
        if request is not None and request.action == SEARCH:
            raise RuntimeError("worker crashed")

    try:
        run_cycle(dispatcher, "what is the retry limit", on_progress=crash_after_search)
    except RuntimeError:
        pass

    restored = manager.restore(session.id)
    result = run_cycle(dispatcher, "what is the retry limit", state=restored)

    assert len(search.calls) == 1
    assert SEARCH in result.steps
    assert SYNTHESIZE in result.steps
    assert result.report


def test_apply_does_not_duplicate_a_replayed_action() -> None:
    request = ActionRequest(SEARCH, query="retry limit")
    outcome = ActionOutcome(
        action=SEARCH,
        ok=True,
        evidence=(EvidenceRef("e1", "d1", "notes.md", "three"),),
        cost=1,
    )
    state = ResearchState()
    state.apply(request, outcome)
    state.apply(request, outcome)

    assert len(state.evidence) == 1
    assert state.cost == 1
    assert state.iterations == 1


def test_expired_lease_is_requeued_without_losing_the_checkpoint(tmp_path: Path) -> None:
    manager = _manager(tmp_path, lease_seconds=900)
    session = manager.create(user_id="ada", question="what is the retry limit", top_k=3)
    request = ActionRequest(SEARCH, query="what is the retry limit")
    outcome = ActionOutcome(
        action=SEARCH,
        ok=True,
        evidence=(EvidenceRef("e1", "d1", "notes.md", "three"),),
        cost=1,
    )
    state = ResearchState()
    state.apply(request, outcome)
    manager.checkpoint(session.id, state, request, outcome)
    manager.acquire(session.id, owner="worker-a")
    with manager.store._connect() as connection:
        connection.execute(
            """
            UPDATE research_sessions
            SET lease_until = '2000-01-01T00:00:00+00:00'
            WHERE id = ?
            """,
            (session.id,),
        )

    recovered = manager.recover()
    waiting = manager.get(session.id)
    claimed = manager.claim()

    assert recovered == 1
    assert waiting is not None
    assert waiting.status == "queued"
    assert claimed is not None
    assert claimed.id == session.id
    assert manager.already_done(session.id, request) is True
    assert [item.evidence_id for item in manager.restore(session.id).evidence] == ["e1"]


def test_completed_session_is_unchanged_after_restart(tmp_path: Path) -> None:
    first = _manager(tmp_path)
    session = first.create(user_id="ada", question="what is the retry limit", top_k=3)
    state = ResearchState()
    state.report = "The retry limit is three. [1]"
    first.complete(
        session.id,
        {
            "action": "research",
            "report": state.report,
            "sufficient_evidence": True,
            "steps": [SEARCH, REVERIFY, SYNTHESIZE],
            "gaps": [],
            "contradictions": [],
            "cost": 1,
        },
        state,
    )

    restarted = _manager(tmp_path)
    stored = restarted.get(session.id)

    assert stored is not None
    assert stored.status == "completed"
    assert stored.report == "The retry limit is three. [1]"
    assert stored.sufficient_evidence is True


def test_halted_plan_keeps_evidence_and_unresolved_contradictions(tmp_path: Path) -> None:
    first = _manager(tmp_path)
    session = first.create(user_id="ada", question="what is the retry limit", top_k=3)
    request = ActionRequest(SEARCH, query="what is the retry limit")
    outcome = ActionOutcome(
        action=SEARCH,
        ok=True,
        evidence=(EvidenceRef("e1", "d1", "notes.md", "The retry limit is three attempts."),),
        provenance=("e1",),
        cost=1,
    )
    state = ResearchState()
    state.apply(request, outcome)
    state.steps.append(SEARCH)
    state.contradictions = ("notes.md says three; guide.md says five",)
    state.dag = {
        "tasks": [
            {"id": "search:0", "action": SEARCH, "status": "done", "depends_on": []},
            {"id": "reverify:0", "action": REVERIFY, "status": "pending", "depends_on": ["search:0"]},
            {"id": "synthesize", "action": SYNTHESIZE, "status": "pending", "depends_on": ["reverify:0"]},
        ]
    }
    first.checkpoint(session.id, state, request, outcome)
    first.halt(
        session.id,
        {"action": "halt", "steps": [SEARCH], "contradictions": [], "gaps": []},
        state,
    )

    restarted = _manager(tmp_path)
    stored = restarted.get(session.id)
    restored = restarted.restore(session.id)
    events = restarted.action_log(session.id)
    by_action = {item["action"]: item["status"] for item in restored.dag.get("tasks", [])}

    assert stored is not None
    assert stored.status == "halted"
    assert list(stored.contradictions) == ["notes.md says three; guide.md says five"]
    assert list(restored.contradictions) == ["notes.md says three; guide.md says five"]
    assert [item.evidence_id for item in restored.evidence] == ["e1"]
    assert by_action[SEARCH] == "done"
    assert by_action[REVERIFY] == "pending"
    assert by_action[SYNTHESIZE] == "pending"
    assert events[0]["action"] == SEARCH
    assert events[0]["evidence"][0]["evidence_id"] == "e1"
