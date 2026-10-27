"""Assign specialized agents and resolve disagreements on a shared board."""

from dataclasses import asdict, dataclass
from itertools import combinations

from app.research.adapters import cited_answer
from app.research.contracts import (
    RESOLVE_CONTRADICTIONS,
    REVERIFY,
    SEARCH,
    SYNTHESIZE,
    ActionOutcome,
    ActionRequest,
    ResearchState,
)
from app.research.planner import ResearchPlan, build_plan
from app.research.verification import Contradiction
from app.retrieval.conflicts import numeric_facts

SEARCHER = "searcher"
VERIFIER = "verifier"
REFEREE = "referee"
SYNTHESIZER = "synthesizer"

CITE_BOTH = "cite_both"
PREFER_SUPPORTED = "prefer_supported"
UNRESOLVED = "unresolved"
NO_CONFLICT = "no_conflict"

_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "of",
        "and",
        "or",
        "to",
        "for",
        "in",
        "on",
        "with",
        "that",
        "this",
        "by",
        "from",
        "states",
    }
)


@dataclass(frozen=True)
class AgentAssignment:
    agent_id: str
    role: str
    query: str
    tool: str
    action: str
    depends_on: tuple[str, ...] = ()

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["depends_on"] = list(self.depends_on)
        return payload


@dataclass(frozen=True)
class AgentFinding:
    agent_id: str
    role: str
    claim: str
    verdict: str
    source: str = ""
    values: tuple[int, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["values"] = list(self.values)
        payload["evidence_ids"] = list(self.evidence_ids)
        return payload


@dataclass(frozen=True)
class ConflictResolution:
    decision: str
    summary: str
    kept_claims: tuple[str, ...] = ()
    dropped_claims: tuple[str, ...] = ()
    left_agent: str = ""
    right_agent: str = ""

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["kept_claims"] = list(self.kept_claims)
        payload["dropped_claims"] = list(self.dropped_claims)
        return payload


@dataclass(frozen=True)
class CoordinationReport:
    question: str
    assignments: tuple[AgentAssignment, ...]
    findings: tuple[AgentFinding, ...] = ()
    resolutions: tuple[ConflictResolution, ...] = ()
    consensus: tuple[str, ...] = ()
    resolved: bool = True

    def to_payload(self) -> dict:
        return {
            "question": self.question,
            "assignments": [item.to_payload() for item in self.assignments],
            "findings": [item.to_payload() for item in self.findings],
            "resolutions": [item.to_payload() for item in self.resolutions],
            "consensus": list(self.consensus),
            "resolved": self.resolved,
        }

    @classmethod
    def from_payload(cls, payload: dict | None) -> "CoordinationReport":
        data = payload or {}
        return cls(
            question=str(data.get("question", "")),
            assignments=tuple(AgentAssignment(**_assignment(item)) for item in data.get("assignments", ()) or ()),
            findings=tuple(finding_from_payload(item) for item in data.get("findings", ()) or ()),
            resolutions=tuple(_resolution(item) for item in data.get("resolutions", ()) or ()),
            consensus=tuple(str(item) for item in data.get("consensus", ()) or ()),
            resolved=bool(data.get("resolved", True)),
        )


class ConflictReferee:
    """Keep both supported sides of a numeric fight; drop an unsupported one."""

    def resolve(
        self,
        findings: tuple[AgentFinding, ...] | list[AgentFinding] = (),
        conflicts: tuple[Contradiction, ...] | list[Contradiction] = (),
    ) -> tuple[ConflictResolution, ...]:
        rows = list(findings)
        decided: list[ConflictResolution] = []
        for item in conflicts:
            left = _finding_for(rows, item.left_filename, item.left_values)
            right = _finding_for(rows, item.right_filename, item.right_values)
            decided.append(
                ConflictResolution(
                    CITE_BOTH,
                    item.summary,
                    kept_claims=_kept(left, right),
                    dropped_claims=(),
                    left_agent=left.agent_id if left else "",
                    right_agent=right.agent_id if right else "",
                )
            )
        used = {(item.left_agent, item.right_agent) for item in decided}
        for left, right in combinations(rows, 2):
            if (left.agent_id, right.agent_id) in used or (right.agent_id, left.agent_id) in used:
                continue
            resolution = _pair(left, right)
            if resolution is None:
                continue
            decided.append(resolution)
        if not decided:
            return (
                ConflictResolution(
                    NO_CONFLICT,
                    "Agents agree or the findings are different facts.",
                ),
            )
        return tuple(decided)


class MultiAgentCoordinator:
    """Give each clause a searcher and a verifier, then a referee and a synthesizer."""

    def __init__(self, *, denied_actions: frozenset[str] | set[str] = frozenset()) -> None:
        self.denied_actions = frozenset(denied_actions)
        self.referee = ConflictReferee()

    def assign(self, plan: ResearchPlan) -> tuple[AgentAssignment, ...]:
        assignments: list[AgentAssignment] = []
        verifiers: list[str] = []
        seen: set[tuple[str, str]] = set()
        for task in plan.tasks:
            searcher_id = f"searcher:{task.task_id}"
            verifier_id = f"verifier:{task.task_id}"
            search_key = (SEARCHER, task.question.casefold())
            if search_key not in seen:
                assignments.append(
                    AgentAssignment(searcher_id, SEARCHER, task.question, task.tool, SEARCH)
                )
                seen.add(search_key)
            if (VERIFIER, task.question.casefold()) not in seen:
                assignments.append(
                    AgentAssignment(
                        verifier_id,
                        VERIFIER,
                        task.question,
                        task.tool,
                        REVERIFY,
                        (searcher_id,),
                    )
                )
                seen.add((VERIFIER, task.question.casefold()))
                verifiers.append(verifier_id)
        parents = tuple(verifiers)
        if RESOLVE_CONTRADICTIONS not in self.denied_actions:
            assignments.append(
                AgentAssignment(
                    "referee",
                    REFEREE,
                    plan.question,
                    "knowledge",
                    RESOLVE_CONTRADICTIONS,
                    parents,
                )
            )
            parents = ("referee",)
        assignments.append(
            AgentAssignment(
                "synthesizer",
                SYNTHESIZER,
                plan.question,
                "knowledge",
                SYNTHESIZE,
                parents,
            )
        )
        return tuple(assignments)

    def coordinate(
        self,
        question: str,
        *,
        plan: ResearchPlan | None = None,
        findings: tuple[AgentFinding, ...] | list[AgentFinding] = (),
        conflicts: tuple[Contradiction, ...] | list[Contradiction] = (),
        budget: int = 16,
    ) -> CoordinationReport:
        resolved_plan = plan or build_plan(question, budget=budget)
        assignments = self.assign(resolved_plan)
        collected = tuple(findings)
        resolutions = self.referee.resolve(collected, conflicts)
        kept: list[str] = []
        for item in resolutions:
            kept.extend(item.kept_claims)
        if not kept:
            kept = [item.claim for item in collected if item.verdict == "supported"]
        return CoordinationReport(
            question=question.strip(),
            assignments=assignments,
            findings=collected,
            resolutions=resolutions,
            consensus=_unique(kept),
            resolved=all(item.decision != UNRESOLVED for item in resolutions),
        )

    def publish(self, state: ResearchState, outcome: ActionOutcome, *, agent_id: str, role: str) -> None:
        """Write this specialist's evidence onto the shared board."""
        for item in outcome.evidence:
            finding = AgentFinding(
                agent_id=agent_id,
                role=role or SEARCHER,
                claim=item.text,
                verdict="supported",
                source=item.source,
                values=tuple(sorted(numeric_facts(item.text))),
                evidence_ids=(item.evidence_id,),
            )
            payload = finding.to_payload()
            if payload not in state.findings:
                state.findings.append(payload)
        for verdict in outcome.verdicts:
            finding = AgentFinding(
                agent_id=agent_id,
                role=role or VERIFIER,
                claim=verdict.claim,
                verdict=verdict.verdict,
                values=tuple(sorted(numeric_facts(verdict.claim))),
                evidence_ids=verdict.evidence_ids,
            )
            payload = finding.to_payload()
            if payload not in state.findings:
                state.findings.append(payload)

    def settle(self, state: ResearchState, question: str, *, budget: int = 16) -> CoordinationReport:
        """Resolve the board before a synthesizer writes the answer."""
        report = self.coordinate(
            question,
            findings=findings_from_state(state),
            budget=budget,
        )
        state.coordination = report.to_payload()
        return report

    def investigate(
        self,
        dispatcher,
        question: str,
        *,
        state: ResearchState | None = None,
        user_id: str = "local",
        top_k: int = 4,
        budget: int | None = None,
    ) -> tuple[ResearchState, CoordinationReport]:
        """Run searchers and verifiers on one board, then settle before synthesize."""
        session = state if state is not None else ResearchState()
        limit = budget if budget is not None else getattr(dispatcher, "budget", 16)
        for assignment in self.assign(build_plan(question, budget=limit)):
            if assignment.action == SYNTHESIZE:
                self.settle(session, question, budget=limit)
                request = ActionRequest(
                    SYNTHESIZE,
                    query=cited_answer(question, session),
                    user_id=user_id,
                    top_k=top_k,
                )
            elif assignment.action == REVERIFY:
                request = ActionRequest(REVERIFY, claim=assignment.query, user_id=user_id, top_k=top_k)
            else:
                request = ActionRequest(assignment.action, query=assignment.query, user_id=user_id, top_k=top_k)
            if session.fingerprint(request) in session.executed:
                continue
            outcome = dispatcher.dispatch(request, session)
            session.steps.append(assignment.action)
            self.publish(session, outcome, agent_id=assignment.agent_id, role=assignment.role)
        report = self.settle(session, question, budget=limit)
        return session, report


def findings_from_state(state) -> tuple[AgentFinding, ...]:
    stored = [
        finding_from_payload(item)
        for item in getattr(state, "findings", ()) or []
        if isinstance(item, dict)
    ]
    if stored:
        return tuple(stored)
    return findings_from_report(state)


def findings_from_report(report) -> tuple[AgentFinding, ...]:
    """Lift stored claims and documented conflicts onto the shared board."""
    findings: list[AgentFinding] = []
    records = list(getattr(report, "evidence_records", ()) or ())
    for index, record in enumerate(records, start=1):
        findings.append(
            AgentFinding(
                agent_id=f"searcher:{index}",
                role=SEARCHER,
                claim=record.claim,
                verdict=record.verdict or "supported",
                source=record.source.filename,
                values=tuple(sorted(numeric_facts(record.claim))),
                evidence_ids=(record.evidence_id,),
            )
        )
    if findings:
        return tuple(findings)
    for index, item in enumerate(getattr(report, "conflicts", ()) or (), start=1):
        findings.append(
            AgentFinding(
                agent_id=f"searcher:left:{index}",
                role=SEARCHER,
                claim=f"{item.left_filename} states {', '.join(str(value) for value in item.left_values)}",
                verdict="supported",
                source=item.left_filename,
                values=tuple(item.left_values),
            )
        )
        findings.append(
            AgentFinding(
                agent_id=f"searcher:right:{index}",
                role=SEARCHER,
                claim=f"{item.right_filename} states {', '.join(str(value) for value in item.right_values)}",
                verdict="supported",
                source=item.right_filename,
                values=tuple(item.right_values),
            )
        )
    return tuple(findings)


def finding_from_payload(payload: dict) -> AgentFinding:
    return AgentFinding(
        agent_id=str(payload.get("agent_id", "")),
        role=str(payload.get("role", SEARCHER)),
        claim=str(payload.get("claim", "")),
        verdict=str(payload.get("verdict", "")),
        source=str(payload.get("source", "")),
        values=tuple(int(value) for value in payload.get("values", ()) or ()),
        evidence_ids=tuple(str(item) for item in payload.get("evidence_ids", ()) or ()),
    )


def _pair(left: AgentFinding, right: AgentFinding) -> ConflictResolution | None:
    if not _same_scope(left, right):
        return None
    left_values = set(left.values) or numeric_facts(left.claim)
    right_values = set(right.values) or numeric_facts(right.claim)
    left_ok = left.verdict == "supported"
    right_ok = right.verdict == "supported"
    if left_values and right_values and not left_values & right_values and left_ok and right_ok:
        return ConflictResolution(
            CITE_BOTH,
            f"{left.source or left.agent_id} and {right.source or right.agent_id} disagree.",
            kept_claims=(left.claim, right.claim),
            dropped_claims=(),
            left_agent=left.agent_id,
            right_agent=right.agent_id,
        )
    if left_ok != right_ok:
        winner, loser = (left, right) if left_ok else (right, left)
        return ConflictResolution(
            PREFER_SUPPORTED,
            "Keep the supported finding and drop the unsupported one.",
            kept_claims=(winner.claim,),
            dropped_claims=(loser.claim,),
            left_agent=left.agent_id,
            right_agent=right.agent_id,
        )
    if not left_ok and not right_ok:
        return ConflictResolution(
            UNRESOLVED,
            "Neither agent supported the claim.",
            kept_claims=(),
            dropped_claims=(left.claim, right.claim),
            left_agent=left.agent_id,
            right_agent=right.agent_id,
        )
    return None


def _same_scope(left: AgentFinding, right: AgentFinding) -> bool:
    if left.source and right.source and left.source != right.source and (left.values or right.values):
        return True
    left_words = _tokens(left.claim)
    right_words = _tokens(right.claim)
    return bool(left_words & right_words)


def _tokens(text: str) -> frozenset[str]:
    words = {token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if token}
    return frozenset(word for word in words if word not in _STOP and not word.isdigit())


def _finding_for(
    findings: list[AgentFinding],
    filename: str,
    values: tuple[int, ...],
) -> AgentFinding | None:
    wanted = set(values)
    for item in findings:
        if filename and item.source == filename:
            return item
        if wanted and set(item.values) == wanted:
            return item
    return None


def _kept(left: AgentFinding | None, right: AgentFinding | None) -> tuple[str, ...]:
    return tuple(item.claim for item in (left, right) if item is not None)


def _assignment(payload: dict) -> dict:
    return {
        "agent_id": str(payload.get("agent_id", "")),
        "role": str(payload.get("role", "")),
        "query": str(payload.get("query", "")),
        "tool": str(payload.get("tool", "")),
        "action": str(payload.get("action", "")),
        "depends_on": tuple(str(item) for item in payload.get("depends_on", ()) or ()),
    }


def _resolution(payload: dict) -> ConflictResolution:
    return ConflictResolution(
        decision=str(payload.get("decision", "")),
        summary=str(payload.get("summary", "")),
        kept_claims=tuple(str(item) for item in payload.get("kept_claims", ()) or ()),
        dropped_claims=tuple(str(item) for item in payload.get("dropped_claims", ()) or ()),
        left_agent=str(payload.get("left_agent", "")),
        right_agent=str(payload.get("right_agent", "")),
    )


def _unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in values:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return tuple(ordered)
