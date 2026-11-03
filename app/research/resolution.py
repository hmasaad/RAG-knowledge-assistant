"""Resolve contradicting claims by version, source authority, or human approval."""

from datetime import datetime, timezone

AUTHORITY = {
    "news": 3,
    "aggregator": 2,
    "social": 1,
}


class ApprovalLog:
    """In-memory record of human decisions on conflicts the engine could not settle."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], dict] = {}

    def decide(self, asset: str, conflict_id: str, winner_id: str, reviewer: str) -> dict:
        row = {
            "winner_id": winner_id,
            "reviewer": reviewer or "local",
            "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        self._rows[(asset.strip().upper(), conflict_id)] = row
        return row

    def get(self, asset: str, conflict_id: str) -> dict | None:
        return self._rows.get((asset.strip().upper(), conflict_id))

    def clear(self) -> None:
        self._rows.clear()


approval_log = ApprovalLog()


def resolve_conflicts(scene: dict, asset: str) -> list[dict]:
    """Detect every contradiction and record how it was settled."""
    claims = {item["id"]: item for item in scene["claims"]}
    seen: set[tuple[str, str]] = set()
    found: list[dict] = []
    for claim in scene["claims"]:
        for other_id in claim.get("contradicts") or []:
            pair = tuple(sorted((claim["id"], other_id)))
            if pair in seen or other_id not in claims:
                continue
            seen.add(pair)
            resolution = _decide(claims[pair[0]], claims[pair[1]], asset)
            found.append(resolution)
            _annotate(claims[pair[0]], claims[pair[1]], resolution)
    return found


def _decide(left: dict, right: dict, asset: str) -> dict:
    conflict_id = f"{left['id']}|{right['id']}"
    left_role, left_rank = _authority(left)
    right_role, right_rank = _authority(right)
    base = {
        "id": conflict_id,
        "left_id": left["id"],
        "right_id": right["id"],
        "left_role": left_role,
        "right_role": right_role,
        "left_authority": left_rank,
        "right_authority": right_rank,
        "reviewer": "",
    }
    superseded = _by_version(left, right)
    if superseded is not None:
        winner, loser = superseded
        return {
            **base,
            "status": "superseded",
            "winner_id": winner["id"],
            "loser_id": loser["id"],
            "reason": (
                f"{winner['short']} v{winner['provenance']['version']} supersedes "
                f"{loser['short']} v{loser['provenance']['version']}"
            ),
        }
    if left_rank != right_rank:
        winner, loser = (left, right) if left_rank > right_rank else (right, left)
        return {
            **base,
            "status": "authority",
            "winner_id": winner["id"],
            "loser_id": loser["id"],
            "reason": (
                f"{_authority(winner)[0]} (rank {_authority(winner)[1]}) outranks "
                f"{_authority(loser)[0]} (rank {_authority(loser)[1]})"
            ),
        }
    approval = approval_log.get(asset, conflict_id)
    if approval and approval["winner_id"] in {left["id"], right["id"]}:
        winner = left if approval["winner_id"] == left["id"] else right
        loser = right if winner is left else left
        return {
            **base,
            "status": "approved",
            "winner_id": winner["id"],
            "loser_id": loser["id"],
            "reviewer": approval["reviewer"],
            "recorded_at": approval["recorded_at"],
            "reason": f"{approval['reviewer']} approved {winner['short']}",
        }
    return {
        **base,
        "status": "pending",
        "winner_id": "",
        "loser_id": "",
        "reason": "Same source authority and version. A person needs to approve a winner.",
    }


def _by_version(left: dict, right: dict) -> tuple[dict, dict] | None:
    left_doc = left["provenance"]["document_id"]
    right_doc = right["provenance"]["document_id"]
    if not left_doc or left_doc != right_doc:
        return None
    left_version = int(left["provenance"]["version"])
    right_version = int(right["provenance"]["version"])
    if left_version == right_version:
        return None
    if left_version > right_version:
        return left, right
    return right, left


def _authority(claim: dict) -> tuple[str, int]:
    role = claim["agent_id"].rsplit(":", 1)[-1]
    return role, AUTHORITY.get(role, 0)


def _annotate(left: dict, right: dict, resolution: dict) -> None:
    detail = resolution["reason"]
    status = resolution["status"]
    for claim in (left, right):
        if status == "pending":
            decision = "pending_approval"
        elif status == "approved":
            decision = "approved"
        elif claim["id"] == resolution["winner_id"] and status == "superseded":
            decision = "supersedes"
        elif claim["id"] == resolution["loser_id"] and status == "superseded":
            decision = "superseded"
        else:
            decision = "authority"
        if status == "approved":
            recorded = resolution.get("recorded_at") or claim["provenance"].get("recorded_at") or ""
        else:
            recorded = claim["provenance"].get("recorded_at") or ""
        claim.setdefault("audit", []).append(
            {
                "decision": decision,
                "detail": detail,
                "recorded_at": recorded,
                "evidence_ids": [left["id"], right["id"]],
            }
        )
