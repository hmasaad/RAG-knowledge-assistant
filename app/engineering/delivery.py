"""Turn a requirement into code, tests, a review, and a pull request draft.

The pipeline understands the request, plans the files, writes a Python change,
runs that change's tests in a temporary directory, and reviews the result.
A passing implementation is ready as a pull request draft. The draft is not opened.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from app.engineering.practice import observe_practice
from app.engineering.repository import inspect_repository

_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")
_FUNCTION = re.compile(r"\bfunction (?:called |named )([A-Za-z_][A-Za-z0-9_]*)", re.I)
_VERB = re.compile(r"\b(greet|sum|add|format|parse|validate|count)\w*\b", re.I)


def deliver(requirement: str) -> dict:
    """Run the six delivery stages for one requirement."""
    text = str(requirement or "").strip()
    if not text:
        raise ValueError("A requirement is required.")
    repository = inspect_repository(requirement=text)
    understood = _understand(text)
    files = _write_files(understood)
    red = _run_tests([files[1]])
    tests = _run_tests(files)
    healed = False
    if tests["status"] != "passed":
        repaired = repair_module(files[0]["content"], tests["output"])
        if repaired:
            files[0]["content"] = repaired
            tests = _run_tests(files)
            healed = tests["status"] == "passed"
    review = _review(understood, files, tests)
    pull = _pull_request(understood, review)
    stages = [
        {
            "stage": "understand",
            "status": "done",
            "detail": understood["summary"],
        },
        {
            "stage": "plan",
            "status": "done",
            "detail": _plan_detail(understood, repository),
        },
        {
            "stage": "code",
            "status": "done",
            "detail": f"Wrote {files[0]['path']} and {files[1]['path']}.",
        },
        {
            "stage": "test",
            "status": tests["status"],
            "detail": tests["summary"],
        },
        {
            "stage": "review",
            "status": review["decision"],
            "detail": review["reason"],
        },
        {
            "stage": "pull_request",
            "status": "draft" if pull["ready"] else "held",
            "detail": pull["title"],
        },
    ]
    if review["decision"] == "approve":
        summary = "Tests passed. Review approves the change. The pull request draft is ready."
    elif review["decision"] == "recorded":
        summary = "Tests passed. The change records the requirement. The pull request stays held."
    else:
        summary = "Review requests changes. The pull request stays held."
    practice = observe_practice(
        understood=understood,
        files=files,
        red=red,
        tests=tests,
        healed=healed,
        review=review,
        pull=pull,
    )
    pull["ready"] = practice["ready"]
    stages[-1]["status"] = "draft" if pull["ready"] else "held"
    return {
        "summary": summary,
        "stages": stages,
        "files": files,
        "tests": tests,
        "review": review,
        "pull_request": pull,
        "practice": {"summary": practice["summary"], "notes": practice["notes"]},
        "repository": repository,
    }


def _understand(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    checks = []
    prose = []
    for line in lines:
        if line[:1] in "-*" or re.match(r"\d+[.)]\s", line):
            checks.append(line.lstrip("-*0123456789.) ").strip())
        else:
            prose.append(line)
    goal = prose[0] if prose else text
    if not checks:
        checks = [goal]
    behaviors = _behaviors(goal, checks)
    return {
        "goal": goal,
        "checks": checks,
        "behaviors": behaviors,
        "function": _function_name(goal),
        "summary": (
            f"Goal: {goal} {len(checks)} acceptance checks. "
            f"{len(behaviors)} can be implemented directly."
        ),
    }


def _behaviors(goal: str, checks: list[str]) -> list[dict]:
    blob = f"{goal} {' '.join(checks)}"
    found = []
    if re.search(r"\b(empty|blank)\b", blob, re.I) and re.search(
        r"\b(reject|refuse|invalid|error)", blob, re.I
    ):
        found.append({"kind": "reject_empty"})
    for check in checks:
        match = re.search(r"\bincludes?\s+(?:the\s+)?(.+)", check, re.I)
        if not match:
            continue
        target = match.group(1).strip(" .")
        if target.casefold() in {"name", "value", "input"}:
            found.append({"kind": "include_argument"})
        elif target:
            found.append({"kind": "include_literal", "target": target})
    return found


def _function_name(goal: str) -> str:
    match = _FUNCTION.search(goal)
    if match and _NAME.match(_snake(match.group(1))):
        return _snake(match.group(1))
    match = re.search(r"\bthat\s+([A-Za-z]+)", goal)
    if match:
        word = _stem(match.group(1))
        if _NAME.match(word) and word not in {"a", "an", "the"}:
            return word
    match = _VERB.search(goal)
    if match:
        word = _stem(match.group(1))
        if _NAME.match(word):
            return word
    return "deliver"


def _stem(word: str) -> str:
    value = word.lower()
    if value.endswith("s") and value not in {"sum"} and len(value) > 3:
        return value[:-1]
    return value


def _write_files(understood: dict) -> list[dict]:
    name = understood["function"]
    if understood["behaviors"]:
        module = _behavior_module(name, understood["behaviors"])
        test = _behavior_test(name, understood["behaviors"])
    else:
        module = _spec_module(understood)
        test = _spec_test(understood)
    return [
        {"path": "change.py", "content": module},
        {"path": "test_change.py", "content": test},
    ]


def _behavior_module(name: str, behaviors: list[dict]) -> str:
    parameter = "name" if any(item["kind"] == "include_argument" for item in behaviors) or name == "greet" else "value"
    literals = [item["target"] for item in behaviors if item["kind"] == "include_literal"]
    lines = [f"def {name}({parameter}: str) -> str:", f"    cleaned = ' '.join(str({parameter}).split())"]
    if any(item["kind"] == "include_argument" for item in behaviors) or name == "greet":
        lines.append('    result = f"Hello, {cleaned}."')
    else:
        lines.append("    result = cleaned")
    for literal in literals:
        lines.append(f"    if {literal!r}.casefold() not in result.casefold():")
        lines.append(f"        result = result + ' ' + {literal!r}")
    lines.append("    return result")
    lines.append("")
    return "\n".join(lines)


def repair_module(source: str, output: str) -> str | None:
    """Insert a missing empty-value check when the tests expected ValueError."""
    if "DID NOT RAISE" not in output:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    changed = False
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if any(isinstance(child, ast.Raise) for child in ast.walk(node)):
            continue
        guard = ast.parse("if not cleaned:\n    raise ValueError('A value is required.')\n").body[0]
        node.body.insert(1, guard)
        changed = True
    if not changed:
        return None
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _behavior_test(name: str, behaviors: list[dict]) -> str:
    parameter = "name" if any(item["kind"] == "include_argument" for item in behaviors) or name == "greet" else "value"
    lines = ["import pytest", "", "from change import " + name, ""]
    if any(item["kind"] == "include_argument" for item in behaviors):
        lines.extend(
            [
                f"def test_{name}_includes_the_argument():",
                f"    assert 'Ada' in {name}('Ada')",
                "",
            ]
        )
    for index, item in enumerate(behavior for behavior in behaviors if behavior["kind"] == "include_literal"):
        lines.extend(
            [
                f"def test_{name}_includes_literal_{index}():",
                f"    assert {item['target']!r}.casefold() in {name}('Ada').casefold()",
                "",
            ]
        )
    if any(item["kind"] == "reject_empty" for item in behaviors):
        lines.extend(
            [
                f"def test_{name}_rejects_empty():",
                "    with pytest.raises(ValueError):",
                f"        {name}('   ')",
                "",
            ]
        )
    if not any(item["kind"] in {"include_argument", "include_literal", "reject_empty"} for item in behaviors):
        lines.extend([f"def test_{name}_returns_text():", f"    assert {name}('Ada')", ""])
    return "\n".join(lines)


def _spec_module(understood: dict) -> str:
    checks = ",\n".join(f"    {check!r}" for check in understood["checks"])
    return (
        f"REQUIREMENT = {understood['goal']!r}\n"
        "ACCEPTANCE = (\n"
        f"{checks},\n"
        ")\n\n"
        "def specification() -> dict:\n"
        '    return {"requirement": REQUIREMENT, "acceptance": ACCEPTANCE}\n'
    )


def _spec_test(understood: dict) -> str:
    lines = ["from change import ACCEPTANCE, specification", ""]
    lines.append("def test_specification_records_the_requirement():")
    lines.append("    recorded = specification()")
    lines.append(f"    assert recorded['requirement'] == {understood['goal']!r}")
    for index, check in enumerate(understood["checks"]):
        lines.append(f"    assert recorded['acceptance'][{index}] == {check!r}")
    lines.append("    assert ACCEPTANCE == recorded['acceptance']")
    lines.append("")
    return "\n".join(lines)


def _run_tests(files: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix="delivery-") as directory:
        root = Path(directory)
        for item in files:
            (root / item["path"]).write_text(item["content"], encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "test_change.py", "-q", "--noconftest"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _test_result("failed", 0, 0, "The test run exceeded 30 seconds.")
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    if len(output) > 2000:
        output = output[:2000].rstrip() + "…"
    passed = _count(output, "passed")
    failed = _count(output, "failed")
    status = "passed" if completed.returncode == 0 and passed > 0 else "failed"
    summary = f"{passed} passed, {failed} failed."
    return _test_result(status, passed, failed, summary, output)


def _review(understood: dict, files: list[dict], tests: dict) -> dict:
    module = files[0]["content"]
    if tests["status"] != "passed":
        return {
            "decision": "request_changes",
            "reason": "The generated tests did not pass, so the change is not ready.",
        }
    if not understood["behaviors"]:
        return {
            "decision": "recorded",
            "reason": "The change records the requirement and its acceptance checks. It does not implement a behavior yet.",
        }
    missing = [item for item in understood["checks"] if item.casefold() not in module.casefold() and item not in files[1]["content"]]
    if tests["passed"] < len(understood["behaviors"]):
        return {
            "decision": "request_changes",
            "reason": "A behavior does not have a passing test.",
        }
    if missing and tests["passed"] == 0:
        return {"decision": "request_changes", "reason": "An acceptance check is missing from the change."}
    return {
        "decision": "approve",
        "reason": f"The tests passed and {len(understood['behaviors'])} behaviors are implemented.",
    }


def _pull_request(understood: dict, review: dict) -> dict:
    title = understood["goal"]
    if len(title) > 72:
        title = title[:69].rstrip() + "..."
    checks = "\n".join(f"- [ ] {check}" for check in understood["checks"])
    body = f"## Summary\n- {understood['goal']}\n\n## Test plan\n{checks}\n- [ ] pytest test_change.py\n"
    ready = review["decision"] == "approve"
    return {
        "title": title,
        "body": body,
        "base": "main",
        "head": f"delivery/{understood['function']}",
        "ready": ready,
        "opened": False,
    }


def _plan_detail(understood: dict, repository: dict) -> str:
    if understood["behaviors"]:
        base = f"Plan: implement {understood['function']} in change.py, then cover it in test_change.py."
    else:
        base = "Plan: record the requirement in change.py and lock the acceptance checks in test_change.py."
    relevant = repository.get("relevant") or []
    if relevant:
        names = ", ".join(item["path"] for item in relevant[:3])
        return f"{base} Read existing patterns in {names} first."
    return f"{base} No existing file matches the requirement."


def _test_result(status: str, passed: int, failed: int, summary: str, output: str = "") -> dict:
    return {
        "status": status,
        "passed": passed,
        "failed": failed,
        "summary": summary,
        "output": output or summary,
    }


def _count(output: str, label: str) -> int:
    match = re.search(rf"(\d+) {label}", output)
    return int(match.group(1)) if match else 0


def _snake(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "", value).lower()
