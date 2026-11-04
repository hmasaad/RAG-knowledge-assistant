"""Engineering practice notes for one delivery.

The notes read the generated files, the red and green test runs, and the
review. They do not open the pull request. A person does that.
"""

from __future__ import annotations

import ast


def observe_practice(
    *,
    understood: dict,
    files: list[dict],
    red: dict,
    tests: dict,
    healed: bool,
    review: dict,
    pull: dict,
) -> dict:
    """Return the twelve practice notes and whether the draft may be marked ready."""
    module = _file(files, "change.py")
    tests_source = _file(files, "test_change.py")
    graph = _graph(module, tests_source)
    policy = _policy(module, tests_source)
    votes = _votes(tests, review, policy)
    notes = [
        _repository(module, tests_source),
        graph,
        _agents(review),
        _specification(understood),
        _planning(),
        _ast_note(module),
        _tdd(red, tests),
        _healing(healed, tests),
        _review_note(votes),
        policy,
        _debug(red, tests),
        _governance(pull, policy),
    ]
    ready = bool(pull.get("ready")) and policy["decision"] == "enforced"
    status = next(note["decision"] for note in notes if note["component"] == "review")
    summary = f"Review {status}. Policy {policy['decision']}. The draft is held for a person."
    return {"summary": summary, "notes": notes, "ready": ready}


def _repository(module: str, tests_source: str) -> dict:
    functions = _functions(module)
    imports = _imports(module) + _imports(tests_source)
    names = ", ".join(functions) if functions else "no functions"
    imported = ", ".join(imports) if imports else "no imports"
    return _note(
        "repository",
        "mapped",
        f"The change spans change.py and test_change.py. Functions: {names}. Imports: {imported}.",
    )


def _graph(module: str, tests_source: str) -> dict:
    functions = set(_functions(module))
    edges = []
    for name, calls in _calls(tests_source).items():
        for target in calls:
            if target in functions:
                edges.append(f"{name} calls {target}")
    if not edges:
        detail = "The test module does not call a function defined in change.py."
    else:
        detail = "Dependencies: " + "; ".join(edges) + "."
    return _note("graph", "linked" if edges else "isolated", detail)


def _agents(review: dict) -> dict:
    return _note(
        "agents",
        "assigned",
        (
            "Specifier wrote the checks. Planner named the files. Coder wrote the change. "
            f"Tester ran pytest. Reviewer decided {review.get('decision') or 'unknown'}. "
            "Governor holds the draft."
        ),
    )


def _specification(understood: dict) -> dict:
    checks = understood.get("checks") or []
    if understood.get("behaviors"):
        return _note(
            "specification",
            "generated",
            f"{len(checks)} acceptance checks selected the implementation.",
        )
    return _note(
        "specification",
        "recorded",
        "No direct behavior was recognized, so the change stores the requirement and its checks.",
    )


def _planning() -> dict:
    return _note(
        "planning",
        "followed",
        "The plan named change.py and test_change.py. The coder wrote those files and no others.",
    )


def _ast_note(module: str) -> dict:
    try:
        tree = ast.parse(module)
    except SyntaxError as exc:
        return _note("ast", "invalid", f"The module did not parse: {exc.msg}.")
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    raises = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Raise))
    names = ", ".join(functions) if functions else "no functions"
    return _note(
        "ast",
        "parsed",
        (
            f"Parsed {_count(len(functions), 'function', 'functions')} ({names}) "
            f"and {_count(raises, 'raise statement', 'raise statements')}."
        ),
    )


def _tdd(red: dict, tests: dict) -> dict:
    if red.get("status") == "failed" and tests.get("status") == "passed":
        return _note(
            "tdd",
            "red_then_green",
            (
                "The tests failed before change.py existed. "
                f"After the change, {tests.get('summary', 'the tests passed')}"
            ),
        )
    if tests.get("status") == "passed":
        return _note("tdd", "green", f"The tests passed. {tests.get('summary', '')}".strip())
    return _note("tdd", "red", f"The tests are still failing. {tests.get('summary', '')}".strip())


def _healing(healed: bool, tests: dict) -> dict:
    if healed and tests.get("status") == "passed":
        return _note(
            "healing",
            "healed",
            "The first implementation omitted the empty-value check. An AST edit inserted the raise, and the tests then passed.",
        )
    if tests.get("status") == "passed":
        return _note("healing", "stable", "The first implementation passed its tests. No repair was required.")
    return _note("healing", "unresolved", "A repair did not make the tests pass.")


def _review_note(votes: dict[str, str]) -> dict:
    tally = f"Tester {votes['tester']}, Spec {votes['spec']}, Policy {votes['policy']}"
    if all(vote == "yes" for vote in votes.values()):
        return _note("review", "accept", f"The reviewers agree. {tally}.")
    return _note("review", "dissent", f"A reviewer voted no. {tally}.")


def _policy(module: str, tests_source: str) -> dict:
    banned = _banned_calls(module)
    functions = _functions(module)
    called = set()
    for calls in _calls(tests_source).values():
        called.update(calls)
    missing = [name for name in functions if name not in called]
    problems = []
    if banned:
        problems.append("Banned calls: " + ", ".join(banned) + ".")
    if missing:
        problems.append("Functions without a test: " + ", ".join(missing) + ".")
    if problems:
        return _note("policy", "blocked", " ".join(problems))
    return _note(
        "policy",
        "enforced",
        "No banned call is present, and every function has a test.",
    )


def _debug(red: dict, tests: dict) -> dict:
    if tests.get("status") == "passed":
        return _note(
            "debug",
            "clear",
            "The final tests passed. The earlier failure was the missing change.py module.",
        )
    cause = _root_cause(str(tests.get("output") or red.get("output") or ""))
    return _note("debug", "diagnosed", cause)


def _governance(pull: dict, policy: dict) -> dict:
    head = pull.get("head") or "the delivery branch"
    if policy["decision"] == "blocked":
        return _note(
            "governance",
            "hold",
            f"A person has to clear the policy block before {head} can open.",
        )
    if pull.get("ready"):
        return _note(
            "governance",
            "hold",
            f"{head} is ready. A person has to open it.",
        )
    return _note(
        "governance",
        "hold",
        f"{head} stays held until a person accepts the change.",
    )


def _votes(tests: dict, review: dict, policy: dict) -> dict[str, str]:
    tester = "yes" if tests.get("status") == "passed" else "no"
    spec = "yes" if review.get("decision") in {"approve", "recorded"} else "no"
    policy_vote = "yes" if policy["decision"] == "enforced" else "no"
    return {"tester": tester, "spec": spec, "policy": policy_vote}


def _file(files: list[dict], path: str) -> str:
    for item in files:
        if item.get("path") == path:
            return str(item.get("content") or "")
    return ""


def _parse(source: str) -> ast.AST | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _functions(source: str) -> list[str]:
    tree = _parse(source)
    if tree is None:
        return []
    return [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]


def _imports(source: str) -> list[str]:
    tree = _parse(source)
    if tree is None:
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _calls(source: str) -> dict[str, list[str]]:
    tree = _parse(source)
    if tree is None:
        return {}
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                calls.append(child.func.id)
        found[node.name] = calls
    return found


def _banned_calls(source: str) -> list[str]:
    tree = _parse(source)
    if tree is None:
        return ["unparseable module"]
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in {"eval", "exec"}:
            found.append(func.id)
        elif isinstance(func, ast.Attribute) and func.attr == "system":
            found.append("os.system")
    return found


def _root_cause(output: str) -> str:
    for line in reversed(output.splitlines()):
        text = line.strip()
        if not text:
            continue
        if text.startswith("E ") or "Error" in text or text.startswith("Failed"):
            return f"Root cause: {text[:180]}"
    if "No module named" in output or "ModuleNotFoundError" in output:
        return "Root cause: change.py was not available to the tests."
    return "Root cause: the test output did not name an exception."


def _count(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _note(component: str, decision: str, reason: str) -> dict:
    return {"component": component, "decision": decision, "reason": reason}
