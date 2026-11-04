"""Read a software repository before any code is generated.

The scan walks the tree, names languages and packages, parses Python symbols,
links internal imports, assigns architectural layers, and selects files that
match a requirement. It returns an architecture report and does not write code.
"""

from __future__ import annotations

import ast
import os
import re
import tomllib
from pathlib import Path

_SKIP = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "data",
    ".chroma",
}
_BINARY = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".sqlite", ".sqlite3", ".db", ".bin", ".pyc", ".zip", ".gz"}
_EXT = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".html": "HTML",
    ".css": "CSS",
    ".md": "Markdown",
    ".json": "JSON",
    ".toml": "TOML",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".txt": "Text",
}
_LAYER_PARTS = {
    "tests": "tests",
    "test": "tests",
    "api": "api",
    "web": "web",
    "static": "web",
    "templates": "web",
    "research": "research",
    "engineering": "engineering",
    "ingestion": "ingestion",
    "retrieval": "retrieval",
    "generation": "generation",
    "observability": "observability",
    "evaluation": "evaluation",
}
_LAYER_ROLE = {
    "api": "HTTP routes",
    "web": "Browser interface",
    "research": "Research and evidence",
    "engineering": "Delivery and evaluation",
    "ingestion": "Parsing and indexing",
    "retrieval": "Search",
    "generation": "Answers",
    "observability": "Logs and metrics",
    "evaluation": "Benchmarks",
    "tests": "Automated checks",
    "application": "Application modules",
}
_FRAMEWORKS = {
    "fastapi": "FastAPI",
    "uvicorn": "Uvicorn",
    "pydantic": "Pydantic",
    "pydantic_settings": "Pydantic",
    "pytest": "pytest",
    "chromadb": "Chroma",
    "openai": "OpenAI",
    "fitz": "PyMuPDF",
    "pymupdf": "PyMuPDF",
}
_STOP = {
    "a", "an", "the", "to", "for", "of", "and", "or", "in", "on", "by", "with",
    "that", "this", "is", "are", "be", "from", "into", "as", "at", "it", "its",
    "their", "when", "where", "how", "what", "which", "who", "add", "make",
    "find", "show", "use", "using", "should", "must", "can", "will", "than",
    "then", "also", "only", "new", "existing", "implementation", "code",
    "function", "method", "class", "file", "files", "module",
    "name", "names", "person", "value", "input", "data", "text", "item",
    "result", "type", "get", "set", "run", "empty", "include", "reject",
    "error", "test", "tests",
}
_INTERFACE = re.compile(r"\binterface\s+([A-Za-z_]\w*)")
_DEP = re.compile(r"^[A-Za-z0-9_.-]+")
_CACHE: dict[str, tuple[tuple, dict]] = {}


def inspect_repository(root: Path | None = None, requirement: str = "") -> dict:
    """Scan ``root`` and report the architecture. The default root is this project."""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    if not base.is_dir():
        raise ValueError("A repository directory is required.")
    index = _index(base.resolve())
    return _report(index, str(requirement or "").strip())


def _index(root: Path) -> dict:
    signature = _signature(root)
    cached = _CACHE.get(str(root))
    if cached and cached[0] == signature:
        return cached[1]
    built = _build(root)
    _CACHE[str(root)] = (signature, built)
    return built


def _signature(root: Path) -> tuple:
    count = 0
    size = 0
    newest = 0
    for path in _walk(root):
        count += 1
        stat = path.stat()
        size += stat.st_size
        newest = max(newest, stat.st_mtime_ns)
    return (count, size, newest)


def _build(root: Path) -> dict:
    files = list(_walk(root))
    modules: dict[str, dict] = {}
    languages: dict[str, int] = {}
    directories: dict[str, int] = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        languages[_language(path)] = languages.get(_language(path), 0) + 1
        parts = relative.split("/")
        if len(parts) > 1:
            directories[parts[0]] = directories.get(parts[0], 0) + 1
        if path.suffix.lower() == ".py":
            record = _parse_python(path, relative)
            if record:
                modules[record["module"]] = record
        elif path.suffix.lower() in {".js", ".ts"}:
            record = _parse_script(path, relative)
            if record:
                modules[record["module"]] = record
    _link(modules)
    return {
        "root": root.as_posix(),
        "files": [path.relative_to(root).as_posix() for path in files],
        "languages": languages,
        "directories": [
            {"path": name, "files": count}
            for name, count in sorted(directories.items())
        ],
        "packages": _packages(root),
        "modules": modules,
    }


def _walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames if name not in _SKIP and not name.endswith(".egg-info")
        )
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() in _BINARY:
                continue
            try:
                if path.stat().st_size > 300_000:
                    continue
            except OSError:
                continue
            yield path


def _language(path: Path) -> str:
    if path.name.lower() == "dockerfile":
        return "Docker"
    return _EXT.get(path.suffix.lower(), "Other")


def _parse_python(path: Path, relative: str) -> dict | None:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeError, SyntaxError):
        return None
    classes = []
    interfaces = []
    functions = []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if _is_interface(node):
                interfaces.append(node.name)
            else:
                classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
    doc = ast.get_docstring(tree) or ""
    return {
        "path": relative,
        "module": _module_name(relative),
        "language": "Python",
        "classes": classes,
        "interfaces": interfaces,
        "functions": functions,
        "models": _model_count(tree),
        "router": _assigns_router(tree),
        "doc": doc.splitlines()[0] if doc else "",
        "import_nodes": imports,
        "imports": [],
        "external": [],
        "relative_imports": 0,
    }


def _parse_script(path: Path, relative: str) -> dict | None:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    return {
        "path": relative,
        "module": relative,
        "language": _language(path),
        "classes": [],
        "interfaces": _INTERFACE.findall(source),
        "functions": [],
        "models": 0,
        "router": False,
        "doc": "",
        "import_nodes": [],
        "imports": [],
        "external": [],
        "relative_imports": 0,
    }


def _module_name(relative: str) -> str:
    parts = list(Path(relative).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_interface(node: ast.ClassDef) -> bool:
    bases = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            bases.append(base.attr)
    return bool({"Protocol", "ABC", "Interface"} & set(bases)) or node.name.endswith("Protocol")


def _model_count(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            name = base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else ""
            if name == "BaseModel":
                count += 1
    return count


def _assigns_router(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name == "APIRouter":
            return True
    return False


def _link(modules: dict[str, dict]) -> None:
    names = set(modules)
    for record in modules.values():
        targets = []
        external = []
        relative = 0
        for node in record["import_nodes"]:
            found, outside, is_relative = _targets(record["module"], node, names)
            targets.extend(found)
            external.extend(outside)
            relative += is_relative
        record["imports"] = sorted(set(targets))
        record["external"] = sorted(set(external))
        record["relative_imports"] = relative
        record.pop("import_nodes", None)


def _targets(current: str, node: ast.AST, modules: set[str]) -> tuple[list[str], list[str], int]:
    found: list[str] = []
    external: list[str] = []
    relative = 0
    if isinstance(node, ast.Import):
        for alias in node.names:
            bound = _bind(modules, alias.name)
            if bound and bound != current:
                found.append(bound)
            elif not bound:
                external.append(alias.name.split(".")[0])
        return found, external, 0
    if not isinstance(node, ast.ImportFrom):
        return found, external, 0
    if node.level:
        relative = 1
        target = _relative_module(current, node.level, node.module)
    else:
        target = node.module or ""
    bound = _bind(modules, target) if target else None
    if bound and bound != current:
        found.append(bound)
    elif target and not bound and not node.level:
        external.append(target.split(".")[0])
    for alias in node.names:
        candidate = f"{target}.{alias.name}" if target else alias.name
        bound_name = _bind(modules, candidate)
        if bound_name and bound_name != current:
            found.append(bound_name)
    return found, external, relative


def _relative_module(current: str, level: int, module: str | None) -> str:
    parts = current.split(".")
    if len(parts) < level:
        return ""
    base = parts[:-level]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def _bind(modules: set[str], target: str) -> str | None:
    if target in modules:
        return target
    return None


def _packages(root: Path) -> list[dict]:
    rows = []
    path = root / "pyproject.toml"
    if path.is_file():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            data = {}
        project = data.get("project") or {}
        for spec in project.get("dependencies") or []:
            rows.append({"name": _dep_name(spec), "source": "pyproject.toml"})
        for group, specs in (project.get("optional-dependencies") or {}).items():
            for spec in specs:
                rows.append({"name": _dep_name(spec), "source": f"pyproject.toml[{group}]"})
    requirements = root / "requirements.txt"
    if requirements.is_file():
        try:
            lines = requirements.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            lines = []
        for line in lines:
            name = _dep_name(line.strip())
            if name and not name.startswith("#"):
                rows.append({"name": name, "source": "requirements.txt"})
    return rows


def _dep_name(spec: str) -> str:
    match = _DEP.match(spec.strip())
    return match.group(0) if match else ""


def _report(index: dict, requirement: str) -> dict:
    modules = list(index["modules"].values())
    languages = sorted(index["languages"].items(), key=lambda item: (-item[1], item[0]))
    packages = sorted({row["name"] for row in index["packages"] if row["name"]})
    frameworks = _frameworks(index, modules)
    edges = _edges(modules)
    layers = _layers(index["files"], modules)
    relevant = _relevant(modules, requirement)
    symbols = {
        "classes": sum(len(item["classes"]) for item in modules),
        "functions": sum(len(item["functions"]) for item in modules),
        "interfaces": sum(len(item["interfaces"]) for item in modules),
        "imports": sum(len(item["imports"]) + len(item["external"]) for item in modules),
    }
    patterns = _patterns(modules)
    notes = [
        _structure_note(index),
        _stack_note(languages, frameworks, packages),
        _symbol_note(symbols),
        _dependency_note(edges),
        _layer_note(layers),
        _file_note(requirement, relevant),
        _pattern_note(patterns),
        _report_note(index, languages, symbols, edges, layers),
    ]
    summary = _summary(index, languages, layers, edges, requirement, relevant)
    return {
        "summary": summary,
        "languages": [name for name, _count in languages],
        "frameworks": frameworks,
        "packages": packages,
        "directories": index["directories"],
        "layers": [
            {"name": name, "role": _LAYER_ROLE.get(name, "Modules"), "files": count}
            for name, count in layers
        ],
        "relevant": relevant,
        "dependencies": [{"source": source, "target": target} for source, target in edges],
        "dependency_count": len(edges),
        "symbols": symbols,
        "patterns": patterns,
        "notes": notes,
    }


def _frameworks(index: dict, modules: list[dict]) -> list[str]:
    names = {_normalize(row["name"]) for row in index["packages"]}
    for record in modules:
        names.update(_normalize(item) for item in record["external"])
    found = {_FRAMEWORKS[name] for name in names if name in _FRAMEWORKS}
    return sorted(found)


def _normalize(name: str) -> str:
    return name.lower().replace("-", "_")


def _edges(modules: list[dict]) -> list[tuple[str, str]]:
    rows = []
    for record in modules:
        for target in record["imports"]:
            rows.append((record["module"], target))
    return sorted(set(rows))


def _layers(files: list[str], modules: list[dict]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    parsed = {record["path"] for record in modules}
    for relative in files:
        if relative not in parsed and not relative.endswith((".py", ".js", ".ts", ".html", ".css")):
            continue
        layer = _layer_name(relative)
        counts[layer] = counts.get(layer, 0) + 1
    return sorted(counts.items())


def _layer_name(relative: str) -> str:
    parts = relative.split("/")[:-1]
    for part in reversed(parts):
        if part in _LAYER_PARTS:
            return _LAYER_PARTS[part]
    return "application"


def _relevant(modules: list[dict], requirement: str) -> list[dict]:
    tokens = _tokens(requirement)
    if not tokens:
        return []
    ranked = []
    for record in modules:
        score, matched = _score(record, tokens)
        if score <= 0:
            continue
        ranked.append(
            {
                "path": record["path"],
                "score": score,
                "reason": "Matches " + ", ".join(matched) + ".",
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["path"]))
    if not ranked:
        return []
    best = ranked[0]["score"]
    floor = max(2, best - 2)
    return [item for item in ranked if item["score"] >= floor][:5]


def _score(record: dict, tokens: list[str]) -> tuple[int, list[str]]:
    path_tokens = _tokens(record["path"].replace("/", " ").replace("_", " ").replace(".", " "))
    name_tokens = []
    for name in record["classes"] + record["interfaces"] + record["functions"]:
        if name.startswith("_"):
            continue
        name_tokens.extend(_tokens(name.replace("_", " ")))
    doc_tokens = _tokens(record["doc"])
    score = 0
    matched = []
    for token in tokens:
        if _contains(token, path_tokens):
            score += 3
            matched.append(token)
        elif _contains(token, name_tokens):
            score += 2
            matched.append(token)
        elif _contains(token, doc_tokens):
            score += 1
            matched.append(token)
    return score, matched


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text.casefold())
    found = []
    seen = set()
    for word in words:
        word = _stem_word(word)
        if word in _STOP or len(word) <= 2 or word in seen:
            continue
        seen.add(word)
        found.append(word)
    return found


def _stem_word(word: str) -> str:
    if word.endswith("ing") and len(word) - 3 >= 4:
        return word[:-3]
    if word.endswith("ed") and len(word) - 2 >= 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) - 1 >= 4:
        return word[:-1]
    return word


def _contains(token: str, haystack: list[str]) -> bool:
    for item in haystack:
        if token == item or token == _stem_word(item) or _stem_word(token) == item:
            return True
    return False


def _patterns(modules: list[dict]) -> list[str]:
    routers = [item for item in modules if item["router"]]
    models = sum(item["models"] for item in modules)
    tests = sum(1 for item in modules for name in item["functions"] if name.startswith("test_"))
    relative = sum(item["relative_imports"] for item in modules)
    rows = []
    if routers:
        rows.append(f"{_count(len(routers), 'module assigns', 'modules assign')} APIRouter.")
    if models:
        rows.append(f"{_count(models, 'data model subclasses', 'data models subclass')} BaseModel.")
    if tests:
        rows.append(f"{_count(tests, 'check is a function', 'checks are functions')} named test_*.")
    if relative:
        rows.append(f"{_count(relative, 'internal import is', 'internal imports are')} relative.")
    elif any(item["imports"] for item in modules):
        rows.append("Internal dependencies use absolute imports.")
    return rows


def _structure_note(index: dict) -> dict:
    directories = index["directories"]
    scanned = _count(len(index["files"]), "file", "files")
    if not directories:
        return _note("structure", "scanned", f"Scanned {scanned}.")
    listed = ", ".join(
        f"{item['path']} ({_count(item['files'], 'file', 'files')})" for item in directories
    )
    return _note(
        "structure",
        "scanned",
        (
            f"Scanned {scanned} across "
            f"{_count(len(directories), 'top-level directory', 'top-level directories')}: {listed}."
        ),
    )


def _stack_note(languages: list[tuple[str, int]], frameworks: list[str], packages: list[str]) -> dict:
    if not languages:
        return _note("stack", "empty", "No source files were recognized.")
    language_text = ", ".join(f"{name} ({count})" for name, count in languages)
    framework_text = ", ".join(frameworks) if frameworks else "none recognized"
    if packages:
        shown = packages[:12]
        extra = f" and {len(packages) - len(shown)} more" if len(packages) > len(shown) else ""
        package_text = ", ".join(shown) + extra
    else:
        package_text = "none declared"
    return _note(
        "stack",
        "identified",
        f"Languages: {language_text}. Frameworks: {framework_text}. Packages: {package_text}.",
    )


def _symbol_note(symbols: dict) -> dict:
    if not any(symbols.values()):
        return _note("symbols", "empty", "No classes, functions, interfaces, or imports were parsed.")
    return _note(
        "symbols",
        "parsed",
        (
            f"Parsed {_count(symbols['classes'], 'class', 'classes')}, "
            f"{_count(symbols['functions'], 'function', 'functions')}, "
            f"{_count(symbols['interfaces'], 'interface', 'interfaces')}, "
            f"and {_count(symbols['imports'], 'import', 'imports')}."
        ),
    )


def _dependency_note(edges: list[tuple[str, str]]) -> dict:
    if not edges:
        return _note("dependencies", "isolated", "No internal import connects these modules.")
    incoming: dict[str, set[str]] = {}
    for source, target in edges:
        incoming.setdefault(target, set()).add(source)
    target, sources = sorted(incoming.items(), key=lambda item: (-len(item[1]), item[0]))[0]
    return _note(
        "dependencies",
        "linked",
        (
            f"{_count(len(edges), 'internal import edge', 'internal import edges')}. "
            f"{target} is imported by {_count(len(sources), 'module', 'modules')}."
        ),
    )


def _layer_note(layers: list[tuple[str, int]]) -> dict:
    if len(layers) < 2:
        if not layers:
            return _note("layers", "flat", "No architectural layer was found.")
        name, count = layers[0]
        return _note(
            "layers",
            "flat",
            f"The files sit in the {name} layer ({_count(count, 'source file', 'source files')}).",
        )
    listed = ", ".join(
        f"{name} ({_count(count, 'source file', 'source files')})" for name, count in layers
    )
    return _note("layers", "layered", f"{_count(len(layers), 'layer', 'layers')}: {listed}.")


def _file_note(requirement: str, relevant: list[dict]) -> dict:
    if not requirement:
        return _note(
            "files",
            "open",
            "No requirement was given. The report describes the repository without choosing files.",
        )
    if not relevant:
        return _note("files", "unmatched", "No file name or symbol matches the requirement.")
    names = ", ".join(item["path"] for item in relevant)
    return _note("files", "matched", f"The requirement matches {names}.")


def _pattern_note(patterns: list[str]) -> dict:
    if not patterns:
        return _note("patterns", "none", "No repeated implementation pattern was recognized.")
    return _note("patterns", "found", " ".join(patterns))


def _report_note(index: dict, languages, symbols: dict, edges, layers) -> dict:
    return _note(
        "report",
        "written",
        (
            f"The architecture report covers {_count(len(index['files']), 'file', 'files')}, "
            f"{_count(len(languages), 'language', 'languages')}, "
            f"{_count(symbols['classes'], 'class', 'classes')}, "
            f"{_count(symbols['functions'], 'function', 'functions')}, "
            f"{_count(len(edges), 'dependency', 'dependencies')}, "
            f"and {_count(len(layers), 'layer', 'layers')}."
        ),
    )


def _summary(index: dict, languages, layers, edges, requirement: str, relevant: list[dict]) -> str:
    names = ", ".join(name for name, _count in languages[:3]) or "no recognized language"
    text = (
        f"Scanned {_count(len(index['files']), 'file', 'files')} in {names}. "
        f"{_count(len(layers), 'architectural layer', 'architectural layers')} and "
        f"{_count(len(edges), 'internal dependency', 'internal dependencies')}."
    )
    if not requirement:
        return text + " No requirement was applied."
    if relevant:
        return text + f" {_count(len(relevant), 'file matches', 'files match')} the requirement."
    return text + " No file matches the requirement."


def _count(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _note(component: str, decision: str, reason: str) -> dict:
    return {"component": component, "decision": decision, "reason": reason}
