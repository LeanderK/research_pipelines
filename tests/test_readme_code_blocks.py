"""Execute and validate all python fenced code blocks in README.md.

This test extracts all ```python fenced blocks and executes them sequentially
in a shared namespace. The README is the canonical documentation; keeping its
examples runnable helps catch bitrot.
"""
import re
from pathlib import Path

from examples import readme_helpers as readme_helpers
from research_pipelines.backends.manager import get_backend
import research_pipelines.query as query
from research_pipelines.decorators import dataset, evaluation, model, traced


def find_readme():
    cur = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = cur / "README.md"
        if candidate.exists():
            return candidate
        cur = cur.parent
    raise FileNotFoundError("README.md not found in parent hierarchy")

README = find_readme()


def extract_python_blocks(readme_text: str):
    pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    return [m.group(1).strip() for m in pattern.finditer(readme_text)]


def test_readme_python_blocks_execute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = README.read_text(encoding="utf-8")
    blocks = extract_python_blocks(text)
    assert blocks, "No python blocks found in README.md"

    # Seed the backend with a tiny traced run so the rebuild example has real
    # traced objects to load from.
    readme_helpers.setup_readme_backend(str(tmp_path / "readme_demo"))
    seeded_train_set = readme_helpers.load_data(path="/data/train.csv", split="train")
    seeded_model = readme_helpers.build_model(architecture="bert")
    readme_helpers.evaluate(seeded_model, seeded_train_set)

    # Execute all blocks in one shared namespace so they can inherit local
    # variables and state in the same way a reader would run them top to bottom.
    shared_ns = {
        "__name__": "__main__",
        "dataset": dataset,
        "evaluation": evaluation,
        "model": model,
        "traced": traced,
        "query": query,
        "get_backend": get_backend,
        "load_data": readme_helpers.load_data,
        "build_model": readme_helpers.build_model,
        "evaluate": readme_helpers.evaluate,
        "state_dict": readme_helpers.state_dict,
    }
    for i, block in enumerate(blocks, start=1):
        if i in (2, 3, 4):
            get_backend().set_recording_enabled(False)
        elif i == 5:
            get_backend().set_recording_enabled(True)

        if i > 1:
            # Prefer the importable helper versions once the definition-only
            # example block has run.
            shared_ns["load_data"] = readme_helpers.load_data
            shared_ns["build_model"] = readme_helpers.build_model
            shared_ns["evaluate"] = readme_helpers.evaluate
            shared_ns["state_dict"] = readme_helpers.state_dict
        if i == 5:
            shared_ns["model"] = seeded_model
            shared_ns["train_set"] = seeded_train_set
        elif i > 5:
            shared_ns["dataset"] = dataset
            shared_ns["evaluation"] = evaluation
            shared_ns["model"] = model
            shared_ns["traced"] = traced

        try:
            exec(block, shared_ns)
        except Exception as e:
            # Show which block failed and a short preview
            preview = block[:400].replace("\n", " ")
            raise AssertionError(f"README python block #{i} raised {e!r}: {preview}") from e
