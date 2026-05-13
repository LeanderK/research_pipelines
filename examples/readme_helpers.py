"""Small helpers used to keep README code blocks runnable.

The README should still showcase the real library calls; this module only
provides tiny support objects and a clean backend setup so the examples can be
executed in tests without external dependencies.
"""

from research_pipelines.backends.manager import get_backend, set_backend
from research_pipelines.backends.pickle_backend import PickleBackend
from research_pipelines.core import clear_traced_registry
from research_pipelines.decorators import dataset, evaluation, model


def setup_readme_backend(trace_dir: str = "data/readme_demo") -> None:
	"""Configure a clean PickleBackend for README execution."""
	set_backend(PickleBackend(directory=trace_dir, recording_enabled=True))
	clear_traced_registry()
	get_backend().clear()


class SimpleDataset:
	def __init__(self, split: str, size: int):
		self.split = split
		self.size = size


class SimpleModel:
	def __init__(self, architecture: str, weight: int = 1):
		self.architecture = architecture
		self.weight = weight

	def load_state_dict(self, state_dict):
		self.weight = int(state_dict.get("weight", self.weight))


@dataset()
def load_data(path: str, split: str):
	return SimpleDataset(split=split, size=3)


@model()
def build_model(architecture: str):
	return SimpleModel(architecture=architecture)


@evaluation()
def evaluate(model_obj, test_set, full_evaluation=False):
	return {"score": 1.0 if full_evaluation else 0.0}


state_dict = {"weight": 7}
