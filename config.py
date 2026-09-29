"""Central configuration.

Every setting can be overridden with an environment variable of the same name
prefixed with ROUTEWRIGHT_, so the same build runs unchanged on a laptop, in a
container or behind a load balancer.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name, default):
    return os.environ.get("ROUTEWRIGHT_" + name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", ROOT / "data")))
    artifact_dir: Path = field(default_factory=lambda: Path(_env("ARTIFACT_DIR", ROOT / "artifacts")))
    report_dir: Path = field(default_factory=lambda: Path(_env("REPORT_DIR", ROOT / "reports")))
    docs_dir: Path = field(default_factory=lambda: ROOT / "docs")

    dataset_name: str = "routewright_incidents.csv"
    ground_truth_name: str = "ground_truth_efficacy.json"

    rows: int = int(_env("ROWS", 16000))
    seed: int = int(_env("SEED", 42))

    embedding_dims: int = int(_env("EMBEDDING_DIMS", 160))
    bm25_k1: float = float(_env("BM25_K1", 1.4))
    bm25_b: float = float(_env("BM25_B", 0.72))
    rrf_k: int = int(_env("RRF_K", 60))
    candidate_pool: int = int(_env("CANDIDATE_POOL", 400))
    evidence_min_cases: int = int(_env("EVIDENCE_MIN_CASES", 200))
    context_cases: int = int(_env("CONTEXT_CASES", 8))
    mmr_lambda: float = float(_env("MMR_LAMBDA", 0.72))

    llm_provider: str = _env("LLM_PROVIDER", "extractive")
    llm_model: str = _env("MODEL", "")
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    llm_max_tokens: int = int(_env("MAX_TOKENS", 1200))
    llm_timeout: float = float(_env("LLM_TIMEOUT", 60))

    @property
    def dataset_path(self):
        return self.data_dir / self.dataset_name

    @property
    def ground_truth_path(self):
        return self.data_dir / self.ground_truth_name


settings = Settings()
