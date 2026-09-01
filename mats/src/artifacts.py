"""HF Hub push + wandb reference logging (plan sections 0.2, 0.4, 0.5).

WHAT THIS IS FOR
----------------
Model weights go to the **HF Hub** (private). wandb gets a **reference** to the
Hub commit, never the bytes (plan section 0.2: "Do not put model weights in
wandb Artifacts"). The reference pins the **commit SHA**, not ``main`` — a
reference to a branch floats, and plan section 11 lists a ``main``-pointing
artifact as an abort condition, because it destroys the provenance the artifact
existed to provide.

NAMING (plan section 0.4, followed exactly)
-------------------------------------------
    {HF_ORG}/gcvl-sdf-{parent}-{direction}-{dose_pct:03d}
        e.g. will-org/gcvl-sdf-M_RL_late-GS_DA-050
    {HF_ORG}/gcvl-dapo-{step_pct:03d}
        e.g. will-org/gcvl-dapo-100

Use :func:`sdf_repo_id` / :func:`dapo_repo_id` rather than formatting by hand.
The eval sweep (section 8) reconstructs these paths mechanically, so a
hand-written name that drifts produces a plausible-looking wrong number rather
than an error.

CREDENTIALS
-----------
``HF_TOKEN`` and ``WANDB_API_KEY`` come from the process environment, with
``$EXP_ROOT/.env`` loaded first via python-dotenv (plan section 0.8). If a
credential needed for a real push is missing we raise with an actionable
message. **A push is never silently skipped** — the only way to not push is to
ask for it, with ``dry_run=True`` / ``--dry-run``.

DRY RUN
-------
``dry_run=True`` resolves and validates everything (paths, names, metadata,
which credentials would be needed) and prints the plan, touching neither the
Hub nor wandb. It needs no credentials, so the pipeline can be exercised
off-GPU and off-network.

CLI::

    python -m src.artifacts --local-dir $EXP_ROOT/ckpt/sdf_M_base_GS_DA/checkpoint-400 \
        --parent M_base --direction GS_DA --dose-pct 50 --dry-run
    python -m src.artifacts --local-dir ... --repo-id will-org/gcvl-dapo-100
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import exp_root, logs_dir

LOG = logging.getLogger("artifacts")

#: HF repo names may contain only these characters, and must be <= 96 chars.
_REPO_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
#: A git commit SHA as returned by the Hub. Anything else is not provenance.
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

_ENV_HELP = {
    "HF_TOKEN": (
        "a HF write token. Create one at https://huggingface.co/settings/tokens "
        "(role: write), then add `HF_TOKEN=hf_...` to $EXP_ROOT/.env on the box, "
        "or run `huggingface-cli login`."
    ),
    "WANDB_API_KEY": (
        "your wandb API key from https://wandb.ai/authorize. Add "
        "`WANDB_API_KEY=...` to $EXP_ROOT/.env, or run `wandb login`."
    ),
    "HF_ORG": (
        "the HF user or org that owns the private repos, e.g. `HF_ORG=will-org` "
        "in $EXP_ROOT/.env. Plan section 0.4 names every artifact "
        "`{HF_ORG}/gcvl-...`."
    ),
}


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #
def load_env(env_path: Path | None = None, *, override: bool = False) -> Path | None:
    """Load ``$EXP_ROOT/.env`` into ``os.environ`` (plan section 0.8).

    Returns the path loaded, or None if there was no .env file. Missing
    python-dotenv is not fatal — real environment variables still work — but it
    is reported, because the usual cause of "credential not found" is that the
    .env was never read.
    """
    path = Path(env_path) if env_path else exp_root() / ".env"
    if not path.exists():
        LOG.debug("no .env at %s (using process environment only)", path)
        return None
    try:
        from dotenv import load_dotenv
    except ImportError:
        LOG.warning(
            "python-dotenv is not installed, so %s was NOT loaded. "
            "`uv pip install python-dotenv`, or export the variables by hand.", path
        )
        return None
    load_dotenv(path, override=override)
    LOG.info("loaded credentials from %s", path)
    return path


def require_env(names: Sequence[str], *, env_path: Path | None = None) -> dict[str, str]:
    """Return the named environment variables, raising an actionable error.

    Never returns partially: either every requested name is present and
    non-empty, or we raise. Values are never logged.
    """
    load_env(env_path)
    missing = [n for n in names if not os.environ.get(n, "").strip()]
    if missing:
        lines = [
            f"missing required credential(s): {', '.join(missing)}",
            f"Looked in the process environment and {exp_root() / '.env'}.",
            "",
        ]
        lines += [f"  {n}: {_ENV_HELP.get(n, 'set this variable')}" for n in missing]
        lines += [
            "",
            "Nothing was pushed. Re-run with --dry-run to exercise the pipeline "
            "without credentials.",
        ]
        raise RuntimeError("\n".join(lines))
    return {n: os.environ[n] for n in names}


def hf_org(*, env_path: Path | None = None) -> str:
    """The HF namespace from ``$HF_ORG`` (plan section 0.4)."""
    return require_env(["HF_ORG"], env_path=env_path)["HF_ORG"].strip().rstrip("/")


# --------------------------------------------------------------------------- #
# naming — plan section 0.4
# --------------------------------------------------------------------------- #
def sdf_repo_name(parent: str, direction: str, dose_pct: int) -> str:
    """``gcvl-sdf-{parent}-{direction}-{dose_pct:03d}`` (no org prefix)."""
    if not 0 <= int(dose_pct) <= 100:
        raise ValueError(f"dose_pct must be a percentage in [0, 100], got {dose_pct!r}")
    return f"gcvl-sdf-{parent}-{direction}-{int(dose_pct):03d}"


def dapo_repo_name(step_pct: int) -> str:
    """``gcvl-dapo-{step_pct:03d}`` (no org prefix)."""
    if not 0 <= int(step_pct) <= 100:
        raise ValueError(f"step_pct must be a percentage in [0, 100], got {step_pct!r}")
    return f"gcvl-dapo-{int(step_pct):03d}"


def sdf_repo_id(parent: str, direction: str, dose_pct: int, org: str | None = None) -> str:
    """Full ``{HF_ORG}/gcvl-sdf-{parent}-{direction}-{pct:03d}``."""
    return qualify(sdf_repo_name(parent, direction, dose_pct), org)


def dapo_repo_id(step_pct: int, org: str | None = None) -> str:
    """Full ``{HF_ORG}/gcvl-dapo-{pct:03d}``."""
    return qualify(dapo_repo_name(step_pct), org)


def qualify(name: str, org: str | None = None) -> str:
    """Prefix a bare repo name with the org, and validate the result."""
    if "/" in name:
        repo_id = name
    else:
        repo_id = f"{org or hf_org()}/{name}"
    validate_repo_id(repo_id)
    return repo_id


def validate_repo_id(repo_id: str) -> None:
    """Fail loudly on a repo id the Hub would reject or that breaks section 8."""
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"repo_id must be '<org>/<name>', got {repo_id!r}. "
            "Plan section 0.4: {HF_ORG}/gcvl-sdf-..."
        )
    for part in parts:
        if not _REPO_NAME_RE.match(part):
            raise ValueError(
                f"{part!r} is not a valid HF namespace/name component "
                "(allowed: alphanumerics . _ -, max 96 chars, must start alphanumeric)"
            )


# --------------------------------------------------------------------------- #
# push + register
# --------------------------------------------------------------------------- #
@dataclass
class PushResult:
    """Outcome of one push. ``sha`` is the pinned commit, never a branch."""

    repo_id: str
    sha: str | None
    url: str | None
    local_dir: str
    n_files: int
    n_bytes: int
    pushed: bool
    registered: bool
    dry_run: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Trainer checkpoint files that are resume state, not model weights. Uploading
#: them wastes Hub bandwidth and quota, and nothing downstream reads them: vLLM
#: and PEFT load the adapter/model files only (plan section 0.2).
DEFAULT_IGNORE = ("optimizer.pt", "optimizer.bin", "scheduler.pt", "rng_state*.pth",
                  "rng_state*.pt", "*.distcp", "global_step*/*")


def _inventory(local_dir: Path, ignore: Sequence[str] | None = None) -> tuple[int, int, list[str]]:
    """Files that would actually be uploaded, after ``ignore`` patterns."""
    from fnmatch import fnmatch

    rel = [str(p.relative_to(local_dir)) for p in sorted(local_dir.rglob("*")) if p.is_file()]
    if ignore:
        rel = [r for r in rel if not any(fnmatch(r, pat) for pat in ignore)]
    n_bytes = sum((local_dir / r).stat().st_size for r in rel)
    return len(rel), n_bytes, rel


def _check_adapter_dir(local_dir: Path, names: Sequence[str]) -> None:
    """Warn if the directory does not look like something PEFT/vLLM can load.

    Not fatal — merged full models and verl exports look different — but a
    silently-empty or wrong directory would upload fine and fail at eval time,
    hours later, on the GPU box.
    """
    lower = {n.lower() for n in names}
    adapter = {"adapter_config.json"} & lower
    merged = {"config.json"} & lower
    if not adapter and not merged:
        LOG.warning(
            "!! %s contains neither adapter_config.json nor config.json. "
            "vLLM/PEFT will not be able to load this by repo id (plan section 0.2). "
            "Are you pointing at a checkpoint-N directory?", local_dir
        )
    if adapter and not any(
        n.startswith("adapter_model.") for n in lower
    ):
        LOG.warning("!! %s has adapter_config.json but no adapter_model.* weights", local_dir)


def _commit_sha(commit: Any) -> str:
    """Extract the commit SHA from whatever ``upload_folder`` returned.

    huggingface_hub 1.x returns a ``CommitInfo`` with ``.oid``; older versions
    returned the commit URL as a plain string. Handle both, and refuse anything
    that is not a hex SHA — a reference to ``main`` is the plan section 11
    abort condition.
    """
    sha = getattr(commit, "oid", None)
    if not sha and isinstance(commit, str):
        m = re.search(r"/commit/([0-9a-f]{7,40})", commit)
        sha = m.group(1) if m else None
    if not sha:
        raise RuntimeError(
            f"could not read a commit SHA from upload_folder's return value ({commit!r}). "
            "Refusing to register a wandb reference without one: a reference pinned to "
            "`main` floats and loses provenance (plan section 11)."
        )
    sha = str(sha).strip()
    if not _SHA_RE.match(sha):
        raise RuntimeError(f"{sha!r} is not a commit SHA; refusing to register it")
    return sha


def reference_url(repo_id: str, sha: str) -> str:
    """The SHA-pinned tree URL wandb stores as the artifact reference."""
    return f"https://huggingface.co/{repo_id}/tree/{sha}"


def push_and_register(
    local_dir: str | Path,
    repo_id: str,
    run: Any = None,
    metadata: Mapping[str, Any] | None = None,
    private: bool = True,
    *,
    dry_run: bool = False,
    artifact_type: str = "model",
    env_path: Path | None = None,
    require_wandb: bool = False,
    ignore_patterns: Sequence[str] | None = DEFAULT_IGNORE,
) -> PushResult:
    """Push ``local_dir`` to a private HF repo and log a wandb **reference**.

    This is plan section 0.5's ``push_and_register``, with the failure modes
    that snippet leaves open closed off: credentials are checked up front, the
    commit SHA is validated (never ``main``), and the directory is sanity-
    checked before a multi-GB upload.

    Args:
        local_dir: LoRA adapter dir (``adapter_config.json`` + weights) or a
            merged HF model dir.
        repo_id: ``{org}/{name}``, or a bare name to be qualified with ``$HF_ORG``.
        run: an active ``wandb.Run``. Defaults to ``wandb.run`` if one exists.
        metadata: extra artifact metadata; ``hf_repo``/``hf_sha`` are added.
        private: create the Hub repo private (plan section 0.2). Keep it True.
        dry_run: validate and report, upload nothing.
        require_wandb: raise, rather than warn, when there is no wandb run to
            attach the reference to.
        ignore_patterns: glob patterns not to upload. Defaults to Trainer resume
            state (optimizer/scheduler/rng); pass ``None`` to upload everything.

    Returns:
        PushResult with the repo id and the pinned commit SHA.
    """
    local_dir = Path(local_dir).expanduser()
    repo_id = qualify(repo_id)
    meta = dict(metadata or {})

    if not local_dir.is_dir():
        raise FileNotFoundError(
            f"nothing to push: {local_dir} is not a directory. "
            "Check the checkpoint step number (see dose_map.json written by "
            "scripts/06_train_sdf.py)."
        )
    n_files, n_bytes, names = _inventory(local_dir, ignore_patterns)
    if n_files == 0:
        raise RuntimeError(f"nothing to push: {local_dir} is empty")
    _check_adapter_dir(local_dir, names)

    LOG.info(
        "push %s -> %s (%d files, %.1f MB, private=%s)%s",
        local_dir, repo_id, n_files, n_bytes / 1e6, private,
        "  [DRY RUN]" if dry_run else "",
    )

    if dry_run:
        LOG.info("dry run: would need HF_TOKEN%s; nothing uploaded, nothing logged",
                 " and WANDB_API_KEY" if (run is not None or require_wandb) else "")
        LOG.info("dry run: reference would be %s", reference_url(repo_id, "<commit-sha>"))
        LOG.info("dry run: metadata %s", json.dumps(meta, default=str, sort_keys=True))
        return PushResult(repo_id, None, None, str(local_dir), n_files, n_bytes,
                          pushed=False, registered=False, dry_run=True, metadata=meta)

    require_env(["HF_TOKEN"], env_path=env_path)
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - env problem, not logic
        raise RuntimeError(
            "huggingface_hub is not installed; `uv pip install -r requirements.txt`"
        ) from exc

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    commit = api.upload_folder(folder_path=str(local_dir), repo_id=repo_id,
                               ignore_patterns=list(ignore_patterns) if ignore_patterns else None)
    sha = _commit_sha(commit)
    url = reference_url(repo_id, sha)
    LOG.info("pushed %s @ %s", repo_id, sha)

    registered = register_reference(
        repo_id, sha, run=run, metadata=meta, artifact_type=artifact_type,
        require_wandb=require_wandb, env_path=env_path,
    )
    return PushResult(repo_id, sha, url, str(local_dir), n_files, n_bytes,
                      pushed=True, registered=registered, dry_run=False, metadata=meta)


def register_reference(
    repo_id: str,
    sha: str,
    *,
    run: Any = None,
    metadata: Mapping[str, Any] | None = None,
    artifact_type: str = "model",
    require_wandb: bool = False,
    env_path: Path | None = None,
) -> bool:
    """Log a wandb artifact that *references* the Hub commit (no bytes).

    Returns True if the artifact was logged. Returns False (with a loud
    warning) when there is no active wandb run and ``require_wandb`` is False —
    the weights are already safe on the Hub at that point, so failing the whole
    training run over lineage bookkeeping would be worse than warning.
    """
    if not _SHA_RE.match(sha or ""):
        raise ValueError(
            f"refusing to register reference to {sha!r} — log the commit SHA, not a "
            "branch (plan sections 0.5, 11)"
        )
    try:
        import wandb
    except ImportError:
        msg = "wandb is not installed, so no reference artifact was logged"
        if require_wandb:
            raise RuntimeError(msg + "; `uv pip install -r requirements.txt`")
        LOG.warning("!! %s", msg)
        return False

    run = run or getattr(wandb, "run", None)
    if run is None:
        msg = (
            f"no active wandb run, so {repo_id}@{sha} has NO lineage record. "
            "The weights ARE on the Hub. Re-register with: "
            f"python -m src.artifacts --register-only --repo-id {repo_id} --sha {sha}"
        )
        if require_wandb:
            raise RuntimeError(msg)
        LOG.warning("!! %s", msg)
        return False

    # An active run is already authenticated; no credential check needed here.
    art = wandb.Artifact(
        repo_id.split("/")[-1],
        type=artifact_type,
        metadata={**dict(metadata or {}), "hf_repo": repo_id, "hf_sha": sha},
    )
    art.add_reference(reference_url(repo_id, sha))
    run.log_artifact(art)
    LOG.info("registered wandb reference artifact %s -> %s",
             repo_id.split("/")[-1], reference_url(repo_id, sha))
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _setup_logging(tag: str = "artifacts") -> None:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = logs_dir() / f"{tag}_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(path)],
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--local-dir", type=Path, help="adapter or merged-model directory to push")
    ap.add_argument("--repo-id", help="{org}/{name}, or a bare name qualified with $HF_ORG")
    ap.add_argument("--parent", help="build the repo id from plan 0.4: sdf-{parent}-{direction}-{pct}")
    ap.add_argument("--direction", help="e.g. GA_DS, GS_DA")
    ap.add_argument("--dose-pct", type=int, help="0-100")
    ap.add_argument("--dapo-pct", type=int, help="build a gcvl-dapo-{pct:03d} repo id instead")
    ap.add_argument("--org", default=None, help="override $HF_ORG")
    ap.add_argument("--metadata", default=None, help="JSON object of extra artifact metadata")
    ap.add_argument("--public", action="store_true", help="create a PUBLIC repo (default: private)")
    ap.add_argument("--register-only", action="store_true",
                    help="skip the upload; log a wandb reference to --repo-id at --sha")
    ap.add_argument("--sha", default=None, help="commit sha, with --register-only")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate names/paths/credentials-needed and print the plan; "
                         "touch neither the Hub nor wandb")
    return ap.parse_args(argv)


def _resolve_repo_id(args: argparse.Namespace) -> str:
    org = args.org
    if args.repo_id:
        return qualify(args.repo_id, org)
    if args.dapo_pct is not None:
        return dapo_repo_id(args.dapo_pct, org)
    if args.parent and args.direction and args.dose_pct is not None:
        return sdf_repo_id(args.parent, args.direction, args.dose_pct, org)
    raise SystemExit(
        "give --repo-id, or --dapo-pct, or all of --parent/--direction/--dose-pct "
        "(plan section 0.4 naming)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging()
    if args.org is None and not (args.repo_id and "/" in args.repo_id):
        if args.dry_run and not os.environ.get("HF_ORG"):
            load_env()
        if args.dry_run and not os.environ.get("HF_ORG"):
            args.org = "HF_ORG-UNSET"          # dry run must work without credentials
            LOG.warning("!! HF_ORG is unset; using the placeholder %r for this dry run. "
                        "A real push will refuse until it is set.", args.org)
        else:
            args.org = hf_org()   # raises with an actionable message if unset
    repo_id = _resolve_repo_id(args)
    meta = json.loads(args.metadata) if args.metadata else {}

    if args.register_only:
        if not args.sha:
            raise SystemExit("--register-only needs --sha (the commit to pin)")
        if args.dry_run:
            LOG.info("dry run: would register %s", reference_url(repo_id, args.sha))
            return 0
        register_reference(repo_id, args.sha, metadata=meta, require_wandb=True)
        return 0

    if not args.local_dir:
        raise SystemExit("--local-dir is required (or use --register-only)")
    res = push_and_register(args.local_dir, repo_id, metadata=meta,
                            private=not args.public, dry_run=args.dry_run)
    print(json.dumps(res.as_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
