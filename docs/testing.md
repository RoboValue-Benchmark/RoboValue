# Testing

## Supported scope

The automated suite covers metric scoring and one CPU smoke test for each
supported adapter implementation. It is intentionally not a general regression
suite for the entire evaluation package.

Metric tests retain formulas, strict signs, ties, constant-input conventions,
missing coverage, eligible cohorts, sampling that defines the score, semantic
alignment, history-dependent inputs, and metric-specific aggregation order.
Synthetic dataset fixtures and predetermined predictions are sufficient; no
real data or model inference is needed. File writes only build temporary fixture
metadata and mock run artifacts in temporary directories, never published tables.

| Module | Scoring contract |
| --- | --- |
| tests/metrics/test_sa.py | Terminal 2% mean, strict success/failure comparison and diagnostics |
| tests/metrics/test_sia.py | Ground-truth candidate probabilities, within-task geometric mean, task/domain macro |
| tests/metrics/test_tga.py | TGA-CT/TGA-CF inputs, strict top1 and separately labeled pairwise diagnostics |
| tests/metrics/test_voc.py | Cycle-forward VOC, shared cohort, constants, invalid inputs and task/domain scores |
| tests/metrics/test_voc_mem.py | MEM-VOC inputs and progress-order scoring |
| tests/metrics/test_cycle_voc.py | Continuous forward/reverse inputs, shared turn and half-cycle aggregation |
| tests/metrics/test_fpl.py | ZigZag onset, ties, normalized error, no-decline penalty and aggregation |
| tests/metrics/test_trr.py | Time-weighted stage directions and binary branch conjunctions |
| tests/metrics/test_vs.py | Endpoint-processed efficiency ratio times raw nonflat time coverage |
| tests/metrics/test_csvc.py | Current signed ratio, semantic alignment, four boundary pairs and task macro |

Cycle-VOC&VS tests also check the shared normal-ST selection, unchanged cycle
query identities/history, forward-only 5 Hz extraction and one inference call
when VOC, Cycle-VOC and VS run together. The pure VS reference remains unchanged.
Domain controls compare joint and separate ID/ENV-OOD/EMB-OOD CLI runs using
unequal episode counts, different deterministic curves, query/result identity
checks and saved scores. Missing OOD normal-ST coverage must fail explicitly,
not silently create a zero score. These are CPU contract tests, not GPU model
performance measurements.

The CSVC tests cover the current local implementation. They do not establish
archived table reproduction, execution-signature grouping, or an unimplemented
three-pair exception. A scoring change requires an explicit protocol decision,
not adjusting tests to match a published number.

All adapter smoke tests live in tests/adapters/test_interfaces.py: RoboMeter, RoboDopamine,
ProcVLM, RoboReward, VLAC, TOPReward (Qwen/Molmo backend subtests), RoboFAC, RynnValue,
LIV and SIA-only FailSafe. Each implementation has exactly one test method,
using predetermined outputs to check a representative operation. These tests
are minimal interface checks, not comprehensive model correctness checks.
tests/synthetic_data.py contains shared synthetic dataset construction and task metadata
updates. Tests use unittest.TestCase and run through pytest; there are no
per-module standalone runners or workflow regression modules.

## Run

Model service adapter tests are in tests/adapters/test_mock_service.py. They
exercise the ordinary CPU worker and runner with synthetic RGB frames, not
held-out observations or a running API. Coverage includes scalar/comparison/text
outputs, history order, checkpoint recovery, SA aggregation and changed-config
rejection. A test-only frame reader replaces video decoding in the subprocess;
these tests do not validate codecs, network behavior or real models.
Use an existing environment with Pillow and PyYAML:

    PYTHONPATH=src:tests python -m unittest discover -s tests/adapters -p test_mock_service.py -v

Use an existing Python environment with PyYAML and pytest. Do not install model
packages or download weights to run this suite. From the repository root:

    # POSIX shell
    PYTHONPATH=src:tests python -m pytest tests -q

    # PowerShell
    $env:PYTHONPATH='src;tests'
    python -m pytest tests -q

For a scoring change, run its module first:

    python -m pytest tests/metrics/test_csvc.py -q

For an adapter change, run its one smoke test:

    python -m pytest tests/adapters/test_interfaces.py -k robodopamine -q

Then run the full reduced suite. Syntax/whitespace checks remain available:

    python -m compileall -q tests
    git diff --check

The suite requires no GPU, checkpoints, video decoding, external network access,
API credentials or Torch. Remote API tests require Pillow and loopback HTTP;
metric and local adapter smoke tests do not require Pillow or NumPy.
Fixture video files are placeholders.

## Deliberate exclusions

Rerun/rescore workflows, missing-task completion, schedulers, caches, result
publication/merging, CLI/configuration plumbing, environment installation,
workers, and generic data/query/playback infrastructure have no standalone
automated tests in this suite. The test-scope reduction did not remove their
production functionality; later refactors require separate, explicitly reported
validation.
Selected metric tests now exercise dispatch, shared inference and provenance
rejection as regressions for metric-facing changes. This does not turn the suite
into comprehensive workflow coverage; publication and scheduler validation below
is separately executed smoke evidence, not an automated-suite guarantee.
An untested workflow must be reported as unverified, not considered correct
because the scoring tests pass. Historical broader-suite validation documents
remain historical evidence and are not current coverage claims.

Passing this suite verifies only the tested local scoring contracts and minimal
adapter operations. It does not reproduce paper tables, validate real
checkpoints, establish environment readiness, or guarantee every inference path.

## Verified execution on 2026-10-06

After the TOPReward adapter consolidation, the local suite has 107 test methods
and 30 passing subtests. The previous separate Molmo adapter test is now a
backend subtest of the same TOPReward interface test, including Molmo loader
and scorer routing. The 108-test remote evidence below predates this consolidation;
neither consolidated backend has a new real-checkpoint GPU validation yet.

The current local source was copied into an independent remote validation
directory. Existing TOPReward Python ran the reduced suite: 108 tests passed.
Local pytest also reported 28 passing subtests; reporting differs by pytest/plugin
version, so the remote count is not presented as a separate coverage expansion.
No packages or weights were installed or downloaded, and published results were
not overwritten. This was an existing-environment check, not a setup-script test.

RoboDopamine-3B then ran actual inference using GPU 2 on fold_clothes,
episode_0000011, in the ID smoke dataset. Normal annotated forward/reverse
queries produced 101 cycle predictions. VOC and VS each reused 51 forward values,
giving 203 separately identified output records. The model loaded once, result
validation returned valid=true, the second invocation reused the cache, and GPU
2 memory was released after completion.

| Metric | Fresh GPU inference | Saved-prediction rescore |
| --- | --- | --- |
| VOC | 0.7653354500840708 | 0.773808178808216 |
| Cycle-VOC | 0.7929084724910904 | 0.7978955374017038 |
| VS | 0.28754080145262967 | 0.2946687102118701 |

Saved-prediction Cycle-VOC and VS match the prior recorded scores. Fresh inference
is not numerically identical; no cause is established here and no score tolerance
was chosen to relabel it as exact reproduction. These are one-episode smoke
scores, not benchmark table entries or evidence of general model quality.

Separately, temporary synthetic artifacts verified that validation API, CLI and
batch inspection agree, and that manual publication and PublicationRequest
produce equivalent summaries apart from timestamps. No real publication tree
was modified. This does not exhaust merge, rerun or transaction failure cases.

The remote evidence is retained in an independent validation workspace, not in
the distributed package. There is no latest-code all-model/all-metric GPU sweep,
real OOD evaluation, live SIA judge validation or clean-install certification.
