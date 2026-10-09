# Model API integration (v2)

This guide defines model-side inference for organizer-run RoboValue evaluation.
Participants host a model service; the organizers retain query planning,
annotations, scoring, aggregation and result reporting. This is not a submission
platform, an evaluation endpoint, or a test-set download interface.

RoboValue uses a private, held-out test set. The test set is not publicly released
for download or local evaluation. An externally hosted model API necessarily
receives the observations needed for inference. If observations must not leave
the organizer environment, host the model service within that environment.

## Configure the organizer client

Use `backend: remote_api` in a normal `vmbmk run` configuration. The `model` field
is the participant's model identifier, not an adapter family. No local Python
environment path, checkpoint or GPU field is required. Local configurations
without `backend` retain their existing behavior.

See [the mock configuration](../configs/remote_api.example.yaml). For a real
service, replace the URL, model identity, model/preprocessing versions, input
profile and capability list. Use HTTPS, remove `allow_local_http`, and set
`api.token_env` to the name of an environment variable containing the bearer
token if authentication is required. Never put the credential value in YAML,
the URL, output artifacts or public messages.

The organizer's CPU environment needs the package's existing PyYAML dependency,
Pillow and a video decoder supported by `adapters/video_inputs.py`. No model
checkpoint or GPU inference library is needed by the remote backend. This guide
does not install or upgrade dependencies.

Only `mode: base` is supported in v2. Configure supported metrics explicitly:

| Metric interface | Required service capability |
| --- | --- |
| SA/SD, VOC, MEM-VOC, Cycle-VOC, FPL, TRR, VS | `value` |
| TGA-CT/TGA-CF and CSVC | `compare`, or `value` with explicit `value_difference` |
| SIA/SI | `subtask` |

Unsupported configured capabilities fail before requests are sent. Missing
reported coverage is N/A, not zero. Use single-config `vmbmk run`, not the local
GPU batch scheduler or legacy local `vmbmk infer` command.

## Request and response contract

POST JSON to the configured `api.url`; no endpoint path is prescribed. Use
`Content-Type: application/json` and, when configured, `Authorization: Bearer`
with the credential read from the named environment variable.

The request envelope contains exactly:

| Field | Meaning |
| --- | --- |
| `protocol` | Literal `robovalue-inference-v2` |
| `model_id` | Model identifier from the run configuration |
| `model_version` | Frozen model/checkpoint revision |
| `preprocessing_version` | Frozen model-side preprocessing/prompt revision |
| `shot_mode` | Agreed `zero_shot` or `one_shot` evaluation setting |
| `items` | Nonempty batch of self-contained inference requests |

Each item contains only `request_id`, `op`, `instruction` and `contexts`.
Training references are prepared by the model service and are never attached
to inference requests. The request ID is an opaque UUID;
it is not an internal benchmark query ID. The service must treat repeated IDs
with identical inputs as idempotent and return the same prediction during a run.
Reject reuse of an ID with different inputs.

Each context is `{"frames": [...]}`. Every frame maps the declared camera view
names to base64-encoded PNG strings, without a data-URL prefix. PNG encoding is
lossless relative to the decoded RGB input. Sequence order, repeated frames and
padding are significant and must not be deduplicated or reordered.

| Operation | Contexts | Result |
| --- | --- | --- |
| `value` | One observation context | Finite numeric `score` |
| `compare` | Two contexts, ordered state a then state b | Finite numeric `score` |
| `subtask` | One observation context | Nonempty string `output` |

The response envelope echoes `protocol`, `model_id`, `model_version`,
`preprocessing_version` and `shot_mode`, and contains `results`.
Every result contains exactly
`request_id`, `op` and either `score` or `output`. Return one result for every
request ID, with no extras or duplicates. Results may arrive in any order;
the client restores the original query order. Boolean, NaN and infinite scores,
empty text, mismatched operations/versions and incomplete coverage are errors.

Do not return benchmark metric scores, ground-truth labels or annotations.
Preserve the agreed Adapter value units and direction. If a native model head
requires a documented conversion, such as negating remaining time, reproduce
that conversion in the service wrapper and freeze its preprocessing version.
Do not add a common min-max normalization.

`compare_mode: native` uses the service's agreed ordered comparison semantics.
`compare_mode: value_difference` sends scalar value requests and computes
`V(b) - V(a)` through the existing helper, retaining its left-before-right order.
There is no automatic fallback between these routes. For SIA, return a subtask
description; the organizer's existing forced-choice judge obtains candidate
probabilities and applies the versioned geometric aggregation.

## Observation profiles

`api.input_profile` must specify `name`, a nonempty unique `views` list, and the
parameters listed below. There is no default input profile or history policy.

| Profile | Additional fields | Behavior |
| --- | --- | --- |
| `current_frame_v1` | None | Logical anchor frame only |
| `full_prefix_v1` | None | Every logical frame through the anchor |
| `robometer_prefix_v1` | `num_frames` >= 2 | Existing RoboMeter rounding and final-frame padding |
| `rynnvalue_prefix_v1` | `num_frames` >= 2 | Existing RynnValue floor sampling and repeated indices |
| `procvlm_window_v1` | Positive `window_size`, `max_sampled_frames` | Existing ProcVLM logical-grid window and initial-frame padding |

Unknown profiles or extra/missing parameters are errors, not a request to choose
a fallback sampler. A profile is an input contract, not a claim that an API
replicates all behavior or historical results of its namesake baseline.

The organizer materializes the selected context using the existing playback
mapping. History-bearing inputs remain history-bearing. Cycle inputs retain
the continuous `0..T-1,T-2..0` timeline and a shared turn; the descending half is
not an independently reset video. Source frame indices, timestamps used for
scoring, task/episode IDs, playback labels and annotations stay organizer-local.

## Model-side training references

RoboValue specifies which training demonstrations are eligible; the model
provider prepares them in the service before evaluation. The client does not
read reference videos, sample their frames, upload them or perform adaptation.
Both evaluation settings use the same observation-only request items.

For real-world integration, the designated reference pool is the first 20
training trajectories in the organizer-confirmed ordering. Simulation
references are designated separately. Confirm the selected assets and ordering
with the team before preparing the service; this client does not select them.
An eligible pool is not permission to use all of it in a one-shot evaluation:
the allowed demonstration count follows the recorded evaluation setting.

Keep `api.shot_mode: one_shot` for one-shot evaluation even though no reference
is sent. The provider handles native reference sampling, preprocessing or
adaptation and records the selected demonstrations during integration.
Freeze this preparation before evaluation and update `preprocessing_version`
when reference selection or preprocessing changes, or `model_version` when
adaptation changes the weights. The client trusts those declared versions;
it cannot fingerprint reference files held by the service. Never use held-out
test trajectories as training references.

## Service wrapper and synthetic check

Wrap the model's existing inference entry point: decode the PNGs, preserve their
ordering/view grouping, apply its frozen native preprocessing and prompts,
then serialize the declared numeric or textual output. Each item must be
independent of other items and previous calls. Fixed model-side training
references are allowed under the agreed setting; they are not session memory.
Stateful streaming services are not supported by this protocol.
Cache predictions by request ID within a run
so retries do not produce new stochastic samples.

The [CPU mock service](../src/vmbmk/tools/mock_api.py) implements this envelope,
ID replay and deliberately reversed result order. Its predictions come from
synthetic image pixels and are not benchmark measurements:

```powershell
$env:PYTHONPATH='src'
python -m vmbmk.tools.mock_api --port 8765
```

The mock binds only to `127.0.0.1` and accepts at most 32 MiB per request. This
limit belongs to the toy service, not the inference protocol. Do not send real
held-out data to it. The [remote API tests](../tests/adapters/test_remote_api.py)
exercise it with synthetic frames without needing checkpoints or video codecs:

```powershell
$env:PYTHONPATH='src;tests'
python -m unittest discover -s tests/adapters -p test_remote_api.py -v
```

The sample configuration is for organizer-side evaluation with separately
provided assets. After selecting the real data/task paths and compatible CPU
environment, the existing entry point is `python -m vmbmk run CONFIG.yaml`.

## Failure, resume and privacy

Requests are serial. Timeout defaults to 120 seconds; attempts default to three
and cannot exceed three. Retries use one- and two-second backoff for connection
failures, timeouts, HTTP 429 and HTTP 500/502/503/504. Other statuses, invalid
responses and TLS certificate verification failures fail without fallback.
Response bodies are limited to 8 MiB; an oversized response fails without retry.
HTTPS certificate checks remain enabled and redirects are not followed.
HTTP is permitted only for explicitly enabled literal loopback mock addresses.

Completed batches remain in ordinary result JSONL files. A sibling
`*.jsonl.remote.json` stores organizer-local request-ID mappings and a digest
binding them to model/preprocessing versions, shot mode, API/profile settings,
queries, package code and dataset metadata/asset identities. Asset
identity uses metadata and asset size/mtime checks, including declared videos
outside episode directories, not a full video-content hash.
Changed inputs or incompatible results require a new output path; credentials
are excluded from that digest and never persisted. Do not publish ID checkpoints.

The wire payload is constructed from an explicit allowlist. It excludes internal
query IDs, filenames, filesystem paths, success labels, correct-candidate
identities, SIA targets and TRR stage/branch labels. Images and task instructions
are still inference data; this is not a claim that external services cannot see
or retain observations. Confirm data-handling terms with the team before a real
run. Synthetic contract checks do not establish archived all-model reproduction.
