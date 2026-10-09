# Provide an inference service and adapter

To participate in RoboValue, provide your model's inference service together
with a model-specific adapter. You implement the adapter; the RoboValue team
reviews it and runs the evaluation. The service may use its native interface.
There is no required universal HTTP payload or shared sampling profile.

RoboValue uses a private, held-out test set. Participants provide a model API,
and the RoboValue team conducts the evaluation. The test set is not publicly
released for download or local evaluation.

## Prepare your service

Keep the model running in your chosen environment. Provide an accessible
inference endpoint, its native request/response description, the model and
preprocessing versions, and authentication instructions if applicable. Share
credentials separately; the adapter must read them from environment variables,
not source code or configuration files.

An external service receives the observations needed for prediction, but not
test annotations or benchmark scores. If observations must not leave the
organizer environment, deploy the service within an organizer-controlled
environment. Use HTTPS with certificate verification and do not forward
credentials across redirects.

## Implement the adapter

Start with the [adapter interface](../src/vmbmk/adapters/base.py) and the
[CPU mock adapter](../src/vmbmk/adapters/mock_service.py). The adapter prepares
your model's native inputs, calls your service, and returns predictions in the
same order as the queries. Implement only the operations your model supports:

| Method | Input | Output |
| --- | --- | --- |
| `value` | One observation context and instruction per query | One native scalar per query |
| `compare` | Ordered contexts A and B, with an instruction | One native comparison score per query |
| `subtask` | One observation context and instruction per query | One predicted subtask description per query |

The adapter owns camera selection, history/frame sampling, padding, resizing,
encoding and output conversion. Preserve your model's original units and score
direction; do not apply a common normalization. For scalar-value models whose
comparison convention is higher value = better, the existing helper computes
`V(b) - V(a)`. Other models must retain their native comparison semantics.
Subtask descriptions are scored separately by the organizer's judge.

Use `query_view` and `PlaybackView` when preparing recorded observations.
History-dependent inputs must retain the necessary prefix. Cycle inputs use a
continuous forward/backward timeline with a shared turn, not an independently
reset reverse segment. The mock's complete-prefix policy is illustrative, not
a prescribed policy for other models.

Only pass necessary observations and instructions to the service. Do not pass
internal query IDs, organizer paths, success labels, test annotations or
ground-truth candidate identities. Validate external responses and expose
service failures; do not turn failed calls into zero scores or invented N/A.

For an adapter that calls an external service, declare `requires_gpu = False`
and `requires_checkpoint = False` when no GPU or weights are required on the
adapter machine. The adapter still runs in a Python environment locally; the
model itself may run on another machine. Register its own model identity, not
a shared generic service identity.

## Prepare training references

References are prepared on the model side before evaluation, not attached by
RoboValue to each inference request:

- **Zero-shot:** do not use training trajectories as inference references.
- **One-shot:** use the first training trajectory for the relevant task in the
  supplied order, consistently across queries.
- **Few-shot:** use all 100 training trajectories for the relevant task.

These selection rules apply to both simulation and real-world tasks. Your
adapter/service documents how the selected references enter the model. The
mock does not implement reference-conditioned inference or train a model.

## Hand over the integration

Provide the following for review:

- The adapter source and supported operations, with native output units and
  score direction.
- The service endpoint, native input/output specification, model/preprocessing
  versions and authentication instructions without credential values.
- The adapter's environment requirements, camera/history/preprocessing rules
  and reference-conditioned modes, if supported.
- A minimal configuration using placeholders for deployment-specific settings,
  and a synthetic example that can be run without private test observations.

The RoboValue team reviews the adapter and evaluates compatible metrics with
the private test set. Unsupported coverage is reported as N/A, not zero. This
guide does not define a submission platform or guarantee an evaluation time.

## Check the mock example

The [mock configuration](../configs/mock_service.example.yaml) selects
`model: mock_service` through the ordinary runner. It needs no GPU, checkpoint
or running API. Its `_predict_value` and `_predict_subtask` methods are local
toy predictions that stand in for service calls. Replace these with your
model-specific calls and adapt input preparation to your native model; do not
use the mock's toy scores as a baseline.

The [mock tests](../tests/adapters/test_mock_service.py) exercise all three
operations, worker execution, history order, failure recovery and SA scoring
using synthetic RGB frames. They do not establish real video decoding, network
service reliability, reference-conditioned performance or model reproduction.
See [testing](testing.md) for the existing-environment test command.
