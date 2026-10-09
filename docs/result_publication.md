# Result interfaces and publication

## SIA execution

All run_sia stages return dictionaries. The default all stage and the judge stage
retain task-internal geometric probability aggregation followed by task-level
averaging, as specified in Appendix F.2 of the author-confirmed paper.
The generate stage returns status=generated, protocol, responses_path and n_queries.
It is an intermediate artifact, not a score: it has no mean and cannot be published.
The CLI records it in sia_generation.json in the resumable output directory, while
responses and query caches retain their existing names and formats. No dummy score
or additional judging call is introduced. Callers previously checking for None must
instead check status=generated. Multi-GPU generation remains supported.

## CSVC coverage and result shape

The canonical metric key and module are now csvc and vmbmk.metrics.csvc.
Legacy cspc configuration/result keys normalize to csvc at input boundaries.
Dataset CSPC capability labels remain readable and normalize to CSVC. Existing
episode metadata cspc.solution_type and cspc query IDs are unchanged to preserve
datasets and frozen query selection. The 2026-10-05 scoring correction uses
semantic-gain-symmetric-ratio-v2; old rmse-v1 metric caches are incompatible.
New outputs use csvc operation names and publication directories; use a new run
directory when migrating, rather than reinterpreting completed old caches.

CSVC retains the csvc CLI key and supports only domains: [id]. Requests for ENV-OOD
or EMB-OOD fail explicitly; omitted domains in direct metric calls select ID only.
No synthetic OOD cells or scores are produced.

CSVC computes (G-D)/(G+D) without clipping, with values in [-1, 1]. A covered
task with G+D=0 receives -1 and participates in task-macro averaging; its
identifiable flag remains false as a diagnostic. Missing matched semantic
coverage remains N/A and is excluded, not assigned the zero-denominator penalty.
The score, task_results, diagnostics and bootstrap fields remain available.
domains.id exposes the same overall score and per-task values, including negatives.
Completed old metric caches are rejected by evaluation and result validation;
publication merge also rejects different protocol versions. Archived result
artifacts are not rewritten, and query records are not silently resampled.

## One publication implementation

vmbmk.tools.results.publication owns result publication and index construction. Shared identity
normalization lives in adapters/registry.py and metrics/registry.py. These modules
import no GPU scheduler or model runtime. Existing tools and the
scheduler delegate to it; no GPU is needed to publish an existing result.

Publication takes an explicit PublicationRequest with baseline, config path,
source run name, scored results, resume contracts and execution mode. It does
not read scheduler status, GPU assignment or task retry state. Manual and batch
callers construct the same request. Validation exposes validate_run; batch
calls it directly while the CLI owns argument parsing and exit codes.
Temporary synthetic validation on 2026-10-06 confirmed equivalent manual/request
summaries apart from timestamps. Real final-results trees were not changed.

Publish a completed run explicitly:

    python -m vmbmk.cli results publish results/my_run/metrics.json \
      --results-root results --final-results-root final_results --mode publish

The supported layout remains baseline/metric/summary.json with per-task JSON files.
Summaries and task files carry schema_version=robovalue-results-v1. Scientific
protocol versions remain inside result and are distinct from the publication schema.
final_results/index.json has one canonical entries list, with baseline, metric,
task count, relative summary link, protocol, source_run, updated_at and SHA256.
The index is rebuilt from current summaries after publication, including derived
VOC exports. Consolidation history moves to archive_index.json (records), not the
website index. Older summaries can be indexed but do not automatically gain a
schema_version; republish them to adopt the new summary format.

--merge supports SA, TGA, VOC, MEM-VOC, Cycle-VOC, CSVC, VS, FPL and TRR. It
rejects differing protocols. SIA merge requests still fail rather than silently
replacing previous tasks. VS component summaries and TRR group diagnostics are
merged alongside task scores; CSVC retains its dedicated scientific aggregation.
Without --merge,
publication explicitly replaces the selected metric's materialized summary/task
files. Generated SIA and nonfinite JSON results are rejected. Publication uses a
nonblocking OS writer lock, staged updates and an undo journal. A failed commit
restores the previous files; the next writer recovers a journal left by a process
interruption. The index commits last. This does not guarantee power-loss recovery.
An active writer causes other writers to fail with a retryable busy message.

Local readers requiring a consistent multi-file view must use the same
PublicationBatch lock. Unlocked website readers can observe intermediate files:
verify summary bytes against the index SHA256 and retry on mismatch. Do not upload
a live directory mid-publication. The stable paths and publication schema remain
unchanged; the lock and journal are private operational files, not website assets.

Reruns remove only task files listed by the previous summary and absent from the
new result. Unowned JSON files remain untouched, and colliding unowned task files
cause an error. Task identities reject traversal, Windows reserved names and
case/Unicode-normalization collisions. Resolved destinations must stay in root.
Consolidation preserves unrelated directories and publishes its selected results
in one batch with one index refresh; it no longer clears the entire output tree.
Unselected old publications remain until explicitly reviewed and removed.

Publication requires a readable YAML mapping with an explicit model. Known
checkpoint sizes retain their identities; an unrecognized checkpoint is not
assigned a known model size. ProcVLM identity follows the adapter
use_lora/checkpoint_by_task rule, not a one-shot substring in a path. VLAC uses
reference_data to select its default shot mode. Review destination identities
before publishing old configurations; no previous directory is renamed here.

Batch missing-task completion uses batch-resume-v2 and checks effective model,
query and relevant implementation fingerprints. Obsolete completion contracts
are rejected. Standalone publication still accepts matching legacy protocol
metadata when no completion contract is required; --merge is not a replacement
for scientific compatibility verification.

## Remaining release boundaries

This is a local result contract, not a website deployment. Existing baseline names
and checkpoint/source metadata are preserved for compatibility. They are not a
complete public-data sanitization policy: review private paths, identifiers and
redistribution rights before uploading artifacts. Sim/real identity still requires
an explicit website export decision; this pass does not combine the two datasets or
claim archived paper-result reproduction. Existing historical publication holds
remain honored. Consolidation remains an administrative rebuilding tool, not a
safe website request handler. Archive copies are outside the scored publication
transaction and may remain after a later batch failure.

Current validation evidence and its limitations are recorded in testing.md.
Historical broader-suite records belong to docs/history, not current coverage claims.
