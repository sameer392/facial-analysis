# H9.3b-E: Evaluation dataset design

Status: planning specification, documentation only. No participants recruited, no recordings collected, no annotations created, no legal approval given. Consent forms, retention wording, the biometric classification of voice data and withdrawal provisions require legal or counsel review before any recruitment or recording. Items marked "proposed" are planning decisions, not legally approved requirements.

## A. Purpose and scope

This specification defines a manually annotated evaluation dataset for:

- English speech
- Hindi speech (Devanagari)
- Natural Hindi-English code-switching

The dataset is intended to measure:

- ASR transcription quality (WER and CER)
- Word timestamp accuracy
- Language identification
- Operational resource use on the target GPU

The initial purpose is evaluation only. This specification does not authorise model training or fine-tuning, and it does not authorise any model download, installation or conversion. Normalisation policy is defined separately in H9.3c (section F).

## B. Dataset composition

| Dataset category | Target duration | Speakers | Content |
|---|---:|---:|---|
| English | 20–30 minutes | at least 8 | Read, conversational and spontaneous speech |
| Hindi | 20–30 minutes | at least 8 | Devanagari, regional accents; read, conversational and spontaneous speech |
| Code-switching | 10–15 minutes | at least 8 bilingual participants | Natural Hindi-English mixed speech |

Durations are proposed starting targets. They are not validated statistical requirements.

Requirements:

- **Minimum speakers:** 8 distinct speakers per language, and at least 8 bilingual participants for code-switching (proposed).
- **Speaker-level split (proposed):** 3 development speakers and 5 evaluation speakers per language, giving 37.5% and 62.5%. The 40% development target is approximate. Speaker separation takes priority over the exact percentage.
- **Shared participants:** a participant may contribute to more than one category. A participant must remain entirely within one split across all categories, which prevents cross-split speaker leakage.
- Adults only, who participate voluntarily.
- Variation in speaking speed and pauses.
- Quiet-room recordings and recordings with realistic background noise.
- Original device sample rate and channel configuration preserved.
- Lossless source recordings.
- Device, room and noise-condition metadata recorded, without identifying participants.

This is a minimum planning target. It is not a claim of statistical sufficiency. Results must be reported with confidence intervals, the number of speakers, and the limitations in section J.

## C. Consent, privacy and retention

Safeguards (proposed):

- Written informed consent, in English or Hindi, obtained before any recording.
- Consent explicitly covers recording, transcription, annotation, timestamping, storage, access, retention, withdrawal, and evaluation use. Consent wording must be reviewed by counsel.
- Maximum retention: two years (proposed), subject to consent and legal review.
- Earlier deletion when a valid withdrawal request is received.
- Use is suspended immediately after a valid withdrawal request.
- Affected audio, annotations, derived artefacts and manifest references are deleted within 30 calendar days.
- No training use without separate written permission.
- Recordings are stored outside the Git repository.
- Pseudonymous speaker and recording identifiers only.
- Minors are excluded.
- A named access list and an access log are maintained.

**Biometric classification (legal review required).** Voice recordings may or may not be biometric data under applicable law. This specification does not assume either answer. Before recording begins, counsel must assess the classification, consent requirements, processing purpose, storage, retention and withdrawal implications.

The identity and consent register is kept separate from the pseudonymous evaluation dataset. The identity mapping is never stored in the evaluation directory, and the evaluation manifest refers to participants only by pseudonymous identifiers and consent references.

Status: consent forms, retention wording, the biometric classification and withdrawal provisions are **not legally approved**. Counsel review is mandatory before recruitment or recording.

## D. Ground-truth transcription policy

- Speech is transcribed as spoken. Nothing is translated.
- Hindi speech is written in Devanagari.
- English speech is written in Latin script.
- Romanised Hindi is excluded from the initial evaluation scope.
- Every token carries a language tag: `en` or `hi`.
- Punctuation and filler conventions are fixed in the annotation guideline before any scoring.
- Numerals, dates and Hindi nukta forms are handled by the H9.3c normalisation policy (section F), not by annotators.
- Unintelligible speech is marked `[UNK]`.
- Non-speech noise is marked `[NOISE]`.
- `[UNK]` and `[NOISE]` are excluded from WER under the agreed scoring policy, and their occurrence is reported separately.
- Each recording has two independent annotators, followed by adjudication.
- Annotator identifiers and adjudication status are recorded.

## E. Word-level timestamp annotation

- Word start and end times are stored as integer microseconds on the session clock.
- The corresponding sample indices are stored alongside.
- **Session-clock origin (decided):** the first sample of each evaluation recording is its origin. Each recording is recorded with `origin = 0` and `external_offset = 0`, both written explicitly in the manifest. This is consistent with the H9.3a timing model, `session_us = audio_start + external_offset − origin`.
- Boundaries are set manually, with audio-editor assistance.
- Independent annotators' boundary differences are compared and reported separately from text disagreement.
- Sample indices and integer microseconds are the authoritative timing representation.
- Original timing values are never silently rounded or replaced.

## F. Metrics

1. **WER and CER**, using Levenshtein distance over the scored tokens, after the frozen H9.3c normalisation policy.
2. **Hindi WER**, reported before and after the H9.3c normalisation policy.
3. **Code-switch WER**, reported separately for English tokens, Hindi tokens and the combined transcript.
4. **Word timestamp error** for correctly matched words:
   - Start-time mean absolute error (MAE)
   - End-time mean absolute error (MAE)
   - Median error
   - 90th percentile error
   - Percentage of words within 50 ms and within 100 ms
5. **Language identification accuracy**, with a confusion matrix over `en`, `hi`, `mixed` and `uncertain`.
6. **Operational processing time.**
7. **Peak RAM and VRAM**, on the confirmed target GPU: NVIDIA GeForce RTX 3080, 10 GB.

**Normalisation (H9.3c).** The normalisation policy is defined in H9.3c, not here. It must:

- be developed using development data only;
- be frozen before evaluation;
- never be tuned against evaluation results;
- resolve numerals and number words, dates and date formats, Hindi nukta forms and Unicode variants, punctuation, and whitespace and tokenisation.

No performance result, and no acceptance criterion, is claimed in this document.

## G. Development and evaluation split

- **Speaker-level split (proposed):** 3 development and 5 evaluation speakers per language, as set in section B. Splitting is by speaker, never by utterance.
- The 40% development target is approximate.
- A participant who contributes to several categories stays in one split for all of them.
- Evaluation data hashes are frozen once the split is made.
- No model settings or normalisation policies are tuned against evaluation results.
- Any re-run after evaluation is recorded as a new run (section I, step 5).
- A leakage-check procedure is proposed as future work. It is not implemented in this milestone.
- **Confidence intervals (proposed):** speaker-level cluster bootstrap resampling. Each interval is reported with the number of speakers it is based on. Five evaluation speakers per language may be insufficient for robust population-level conclusions, and this limitation must be stated with any reported result.

## H. Data contract

Fields in the evaluation manifest:

- Pseudonymous recording ID
- Pseudonymous speaker ID
- Split (development or evaluation)
- Language or category
- Recording condition (quiet or noisy)
- Speech style (read, conversational or spontaneous)
- Source file SHA-256
- Duration
- Sample rate and channel count
- Session-clock origin and external offset (both 0, per section E)
- Consent reference (an internal code, not the consent register itself)
- Ground-truth transcript
- Token-level language tags
- Word timestamps in integer microseconds, with sample indices
- Annotator identifiers
- Adjudication status

The consent register is kept separate from this manifest.

## I. Withdrawal workflow

1. Suspend all further use of the affected data immediately after a valid withdrawal request.
2. Identify every related recording, annotation, derived artefact and manifest entry.
3. Delete the affected data within 30 calendar days.
4. Update the manifest and the deletion log.
5. Re-run affected evaluation results as a new, separately recorded evaluation run. Preserve the previous run and document the reason for the new run. Never silently overwrite historical evaluation results.

The consent documentation must explain that aggregate results already published may not be fully removable, and that this limitation is subject to legal review.

## J. Risks and limitations

- Limited speaker diversity: five evaluation speakers per language may not represent the target users.
- Annotation cost and inter-annotator disagreement, especially for spontaneous speech.
- Consent and legal requirements, including the unresolved biometric classification, which may restrict the purposes or retention periods.
- Possible evaluation leakage if speakers overlap between splits.
- Limited statistical confidence from a small speaker pool.
- Hindi and code-switching accuracy may vary substantially by accent and recording condition.
- A low inter-annotator WER does not prove that annotations are correct.

## K. Current status

| Item | Status |
|---|---|
| Evaluation design | Planning specification approved; this document records it |
| Consent forms | Not yet legally approved |
| Biometric classification of voice data | Not yet assessed; counsel review required |
| Participants | None recruited |
| Recordings | None collected |
| Annotations | None created |
| Model training | Not authorised |
| Model downloads and installation | Not authorised by this document |

## L. Open decisions and approval status

Decisions recorded in this revision (proposed unless marked approved):

| Item | Decision | Status |
|---|---|---|
| Minimum speakers | 8 per language; 8 bilingual participants for code-switching | Proposed |
| Speaker split | 3 development / 5 evaluation per language, speaker-level | Proposed |
| Maximum retention | 2 years, subject to consent and legal review | Proposed; legal review required |
| Withdrawal deletion window | Within 30 calendar days; suspend use immediately | Proposed; legal review required |
| Recruitment | Adults, voluntary participation, written informed consent | Proposed; legal review required |
| Counsel review | Mandatory before recruitment or recording | Required; not yet done |
| Biometric classification of voice | Not assumed; counsel to assess | Open; legal review required |
| Romanised Hindi | Excluded from initial evaluation | Proposed |
| Inter-annotator WER review trigger | 5%: above it, mandatory review and adjudication; at or below it, disagreements are still reviewed and adjudicated | Proposed; not an acceptance threshold |
| Timing boundary disagreement | Reported separately from text disagreement | Proposed |
| Session-clock origin | First sample of each recording; origin 0, offset 0, recorded in manifest | Proposed; consistent with H9.3a |
| Normalisation policy | Defined in H9.3c, developed on development data, frozen before evaluation | Deferred to H9.3c |
| Confidence intervals | Speaker-level cluster bootstrap, reported with speaker counts | Proposed |
| Evaluation re-runs | New, separately recorded run; no silent overwrite | Proposed |

No item in this table is a legally approved requirement. Recruitment, recording and personal-data collection remain unauthorised until counsel review is complete and the consent forms are approved.
