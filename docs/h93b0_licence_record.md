# H9.3b-0: Licence and provenance record

Status: Recorded (documentation only). Commercial use is NOT cleared. Technical and licensing assessment only. Not legal advice. Legal confirmation is required before commercial distribution or deployment.

## 1. Source model

| Item | Record |
|---|---|
| Repository | openai/whisper-large-v3 (Hugging Face) |
| Revision | 06f233fe06e710322aca913c1bc4249a0d71fce1 (latest commit on main, "Update README.md", Aug 12, 2024) |
| Files intended for conversion | model.safetensors (fp16, 3.09 GB) and config and tokenizer files |
| Declared licence | apache-2.0 (YAML front matter and page header) |
| Licence file in repository | None (no LICENSE or NOTICE file at this revision) |
| Licence text in model card body | None |
| Upstream NOTICE | None present, so no NOTICE text to reproduce |
| Status | Declared licence only. Written confirmation from OpenAI or counsel is pending. |

## 2. Converter

| Item | Record |
|---|---|
| Tool | ct2-transformers-converter (CTranslate2 Transformers converter) |
| Source file | python/ctranslate2/converters/transformers.py (no per-file header) |
| Licence | MIT, from the CTranslate2 repository LICENSE (Copyright 2018- SYSTRAN; 2019- The OpenNMT Authors) |
| Version to pin | ctranslate2==4.9.0 (conversion environment and runtime) |
| Documented Whisper route | Yes, CTranslate2 documents Whisper conversion with this tool |

## 3. Dependencies (separate from runtime)

Conversion-only dependencies, which must not be installed into the runtime environment `.venv-asr`:

| Dependency | Licence | Use |
|---|---|---|
| transformers | Apache-2.0 | Loads the source model in conversion |
| PyTorch | BSD-style | Required by the transformers loader during conversion |
| huggingface_hub | not yet checked | Downloads, if used |
| numpy | not yet checked for this record | Array handling |

The conversion environment is `.venv-convert`. It is not yet created and needs separate approval.

## 4. Conversion operation (terminology correction)

The source model is stored in FP16. The conversion is recorded as a **format conversion to CTranslate2 with the float16 storage dtype**, using the flag `--quantization float16`.

This is not described as quantisation. Whether the numerical precision changes must be checked against the converted file, not assumed. The provenance record stores the exact command and flags, and the output file hashes, rather than a precision label.

Planned command, to be recorded verbatim when run:

    ct2-transformers-converter --model <local source path> --output_dir <output path> --copy_files tokenizer.json preprocessor_config.json --quantization float16

## 5. Obligations for converted weights (Apache-2.0)

Assuming the declared licence applies, redistribution of the converted weights would require:

1. Supplying a copy of the Apache-2.0 licence with the distribution.
2. Retaining the applicable copyright and attribution notices from the upstream source.
3. Marking the converted files as modified (format conversion), with a statement of the changes.
4. Including the upstream NOTICE, if one exists. None exists at this revision.

Internal use does not trigger the redistribution duties, but the provenance and attribution record should be kept regardless.

The converted weights should not be relicensed as MIT. The Systran conversion's MIT tag does not change the upstream terms.

## 6. Provenance record to keep for every conversion

- Source repository and full revision hash
- SHA-256 of each source file used
- Converter version and commit, and CTranslate2 version
- Exact command and flags
- Output file names, sizes and SHA-256 hashes
- Date, and the operator who ran it
- Licence declarations and the attribution text used

## 7. Status of each action

| Action | Status |
|---|---|
| Accept H9.3b-0 investigation | Approved (per user decision) |
| Use original OpenAI weights as the preferred route | Preferred |
| Obtain written licence confirmation from OpenAI or counsel | **Required before download. Pending.** |
| Create `.venv-convert` | Awaiting approval |
| Install PyTorch and conversion dependencies | Awaiting approval |
| Download OpenAI weights | **Not approved** |
| Convert the model | **Not approved** |
| Commercial deployment | **Not cleared** |
| Add this record to the repository | Awaiting separate authorisation |
| Commit | Not authorised |

## 8. Open questions

1. Does the Apache-2.0 metadata declaration, with no licence file, establish the applicable licence for the weights? Written confirmation from OpenAI or counsel is needed.
2. Is internal use alone sufficient for the current milestone, with redistribution and commercial use deferred?
3. Should the conversion output be stored only locally, with no distribution?
