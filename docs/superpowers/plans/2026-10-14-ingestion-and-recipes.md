# Ingestion and recipes implementation plan

Goal: complete the four gaps in the user brief while preserving existing routes and CLI commands.

Architecture: retain DataSanitizer for both structured and document inputs. Keep catalog, memory estimates, and intensity mapping in core. Adapters own extraction and model calls. API composition owns persisted ingestion jobs and generated datasets.

Execution: implement inline, with failing unit tests before each production change. Commit each gap after its focused checks pass.

- [ ] Gap 1: document adapter, optional dependencies, settings, source-group splitting, generated-data persistence, upload progress, UI upload support. Test malformed and empty documents, archive safety, token caps, grounding, duplicate filtering, failed model calls, credential placeholders, and split isolation.
- [ ] Gap 2: preference normalization and precedence, strict reward ordering, DPO selection and training dispatch. Test strings, message lists, mixed schemas, invalid rewards, and missing preference responses.
- [ ] Gap 3: Hub-verified catalog, model-size-aware quantization, models route, grouped selector and custom option. Test model fit boundaries, invalid VRAM, unknown model handling, and tool-compatible defaults.
- [ ] Gap 4: domain intensity mapping, server validation, Basic and Advanced panels, override markers and reset. Test preset scaling, GRPO cap, invalid numbers, memory growth, and submission rejection before backend launch.
- [ ] Verify: full pytest, frontend lint and build, manual API uploads and catalog check. Report external runtime blockers without substituting synthetic model results.

Review risks: archive traversal and expansion, document leakage through overlapping chunks, incomplete generation, custom-model size uncertainty, and UI overrides masking model recommendations.
