# Paper evidence matrix — Campaign C canonical status

Date: 2026-10-07. Historical pre-Campaign-C copy: `docs/PAPER_EVIDENCE_MATRIX_20261008_PRE_CAMPAIGN_C.md`.

## Evidence scope

The canonical held-out artifact is `results/controller_shift_test_final_guard1_revision_full_20261007_jetson/` with paired analysis in `results/controller_shift_test_final_guard1_revision_full_20261007_jetson_analysis/`. It contains 20,480 rows, 16 factor conditions, 64 summaries, 640 paired streams, ten held-out seed blocks, four episodes, eight slots, and four controllers. It is a normalized RIS-only simulation: 2-bit phases, equal-power matched beams, no direct BS–user path, no absolute path loss, no OTA waveform, and no calibrated population posterior.

## Claim ledger

| Claim | Status | Evidence and permitted wording |
|---|---|---|
| Causal per-user risk-aware phase reuse is implemented | **PASS** | Source, configuration, raw rows, tests, and hashes are retained. Say “implemented and evaluated under the normalized simulator.” |
| Corrected held-out campaign is complete | **PASS** | Remote manifest/progress are `complete`, 20,480/20,480 rows; remote verifier reports 16 conditions, 64 summaries, 640 paired streams, and raw/JSON/CSV agreement. |
| Absolute weak-user outage upper95 | **PASS (declared simulator grid)** | Maximum one-sided upper95 = 0.000 against 0.05. Refused/missing/failed slots remain in the denominator. |
| Relative outage delta vs always-risk | **PASS (declared simulator grid)** | Maximum one-sided upper95 = 0.000 against 0.01. |
| Goodput relative delta vs always-risk | **PASS (declared simulator grid)** | Minimum lower05 = −0.0061526960 against −0.05. |
| Phase-control rate-evaluation reduction vs always-risk | **PASS (declared simulator grid)** | Minimum lower05 = 0.3339843750 against 0.20. |
| Periodic-risk/nominal work reduction | **FAIL as a gate** | At least one condition has a negative lower bound. Report as sensitivity evidence; do not relabel it PASS. |
| Paired uncertainty | **PASS** | Complete seed block is the unit; 10,000 draws; analysis seed 20261031; slots are nested. |
| Zero-failure finite-sample sensitivity | **PASS (diagnostic)** | Clopper–Pearson audit gives maximum weak-user upper95 `0.009318` over 320 scheduled slots per condition; it does not replace the registered seed-block bootstrap or add independent samples. |
| Baseline freeze | **PASS** | Guard1 revision selected on disjoint development seeds; freeze and hashes precede held-out read. |
| Physical phase-write and slot timing | **UNKNOWN / NO-GO** | No RIS controller endpoint, measured `T_slot`, phase-write completion, or overrun trace. Simulated P95 (388.083 ms) is diagnostic only. |
| OTA/deployment/hardware real-time feasibility | **UNSUPPORTED** | No waveform, payload/error capture, physical phase commit, or physical timing artifact. |
| Calibrated posterior, DRL/learned controller, universal superiority, first/novel claim | **UNSUPPORTED** | The ensemble is a design surrogate; no trained policy was evaluated; evidence is condition-specific and adjacent work is documented. |
| Final bibliography | **PASS for DOI identity / PENDING publisher and classification checks** | 34 DOI records are resolved and field-matched against a fresh Crossref audit (`results/reference_crossref_audit_20261008.json`); 33 are journal articles and 25 journal records are dated 2022–2026. Full-text selection, publisher-version formatting, and current CAS/SCI verification remain before submission. |

## Release rule

The four statistical gates are closed only for the frozen normalized simulator
grid and always-risk primary cost reference. The periodic-baseline cost gate
remains failed. Physical timing and deployment claims are NO-GO. The current
manuscript is suitable for a simulation-only submission package after final
full-text citation selection and journal-format checks.

Hash-linked completion record: `results/controller_shift_test_final_guard1_revision_full_20261007_jetson_analysis/campaign_c_completion.json`.
