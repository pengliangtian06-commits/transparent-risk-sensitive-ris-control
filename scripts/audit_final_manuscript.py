"""Deterministic final manuscript consistency audit."""
from __future__ import annotations

import json
import re
from pathlib import Path


def main() -> None:
    paper = Path("paper/physical_communication_draft.md")
    text = paper.read_text(encoding="utf-8")
    body, refs = text.split("## References and verification status", 1)
    gate = json.loads(Path("results/controller_shift_test_final_guard1_revision_full_20261007_jetson_analysis/gate_audit.json").read_text(encoding="utf-8"))
    completion = json.loads(Path("results/controller_shift_test_final_guard1_revision_full_20261007_jetson_analysis/campaign_c_completion.json").read_text(encoding="utf-8"))
    bib = json.loads(Path("results/reference_metadata_final_20261007.json").read_text(encoding="utf-8"))
    ref_audit = json.loads(Path("results/reference_crossref_audit_20261008.json").read_text(encoding="utf-8"))
    fig = json.loads(Path("results/figures/campaign_c_20261007/figure_provenance.json").read_text(encoding="utf-8"))
    always = gate["paired_comparisons"]["always_risk"]
    checks = {
        "body_words_6000_7000": 6000 <= len(re.findall(r"\b[\w-]+\b", body)) <= 7000,
        "references_30_40": 30 <= sum(bool(re.match(r"^\[\d+\]", line)) for line in refs.splitlines()) <= 40,
        "figures_8": len(fig["figures"]) == 8,
        "campaign_rows_complete": completion["planned_rows"] == completion["completed_rows"] == 20480,
        "campaign_conditions": completion["planned_conditions"] == 640,
        "absolute_gate_pass": gate["absolute_weak_user_outage"]["status"] == "PASS",
        "always_service_pass": always["service_guardrail"]["status"] == "PASS",
        "always_work_pass": always["work_reduction_guardrail"]["status"] == "PASS",
        "periodic_work_failure_retained": gate["paired_comparisons"]["periodic_risk"]["work_reduction_guardrail"]["status"] == "FAIL" and gate["paired_comparisons"]["periodic_nominal"]["work_reduction_guardrail"]["status"] == "FAIL",
        "physical_timing_unknown": gate["latency"]["status"] == "UNKNOWN" and "UNKNOWN" in text,
        "no_pending_markers": "[PENDING" not in text,
        "metadata_records_34": bib["unique_records"] == 34 and bib["metadata_failures"] == 0,
        "reference_crossref_pass": ref_audit["status"] == "PASS" and ref_audit["record_count"] == 34 and ref_audit["counts"]["verified"] == 34,
        "no_stale_campaign_count": "12,288" not in body and "controller_validation_full_20261007" not in body,
        "source_hash_linked": completion["source_sha256"] == gate["source_sha256"],
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "body_words": len(re.findall(r"\b[\w-]+\b", body)),
              "reference_count": sum(bool(re.match(r"^\[\d+\]", line)) for line in refs.splitlines()),
              "figure_count": len(fig["figures"]),
              "campaign_rows": completion["completed_rows"],
              "hardware_timing": gate["latency"]["status"]}
    out = Path("docs/FINAL_MANUSCRIPT_AUDIT_CAMPAIGN_C_20261007.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
