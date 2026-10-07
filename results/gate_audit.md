# Controller Shift Gate Audit

Analysis seed: `20261031`; bootstrap draws: `10000`; unit: `complete independent seed block`.
Config SHA-256: `903fc0a2bbb79f083251d0b905a6dcd1f43dd4986e3a2b4dab40214233f35e2b`; effective config SHA-256: `60f9fc84b04fbe22a02a54353f370581e106cb80cf20a4762c482d873c2b4530`.

## Gate results

- Absolute weak-user outage: **PASS**; maximum one-sided 95% upper bound `0.000000` against threshold `0.05`.
- Compared with `periodic_risk`: service guardrail **PASS**; work-reduction guardrail **FAIL**; minimum lower 95% work reduction `-0.331982`.
- Compared with `always_risk`: service guardrail **PASS**; work-reduction guardrail **PASS**; minimum lower 95% work reduction `0.333984`.
- Compared with `periodic_nominal`: service guardrail **PASS**; work-reduction guardrail **FAIL**; minimum lower 95% work reduction `-0.332031`.
- Physical latency gate: **UNKNOWN**; simulated total-latency P95 maximum `388.083 ms` is diagnostic only.
- Baseline tuning/freeze: **PASS**.

Release decision: **NO-GO / not yet evidenced**.

The report does not support a physical real-time, OTA, or universal-service claim.
