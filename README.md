# Transparent Risk-Sensitive RIS Phase Control Under CSI Uncertainty

This archive accompanies the Physical Communication manuscript by Pengliang Tian, Pengyuan Zhang, Haoshan Sun, and Changjiang Zhang. It contains the normalized simulator, frozen Campaign C summaries and raw records, figure sources/exports, and the LaTeX submission source.

Repository: https://github.com/pengliangtian06-commits/transparent-risk-sensitive-ris-control

## Reproduction

```powershell
python -m pip install -e .
python -m pytest -q
```

The machine-readable Campaign C records are in `results/`; the raw JSONL is gzip-compressed. Configuration files are under `configs/` and every experiment entry point accepts an explicit seed. The manuscript PDF is generated from `paper/physical_communication_submission.tex` with XeLaTeX.

## Evidence boundary

Campaign C has 20,480 rows over 16 declared conditions and 10 held-out seed blocks. Paired bootstrap gates versus always-risk pass on the normalized simulator grid: weak-user outage upper95 = 0.000, outage delta upper95 = 0.000, goodput relative lower05 = -0.00615, and rate-evaluation reduction lower05 = 0.33398. The periodic-baseline work gate remains failed and is retained as sensitivity evidence. No physical RIS phase-write endpoint, measured slot trace, OTA result, calibrated posterior, learned policy, or universal superiority claim is made.

## Authors

Pengliang Tian (ORCID 0009-0003-5941-9797), Pengyuan Zhang, Haoshan Sun, and Changjiang Zhang (corresponding author, ORCID 0009-0002-9936-1377). Department of Communication and Information Engineering, Century College, Beijing University of Posts and Telecommunications.

The authors received no external funding and declare no competing interests.
