# LaTeX build audit — 2026-10-07

## Result

The submission source was regenerated from the audited Markdown manuscript and
compiled successfully with local MiKTeX XeLaTeX. The final PDF is 14 pages and
contains all eight audited figures and six inserted tables.

## Reproducible build

From the project root:

```powershell
python scripts/build_submission_tex.py --seed 20261007
Set-Location paper
xelatex -interaction=nonstopmode -halt-on-error -file-line-error physical_communication_submission.tex
xelatex -interaction=nonstopmode -halt-on-error -file-line-error physical_communication_submission.tex
```

The generator records the byte-level source hash in
`paper/physical_communication_submission.provenance.json`. The recorded TeX
SHA-256 is
`2d73e915ce3387ecdcae7f6f74a15c5939aeac72dc39d0c0c1745d9a02819fb2`, which
matches the generated file after writing with explicit LF line endings. The
final PDF SHA-256 is
`6d4b7c4f1d8bb77c957d3bd0a5941eb1345a85892369df19140ab115a5347fc8`.

## Automated checks

| Check | Result |
| --- | --- |
| XeLaTeX exit status | PASS |
| Pages | 13 |
| Figure environments / PDF captions | 8 / 8 |
| Table environments / PDF captions | 6 / 6 |
| LaTeX errors / undefined control sequences | 0 / 0 |
| Undefined references or citations | 0 |
| Overfull boxes | 0 |
| Underfull boxes | 47, non-fatal; concentrated in narrow tables, long hashes, and the reference list |
| PDF fonts | Embedded (all entries report `emb=yes`) |

Long hashes now contain discretionary breakpoints, long artifact names use
`\path`, and the gate-result table uses wrapping paragraph columns. These
changes are implemented in `scripts/build_submission_tex.py`, so a rebuild
does not depend on hand-editing the generated `.tex` file.

The Codex built-in LaTeX compiler was also attempted, but its host returned
`Unable to find standard directories for platform`. This is an environment
failure; the same source compiled successfully with the installed local
XeLaTeX executable above.

## Submission fields still requiring author input

The source contains the supplied author list, affiliations, contribution
statement, no-external-funding statement, and no-competing-interests statement.
The public data/code archive is https://github.com/pengliangtian06-commits/
transparent-risk-sensitive-ris-control with DOI 10.5281/zenodo.23220663. The
PDF continues to state that physical RIS phase-write timing, physical slot
timing, and OTA validation are UNKNOWN/NO-GO; those claims must not be inferred
from the simulator timing record.

## Figure and rebuild update — 2026-10-08

Figure 3 was revised after final-size inspection identified the shared colorbar
overlapping the right-hand heatmap panels. The plotting source now reserves a
dedicated GridSpec column for the colorbar; the four heatmap panels retain
their measured alignment and the colorbar is outside the comparable panel
group. The authoritative manuscript references the vector PDF exports for all
eight figures.

The updated figure bundle was regenerated from the same frozen Campaign C
inputs. Figure 3's final PDF is
`results/figures/campaign_c_20261007/figure03_error_rho_heatmap.pdf` with
SHA-256
`ebf07b791dade00c4dfae8e44697cbfde3fcb5bed3cd5a695c880d584e90ee5f`.
Its text audit found a 5.8 pt minimum glyph and zero below-floor runs; its
collision audit is `PASS` with zero failures and zero warnings; its panel
alignment audit is `PASS` at the 1.5 pt tolerance.

All eight final figure PDFs now report collision-audit `PASS` (0 failures, 0
warnings). Figures 2–8 report alignment `PASS`; the single-panel schematic
recorded `NOT APPLICABLE`. The rebuilt manuscript remains 14 pages, with zero
LaTeX errors, zero undefined references/citations, and zero overfull boxes in
the local XeLaTeX log. The built-in editor compiler was retried and again
returned the host error `Unable to find standard directories for platform`;
local MiKTeX XeLaTeX is the verified compiler for this environment.

The current TeX SHA-256 is
`2d73e915ce3387ecdcae7f6f74a15c5939aeac72dc39d0c0c1745d9a02819fb2`.
The current manuscript PDF SHA-256 is
`6d4b7c4f1d8bb77c957d3bd0a5941eb1345a85892369df19140ab115a5347fc8`.
