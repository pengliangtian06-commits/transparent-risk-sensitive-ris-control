"""Build a two-column LaTeX submission artifact from the audited Markdown draft."""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

FIGURES = {
    1: "figure01_protocol_provenance.pdf",
    2: "figure02_paired_forest_always_risk.pdf",
    3: "figure03_error_rho_heatmap.pdf",
    4: "figure04_absolute_outage_gate.pdf",
    5: "figure05_latency_unknown_boundary.pdf",
    6: "figure06_cost_service_conditional.pdf",
    7: "figure07_freeze_protocol.pdf",
    8: "figure08_campaign_boundary.pdf",
}
FIGURE_ROOT = "../results/figures/campaign_c_20261007"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tables_block() -> str:
    return r"""
\begin{table}[t]
\caption{Normalized simulator and CSI settings.}
\label{tab:settings}
\scriptsize
\begin{tabularx}{\columnwidth}{@{}l l X@{}}
\toprule
Quantity & Frozen setting & Evidence boundary \\
\midrule
RIS elements & 16, 32 & normalized simulator input \\
Users & 2, 4 & matched multi-user streams \\
Phase alphabet & 2-bit & discrete action space \\
Error scale & 0.12, 0.28 & simulated estimation error \\
Temporal correlation & 0.70, 0.98 & AR(1) simulator factor \\
Scenario ensemble & 4; guard 1 & decision surrogate, not calibrated posterior \\
Direct BS--user path & omitted & no physical channel claim \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[t]
\caption{All-slot service and online-work accounting.}
\label{tab:accounting}
\scriptsize
\begin{tabularx}{\columnwidth}{@{}l X@{}}
\toprule
Record & Rule \\
\midrule
Scheduled slot & always remains in the denominator \\
Refusal, timeout or failed delivery & zero payload and one service violation \\
Weak-user outage & condition-level reference user, all scheduled slots \\
Primary work metric & phase-control rate evaluations \\
Truth evaluation & scores an action; excluded from online controller work \\
Initialization & recorded in campaign totals, not silently amortized \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[t]
\caption{Timing evidence boundary.}
\label{tab:timing}
\scriptsize
\begin{tabularx}{\columnwidth}{@{}l X@{}}
\toprule
Quantity & Status in Campaign C \\
\midrule
Estimator/search/validation timing & simulator diagnostic \\
Transfer and synchronization & not a physical measurement \\
RIS phase-write completion & UNKNOWN; no endpoint detected \\
Physical $T_{\mathrm{slot}}$ and overruns & UNKNOWN; no slot trace \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[t]
\caption{Frozen Campaign C protocol and hashes.}
\label{tab:protocol}
\scriptsize
\begin{tabularx}{\columnwidth}{@{}l X@{}}
\toprule
Item & Value \\
\midrule
Rows / conditions / streams & 20,480 / 16 / 640 \\
Seed blocks & 10 held-out; 4 development \\
Episodes / slots & 4 / 8 per stream \\
Bootstrap & 10,000 draws; seed 20261031 \\
Source SHA-256 & \texttt{bb773999...57f4bd} \\
Configuration SHA-256 & \texttt{903fc0a2...f35e2b} \\
Effective-config SHA-256 & \texttt{60f9fc84...c2b4530} \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[t]
\caption{Primary paired Campaign C gate results versus always-risk.}
\label{tab:gates}
\scriptsize
\begin{tabularx}{\columnwidth}{@{}p{0.28\columnwidth} X p{0.14\columnwidth}@{}}
\toprule
Estimand & Bound & Status \\
\midrule
Weak-user outage & upper95 $=0.000$; target $\leq0.05$ & PASS \\
Outage delta & upper95 $=0.000$; target $\leq0.01$ & PASS \\
Goodput relative delta & lower05 $=-0.00615$; target $\geq-0.05$ & PASS \\
Rate-evaluation reduction & lower05 $=0.33398$; target $\geq0.20$ & PASS \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[t]
\caption{Release ledger and evidence boundaries.}
\label{tab:release}
\scriptsize
\begin{tabularx}{\columnwidth}{@{}l X@{}}
\toprule
Item & Release status \\
\midrule
Always-risk statistical gates & PASS on declared normalized grid \\
Periodic-risk / nominal work gate & FAIL; sensitivity only \\
Exact zero-failure outage audit & diagnostic; max upper95 $=0.009318$ \\
Physical phase-write and slot timing & UNKNOWN / NO-GO \\
OTA and deployment feasibility & UNKNOWN / NO-GO \\
Learned-policy superiority & not evaluated or claimed \\
\bottomrule
\end{tabularx}
\end{table}
"""


def build(markdown: Path, output: Path, seed: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pandoc_tex = Path(tmp) / "draft.tex"
        subprocess.run(
            ["pandoc", str(markdown), "--from",
             "markdown+tex_math_single_backslash+raw_tex", "--to", "latex",
             "--standalone", "--output", str(pandoc_tex)],
            check=True,
        )
        raw = pandoc_tex.read_text(encoding="utf-8")
    body = raw[raw.index("\\begin{document}") + len("\\begin{document}"):raw.rindex("\\end{document}")]

    # Move the Markdown title into a proper manuscript title block.
    body = re.sub(
        r"\\section\{Transparent Risk-Sensitive RIS Phase\s+Control Under CSI\s+Uncertainty\}"
        r"\\label\{[^}]+\}\s*", "", body, count=1,
    )
    # Markdown's title consumes the top heading level; promote the manuscript
    # headings back to section/subsection levels and remove literal prefixes.
    body = re.sub(r"\\section\{([1-7])\.\s*([^{}]+)\}", r"\\section{\2}", body)
    body = re.sub(r"\\subsection\{([1-7])\.\s*([^{}]+)\}", r"\\section{\2}", body)
    body = re.sub(r"\\subsubsection\{([1-7])\.\d+\.?\s*([^{}]+)\}", r"\\subsection{\2}", body)

    # Abstract and administrative sections.
    body = body.replace("\\subsection{Abstract}\\label{abstract}", "\\begin{abstract}")
    body = body.replace("\\textbf{Keywords:}", "\\end{abstract}\n\n\\textbf{Keywords:}", 1)
    body = body.replace("\\textbf{Highlights}", "\\paragraph{Highlights}", 1)
    body = re.sub(
        r"\\subsection\{Figure and table placement\s+summary\}.*?"
        r"(?=\\subsection\{Submission statements\})", "", body, flags=re.S,
    )
    body = body.replace("\\subsection{Submission statements}", "\\section*{Submission statements}")
    body = body.replace("\\subsection{References and verification status}", "\\section*{References}")
    body = re.sub(r"\\subsection\{References and verification\s+status\}",
                  r"\\section*{References}", body)
    for name in ("Data and code availability", "Ethics and competing interests",
                 "Funding", "Author contributions", "AI-assisted work disclosure"):
        body = re.sub(
            r"\\subsubsection\{" + re.escape(name).replace(r"\ ", r"\s+") + r"\}",
            lambda _match, name=name: f"\\subsection*{{{name}}}", body,
        )

    # The Markdown table placeholder is replaced by six compact, real tables.
    body = re.sub(r"\\textbf\{Table 1 placeholder\.\}.*?\n\n", "", body, count=1, flags=re.S)
    body = body.replace("\\section{Discussion}", tables_block() + "\n\\section{Discussion}", 1)

    # Make long audit identifiers safe to typeset in a narrow two-column layout.
    # Keep every character of the identifier, while adding legal discretionary
    # breakpoints between fixed-size chunks.
    def break_hash(match: re.Match[str]) -> str:
        value = match.group(1)
        chunks = [value[i:i + 8] for i in range(0, len(value), 8)]
        return r"\texttt{" + r"\allowbreak{}".join(chunks) + "}"

    body = re.sub(r"\\texttt\{([0-9a-fA-F]{64})\}", break_hash, body)

    # xurl's \path command permits line breaks at slashes and underscores.
    # Pandoc emits escaped underscores inside \texttt; restore the path token
    # before handing it to \path so the displayed artifact name remains exact.
    def break_path(match: re.Match[str]) -> str:
        value = match.group(1).replace(r"\_", "_")
        return r"\path{" + value + "}"

    body = re.sub(r"\\texttt\{([^{}\n]*\/[^{}\n]*)\}", break_path, body)

    # Turn the eight audited caption paragraphs into figure environments.
    seen: set[int] = set()
    pattern = re.compile(r"\\textbf\{Figure (\d+)\.\}\s*(.*?)(?=\n\n)", re.S)

    def figure_repl(match: re.Match[str]) -> str:
        number = int(match.group(1))
        caption = re.sub(r"\s+", " ", match.group(2)).strip()
        if number not in FIGURES or number in seen:
            return match.group(0)
        seen.add(number)
        wide = number in {1, 2, 8}
        env = "figure*" if wide else "figure"
        width = "\\textwidth" if wide else "\\columnwidth"
        return (
            f"\\begin{{{env}}}[t]\n\\centering\n"
            f"\\includegraphics[width={width}]{{{FIGURE_ROOT}/{FIGURES[number]}}}\n"
            f"\\caption{{{caption}}}\\label{{fig:campaign-{number:02d}}}\n"
            f"\\end{{{env}}}"
        )

    body = pattern.sub(figure_repl, body)
    missing = sorted(set(FIGURES) - seen)
    if missing:
        raise RuntimeError(f"missing figure captions in Markdown: {missing}")

    preamble = r"""\documentclass[10pt,twocolumn]{article}
\usepackage[margin=0.72in]{geometry}
\usepackage{fontspec}
\setmainfont{Latin Modern Roman}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{microtype}
\usepackage{enumitem}
\usepackage{caption}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\graphicspath{{../results/figures/campaign_c_20261007/}}
\setcounter{secnumdepth}{3}
\setlength{\columnsep}{0.25in}
\setlength{\tabcolsep}{3pt}
\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{2em}
\captionsetup{font=small,labelfont=bf}
\setlist{nosep,leftmargin=*}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\title{Transparent Risk-Sensitive RIS Phase Control Under CSI Uncertainty}
\author{%
\begin{tabular}{c}
Pengliang Tian\textsuperscript{1}, Pengyuan Zhang\textsuperscript{1}, Haoshan Sun\textsuperscript{1}, Changjiang Zhang\textsuperscript{1,*}\\[0.35ex]
\parbox{0.95\textwidth}{\centering\footnotesize \textsuperscript{1}Department of Communication and Information Engineering, Century College, Beijing University of Posts and Telecommunications, Beijing, China}\\[0.35ex]
\footnotesize First-author ORCID: 0009-0003-5941-9797; corresponding author: \href{mailto:zhangchangjiang@ccbupt.cn}{zhangchangjiang@ccbupt.cn}, ORCID: 0009-0002-9936-1377
\end{tabular}
}
\date{}
\begin{document}
\maketitle
"""
    footer = r"""
\end{document}
"""
    document = preamble + body.strip() + footer
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write with explicit LF endings so the provenance hash matches the
    # bytes compiled on every host, including Windows.
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)
    tex_sha256 = sha256(output)
    provenance = output.with_suffix(".provenance.json")
    provenance.write_text(
        __import__("json").dumps({
            "status": "built",
            "build_seed": int(seed),
            "markdown": str(markdown),
            "markdown_sha256": sha256(markdown),
            "tex_sha256": tex_sha256,
            "figures": FIGURES,
            "figure_root": FIGURE_ROOT,
            "two_column": True,
            "tables": 6,
            "compiler_requirement": "XeLaTeX-compatible",
        }, indent=2) + "\n", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", type=Path, default=Path("paper/physical_communication_draft.md"))
    parser.add_argument("--output", type=Path, default=Path("paper/physical_communication_submission.tex"))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    build(args.markdown, args.output, args.seed)
    print(args.output)


if __name__ == "__main__":
    main()
