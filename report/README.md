# Report

LaTeX source of the research report (IEEEtran journal style) and the built PDF, [report.pdf](report.pdf).

- `main.tex`, `refs.bib`: text and bibliography. The figures are read from `../figures/`.
- `tables/`: the tables of the report; their numbers come from `../results/`.

```bash
make report   # builds main.pdf with latexmk and copies it to report.pdf
```
