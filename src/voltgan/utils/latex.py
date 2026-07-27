from __future__ import annotations

from pathlib import Path


class LatexTableWriter:
    """Generic fluent builder for LaTeX tables.

    Accepts pre-formatted strings only — no domain imports, no metric
    formatting logic. Presentation logic for `MetricSet` etc. lives on the
    DTOs via their `.cells()` methods; the CLI orchestrator wires DTOs ->
    cells -> this writer.
    """

    def __init__(
        self,
        out_path: str | Path,
        *,
        float_pos: str = "[H]",
        body_size: str = "footnotesize",
    ):
        self.out_path = Path(out_path)
        self.float_pos = float_pos
        self.body_size = body_size
        self._caption: str | None = None
        self._label: str | None = None
        self._align: str = "l"
        self._lines: list[str] = []

    def caption(self, s: str) -> "LatexTableWriter":
        self._caption = s
        return self

    def label(self, s: str) -> "LatexTableWriter":
        self._label = s
        return self

    def align(self, spec: str) -> "LatexTableWriter":
        self._align = spec
        return self

    def header(self, cells: list[str]) -> "LatexTableWriter":
        """Add the bolded header row (called once, after caption/label/align)."""
        header_cells = " & ".join(rf"\textbf{{{c}}}" for c in cells)
        self._lines.append(header_cells + r" \\")
        return self

    def hline(self) -> "LatexTableWriter":
        self._lines.append(r"\hline")
        return self

    def row(self, cells: list[str]) -> "LatexTableWriter":
        self._lines.append(" & ".join(cells) + r" \\")
        return self

    def rows(self, list_of_cells: list[list[str]]) -> "LatexTableWriter":
        for cells in list_of_cells:
            self.row(cells)
        return self

    def bold_row(self, cells: list[str]) -> "LatexTableWriter":
        """Each cell is already formatted bold by the caller (DTO .cells(bold=True))."""
        self.row(cells)
        return self

    def section(self, subheader: str) -> "LatexTableWriter":
        """Multicolumn subheader row spanning all columns."""
        n = len(self._align)
        self._lines.append(
            rf"\multicolumn{{{n}}}{{c}}{{\textbf{{{subheader}}}}}" + r" \\"
        )
        return self

    def write(self) -> None:
        """Build the full LaTeX skeleton from accumulated state and write to disk."""
        if self._caption is None or self._label is None:
            raise RuntimeError("caption() and label() are required before write()")
        if not self._lines:
            raise RuntimeError("no rows added; nothing to write")

        skeleton = [
            rf"\begin{{table}}{self.float_pos}",
            rf"    \caption{{{self._caption}}}",
            rf"    \label{{{self._label}}}",
            r"    \begin{center}",
            f"        \\{self.body_size}",
            rf"        \begin{{tabular}}{{{self._align}}}",
            r"            \hline",
        ]
        body = []
        for line in self._lines:
            if line == r"\hline":
                body.append(r"            \hline")
            elif line.startswith(r"\multicolumn"):
                body.append(f"            {line}")
            else:
                body.append(f"            {line}")
        skeleton.extend(body)
        skeleton.extend(
            [
                r"            \hline",
                r"        \end{tabular}",
                r"    \end{center}",
                r"\end{table}",
            ]
        )

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text("\n".join(skeleton) + "\n")
        print(f"LaTeX table saved -> {self.out_path}")

