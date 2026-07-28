from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


class LatexFormatter:
    """Utility class encapsulating all LaTeX string formatting operations."""

    @staticmethod
    def bold_cells(cells: list[str]) -> list[str]:
        """Applies LaTeX bolding to cells, respecting math mode."""
        formatted = []
        for cell in cells:
            if cell == "--":
                formatted.append("--")
            elif cell.startswith("$") and cell.endswith("$"):
                # Unwrap math mode and apply \mathbf
                inner = cell[1:-1]
                formatted.append(rf"$\mathbf{{{inner}}}$")
            else:
                formatted.append(rf"\textbf{{{cell}}}")
        return formatted


class RowItem(ABC):
    """Abstract base class contract for all table components."""

    @abstractmethod
    def render(self, num_cols: int) -> str:
        pass


@dataclass
class TableRow(RowItem):
    cells: list[str]
    bold: bool = False

    def render(self, num_cols: int) -> str:
        cells_to_render = (
            LatexFormatter.bold_cells(self.cells) if self.bold else self.cells
        )
        return " & ".join(cells_to_render) + r" \\"


@dataclass
class SectionHeader(RowItem):
    title: str

    def render(self, num_cols: int) -> str:
        return rf"\multicolumn{{{num_cols}}}{{c}}{{\textbf{{{self.title}}}}}" + r" \\"


@dataclass
class HLine(RowItem):
    def render(self, num_cols: int = 0) -> str:
        return r"\hline"


@dataclass
class LatexTable:
    """Declarative schema for generating a LaTeX table using standard dataclasses."""

    out_path: Path
    caption: str
    label: str
    headers: list[str]
    align: str = "l"
    float_pos: str = "[H]"
    body_size: str = "footnotesize"

    items: list[RowItem] = field(default_factory=list)

    @property
    def num_columns(self) -> int:
        return len(self.align)

    def render_content(self) -> str:
        lines = [
            rf"\begin{{table}}{self.float_pos}",
            rf"    \caption{{{self.caption}}}",
            rf"    \label{{{self.label}}}",
            r"    \begin{center}",
            f"        \\{self.body_size}",
            rf"        \begin{{tabular}}{{{self.align}}}",
            r"            \hline",
            "            "
            + " & ".join(LatexFormatter.bold_cells(self.headers))
            + r" \\",
            r"            \hline",
        ]

        for item in self.items:
            # Polymorphic render call
            lines.append(f"            {item.render(self.num_columns)}")

        lines.extend(
            [
                r"            \hline",
                r"        \end{tabular}",
                r"    \end{center}",
                r"\end{table}",
            ]
        )
        return "\n".join(lines) + "\n"

    def write(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(self.render_content())
        print(f"LaTeX table saved -> {self.out_path}")
