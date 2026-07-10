from __future__ import annotations


def print_box_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    alignments: list[str] | None = None,
    footer_row: list[str] | None = None,
    min_widths: list[int] | None = None,
) -> None:
    n_cols = len(headers)
    if alignments is None:
        alignments = ["left"] * n_cols
    if len(alignments) != n_cols:
        raise ValueError("alignments length must match headers length")

    all_rows = [headers, *rows]
    if footer_row is not None:
        if len(footer_row) != n_cols:
            raise ValueError("footer_row length must match headers length")
        all_rows = [*all_rows, footer_row]

    widths = [0] * n_cols
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    if min_widths is not None:
        for i in range(n_cols):
            widths[i] = max(widths[i], min_widths[i])

    def _fmt(cell: str, idx: int) -> str:
        w = widths[idx]
        if alignments[idx] == "right":
            return f" {cell:>{w}} "
        if alignments[idx] == "center":
            pad = w - len(cell)
            left = pad // 2
            right = pad - left
            return f" {' ' * left}{cell}{' ' * right} "
        return f" {cell:<{w}} "

    border_top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    border_mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    border_bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    def _row_str(row: list[str]) -> str:
        return "│" + "│".join(_fmt(cell, i) for i, cell in enumerate(row)) + "│"

    print(border_top)
    print(_row_str(headers))

    if rows:
        print(border_mid)
        for row in rows:
            print(_row_str(row))

    if footer_row is not None:
        print(border_mid)
        print(_row_str(footer_row))

    print(border_bot)