"""Generate the agent-assisted TIRx kernel workflow diagram."""

from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, title, detail, fc, ec, title_size=11.5, detail_size=8.5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.045",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.7,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        x + w / 2,
        y + h * 0.28,
        detail,
        ha="center",
        va="center",
        fontsize=detail_size,
        color="#374151",
        linespacing=1.12,
    )


def arrow(ax, x1, y1, x2, y2, color="#374151", rad=0.0, lw=1.6):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(14.5, 7.6), dpi=180)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    gray = "#f3f4f6"
    gray_edge = "#374151"
    green = "#dcfce7"
    green_edge = "#16a34a"
    blue = "#dbeafe"
    blue_edge = "#2563eb"
    amber = "#fef3c7"
    amber_edge = "#d97706"

    ax.text(
        7.25,
        7.08,
        "Writing TIRx Kernels with Agents",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
    )
    ax.text(
        7.25,
        6.66,
        "Use the agent to turn a vague goal into a verified TIRx contract.",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#4b5563",
    )

    h = 1.12
    w = 3.15
    xs = [0.95, 5.25, 9.55]
    y_top = 4.55
    y_bottom = 2.05

    steps = [
        ("1. Goal", "what you want\nbut not yet how", gray, gray_edge),
        ("2. Ask for Options", "candidate strategies\nand tradeoffs", green, green_edge),
        ("3. Choose Strategy", "human selects\none direction", amber, amber_edge),
        ("4. Write Instruction", "tile path, roles,\nlayouts, barriers", blue, blue_edge),
        ("5. Agent Works", "edit, explain,\nreview, or debug", green, green_edge),
        ("6. Verify + Record", "CUDA, tests,\nbenchmark, lesson", gray, gray_edge),
    ]

    positions = [
        (xs[0], y_top),
        (xs[1], y_top),
        (xs[2], y_top),
        (xs[2], y_bottom),
        (xs[1], y_bottom),
        (xs[0], y_bottom),
    ]

    for (x, y), (title, detail, fc, ec) in zip(positions, steps):
        box(ax, x, y, w, h, title, detail, fc, ec)

    # Snake-shaped reading order.
    arrow(ax, xs[0] + w, y_top + h / 2, xs[1], y_top + h / 2)
    arrow(ax, xs[1] + w, y_top + h / 2, xs[2], y_top + h / 2)
    arrow(ax, xs[2] + w / 2, y_top, xs[2] + w / 2, y_bottom + h)
    arrow(ax, xs[2], y_bottom + h / 2, xs[1] + w, y_bottom + h / 2)
    arrow(ax, xs[1], y_bottom + h / 2, xs[0] + w, y_bottom + h / 2)

    # Feedback loop from verification back to future goals. Route it outside
    # the boxes so the main workflow stays readable.
    loop_x = 0.25
    loop_y = y_top + h / 2
    ax.add_line(
        Line2D(
            [xs[0], loop_x],
            [y_bottom + h / 2, y_bottom + h / 2],
            linewidth=1.35,
            color="#4f46e5",
        )
    )
    ax.add_line(
        Line2D(
            [loop_x, loop_x],
            [y_bottom + h / 2, loop_y],
            linewidth=1.35,
            color="#4f46e5",
        )
    )
    arrow(ax, loop_x, loop_y, xs[0], loop_y, color="#4f46e5", lw=1.35)
    ax.text(
        3.85,
        1.05,
        "The recorded lesson makes the next prompt more precise.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4f46e5",
        fontweight="bold",
    )

    plt.tight_layout(pad=0.5)
    out_path = Path(__file__).resolve().parents[1] / "ai_assisted_tirx_workflow.png"
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
