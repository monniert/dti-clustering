from matplotlib import pyplot as plt
import pandas as pd


def plot_lines(df, columns, title, figsize=(10, 5.625), drop_na=True, style=None, unit_yaxis=False, lw=2):
    if not isinstance(columns, (list, tuple)):
        columns = [columns]

    fig, ax = plt.subplots(figsize=figsize)
    for y in columns:
        if drop_na:
            s = df[y].dropna()
            if len(s) > 0:
                s.plot(ax=ax, linewidth=lw, style=style)
        else:
            df[y].plot(ax=ax, linewidth=lw, style=style)
    if unit_yaxis:
        ax.set_ylim((0, 1))
        ax.set_yticks([k/10 for k in range(11)])
    ax.grid(axis="x", which='both', color='0.5', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', which='major', color='0.5', linestyle='-', linewidth=0.5)
    ax.legend(framealpha=1, edgecolor='0.3', fancybox=False)
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()

    return fig


def plot_bar(df, title, figsize=(10, 5.625), unit_yaxis=False):
    assert isinstance(df, pd.Series)
    fig, ax = plt.subplots(figsize=figsize)
    df.plot(kind='bar', ax=ax, edgecolor='k', width=0.8, rot=0, linewidth=1)
    if unit_yaxis:
        ax.set_ylim((0, 1))
        ax.set_yticks([k/10 for k in range(11)])
    ax.grid(axis="x", which='both', color='0.5', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', which='major', color='0.5', linestyle='-', linewidth=0.5)
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()

    return fig
