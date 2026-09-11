"""Presentation copies of `eye_pre_flash.counts.census`'s H/S heatmaps.

The chart itself (parula-free diverging blue/orange fill, `H/S` cell text,
red-bold thin marking) is unchanged -- only the title shrinks from three
explanatory lines to one, since the legend it re-derives (blue/orange/grey,
red-bold meaning) belongs in speaker notes, not the slide.

One addition here that `counts.census.run` doesn't make: for a scope wider
than `publication` (`all`/`allplus`/`top_ten`), each of the four publication
sessions' rows gets a plain black box, so a slide reader can see which rows
are the publication subset without cross-referencing session names.

Default scope is `top_ten` -- the 10 highest-`snr_auc` sessions per monkey
(`eye_pre_flash.label_sources.SOURCE_SCOPES["svm"]`), applied to *every* label
source's plot here, not just svm's: both the dendro and svm heatmaps show
exactly the same ten sessions, unvetted, rather than each source picking its
own pool (dendro's own scopes vet down to 4/monkey by anchor agreement, which
is a different ten-vs-four question than "same sessions, different labels").

Usage:
    uv run python -m eye_pre_flash.counts.build
    uv run python -m eye_pre_flash.counts.build --source dendro --monkey Nielsen
    uv run python -m eye_pre_flash.counts.build --scope allplus

Writes eye_pre_flash/pres/counts/<source>/hs_<monkey>_<scope>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eye_pre_flash.classifier.labels import PUBLICATION_SESSIONS
from eye_pre_flash.label_sources import SOURCES, source_scope_sessions
from eye_pre_flash.counts.census import census_counts, plot_census, source_sessions

OUT_ROOT = Path(__file__).resolve().parent
TOP_TEN_SOURCE = "svm"  # top_ten is only defined for this source; see module docstring


def _sessions_for(source, scope, monkey):
    """`sessions` for `(source, scope)` -- `top_ten` is one fixed pool for every source."""
    if scope == "top_ten":
        by_monkey, _missing, _dropped = source_scope_sessions(TOP_TEN_SOURCE, "top_ten")
    else:
        by_monkey, _missing, _dropped = source_sessions(source, scope)
    return by_monkey.get(monkey, ())


def build(monkey, scope, source):
    sessions = _sessions_for(source, scope, monkey)
    if not sessions:
        raise SystemExit(f"no {source} labels for {monkey} in scope {scope}")
    sessions, counts = census_counts(monkey, sessions, source)
    title = f"{monkey} — {source} labels, scope {scope}"
    # Boxing is a no-op (every row would be boxed) when the scope already
    # *is* just the publication sessions, so skip it there.
    highlight = PUBLICATION_SESSIONS if scope != "publication" else None
    return plot_census(
        monkey, scope, source, sessions, counts,
        title=title, out_root=OUT_ROOT, highlight_sessions=highlight,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", nargs="*", default=list(SOURCES), choices=SOURCES)
    parser.add_argument("--monkey", nargs="*", default=["Faure", "Nielsen"], choices=("Faure", "Nielsen"))
    parser.add_argument(
        "--scope", default="top_ten", choices=("publication", "all", "allplus", "top_ten")
    )
    args = parser.parse_args()

    for monkey in args.monkey:
        for source in args.source:
            build(monkey, args.scope, source)


if __name__ == "__main__":
    main()
