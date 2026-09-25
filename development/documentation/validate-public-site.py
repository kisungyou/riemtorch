#!/usr/bin/env python3
"""Validate generated pkgdown links, public-example plots, and local downloads.

Reads docs/ and website sources without changing them. Writes only the JSON
verification report selected by --output. Uses Python's standard library.
"""

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPO = Path(__file__).resolve().parents[2]
ARTICLES = (
    "example-penguin-subspace",
    "example-stackloss-regression",
    "example-airquality-completion",
    "example-covariance-means",
    "example-tree-constraints",
    "example-mtcars-proximal",
)
DOWNLOADS = (
    "penguins.csv",
    "NOTICE-penguins.txt",
    "extract-penguins.R",
    "example-runtimes.csv",
)
PENGUINS_SHA256 = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.anchors = set()
        self.references = []
        self.plots = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.anchors.add(attributes["id"])
        if tag == "a" and attributes.get("name"):
            self.anchors.add(attributes["name"])
        for attribute in ("href", "src"):
            if attributes.get(attribute):
                self.references.append(
                    (attributes[attribute], self.getpos()[0], attribute)
                )
        if tag == "img" and (
            "r-plt" in attributes.get("class", "").split()
            or "/figure-" in attributes.get("src", "")
        ):
            self.plots.append(
                {"src": attributes.get("src", ""),
                 "alt": attributes.get("alt", ""),
                 "line": self.getpos()[0]}
            )

    handle_startendtag = handle_starttag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=REPO / "docs")
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "development/documentation/public-data/local-link-check.json",
    )
    args = parser.parse_args()
    site = args.site.resolve()
    pages = {
        path.resolve(): Page(path.read_text(encoding="utf-8"))
        for path in site.rglob("*.html")
    }
    errors = []
    checked = 0
    checked_fragments = 0
    linked_targets = set()

    def relative(path):
        try:
            return str(path.relative_to(site))
        except ValueError:
            return str(path)

    def error(kind, source, line=None, target=None, **details):
        record = {"kind": kind, "source": relative(source)}
        if line is not None:
            record["line"] = line
        if target is not None:
            record["target"] = target
        record.update(details)
        errors.append(record)

    def check_link(source, raw, line, attribute):
        nonlocal checked, checked_fragments
        url = urlsplit(raw.strip())
        if url.scheme not in ("", "http", "https"):
            return
        if url.netloc and url.hostname not in ("localhost", "127.0.0.1", "::1"):
            return
        if url.scheme and not url.netloc:
            return
        path = unquote(url.path)
        target = (site / path.lstrip("/")) if path.startswith("/") else (
            source.parent / path if path else source
        )
        if target.is_dir():
            target = target / "index.html"
        target = target.resolve()
        checked += 1
        linked_targets.add(target)
        try:
            target.relative_to(site)
        except ValueError:
            error("target_outside_site", source, line, raw, attribute=attribute)
            return
        if not target.is_file():
            error("missing_target", source, line, raw, attribute=attribute)
            return
        fragment = unquote(url.fragment)
        # Text fragments target browser-generated text ranges rather than IDs.
        fragment = fragment.split(":~:text=", 1)[0]
        if fragment and target.suffix.lower() == ".html":
            checked_fragments += 1
            if target not in pages:
                pages[target] = Page(target.read_text(encoding="utf-8"))
            if fragment not in pages[target].anchors:
                error("missing_fragment", source, line, raw, attribute=attribute)

    if not pages:
        error("missing_html_pages", site)
    for path, page in list(pages.items()):
        for raw, line, attribute in page.references:
            check_link(path, raw, line, attribute)

    css_references = 0
    css_pattern = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)
    for path in site.rglob("*.css"):
        text = path.read_text(encoding="utf-8")
        for match in css_pattern.finditer(text):
            raw = match.group(2)
            if raw.strip().startswith("data:"):
                continue
            css_references += 1
            check_link(path.resolve(), raw, text.count("\n", 0, match.start()) + 1,
                       "CSS url")

    plot_checks = {}
    for article in ARTICLES:
        path = site / "articles" / (article + ".html")
        page = pages.get(path.resolve())
        if page is None:
            error("missing_public_article", path)
            continue
        if not 1 <= len(page.plots) <= 2:
            error("unexpected_plot_count", path, count=len(page.plots))
        valid_alt = 0
        for plot in page.plots:
            alt = plot["alt"].strip()
            if len(alt) < 20 or alt.lower().startswith("plot of chunk"):
                error("missing_descriptive_plot_alt", path, plot["line"], plot["src"])
            else:
                valid_alt += 1
        plot_checks[article] = {"plots": len(page.plots), "descriptive_alt": valid_alt}

    downloads = {}
    for filename in DOWNLOADS:
        path = site / "articles/data" / filename
        if not path.is_file() or path.stat().st_size == 0:
            error("missing_download", path)
            continue
        linked = path.resolve() in linked_targets
        if not linked:
            error("unlinked_download", path)
        downloads[filename] = {"bytes": path.stat().st_size, "linked": linked}
    csv_path = site / "articles/data/example-runtimes.csv"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as stream:
            runtime_rows = list(csv.DictReader(stream))
        downloads.setdefault("example-runtimes.csv", {})["rows"] = len(runtime_rows)
        if len(runtime_rows) != 10:
            error("unexpected_gallery_runtime_count", csv_path, count=len(runtime_rows))

    source_csv = REPO / "vignettes/articles/data/penguins.csv"
    built_csv = site / "articles/data/penguins.csv"
    source_hash = hashlib.sha256(source_csv.read_bytes()).hexdigest()
    built_hash = hashlib.sha256(built_csv.read_bytes()).hexdigest() if built_csv.is_file() else None
    hashes_match = source_hash == built_hash == PENGUINS_SHA256
    if not hashes_match:
        error("penguins_checksum_mismatch", built_csv,
              source_sha256=source_hash, built_sha256=built_hash)

    evidence = {
        "status": "pass" if not errors else "fail",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "html_pages": len(pages),
        "local_links_and_assets": checked,
        "html_fragment_links": checked_fragments,
        "css_asset_references": css_references,
        "public_article_plots": plot_checks,
        "downloads": downloads,
        "penguins_sha256": source_hash,
        "penguins_source_and_site_match": hashes_match,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "html_pages": len(pages),
                      "local_links_and_assets": checked, "errors": errors,
                      "report": str(args.output)}))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
