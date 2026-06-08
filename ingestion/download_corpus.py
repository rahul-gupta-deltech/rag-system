"""
download_corpus.py — Day 2: fetch ~40 Kubernetes + GCP concept pages and save
as plain-text .txt files under corpus/.

Usage:
    python download_corpus.py [--out corpus] [--max 50]

Each file is saved as:
    corpus/<slug>.txt

The ingestion pipeline (ingest.py) reads all .txt files in that directory.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup  # pip install beautifulsoup4 lxml

# ---------------------------------------------------------------------------
# Corpus seed URLs — public Kubernetes + GCP documentation pages
# ---------------------------------------------------------------------------
KUBERNETES_URLS = [
    "https://kubernetes.io/docs/concepts/overview/what-is-kubernetes/",
    "https://kubernetes.io/docs/concepts/overview/components/",
    "https://kubernetes.io/docs/concepts/workloads/pods/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/",
    "https://kubernetes.io/docs/concepts/services-networking/service/",
    "https://kubernetes.io/docs/concepts/services-networking/ingress/",
    "https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/",
    "https://kubernetes.io/docs/concepts/storage/persistent-volumes/",
    "https://kubernetes.io/docs/concepts/storage/storage-classes/",
    "https://kubernetes.io/docs/concepts/configuration/configmap/",
    "https://kubernetes.io/docs/concepts/configuration/secret/",
    "https://kubernetes.io/docs/concepts/configuration/resource-management-for-pods-and-containers/",
    "https://kubernetes.io/docs/concepts/security/pod-security-standards/",
    "https://kubernetes.io/docs/concepts/security/rbac-good-practices/",
    "https://kubernetes.io/docs/concepts/cluster-administration/logging/",
    "https://kubernetes.io/docs/concepts/cluster-administration/networking/",
    "https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/",
    "https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/",
    "https://kubernetes.io/docs/concepts/extend-kubernetes/operator/",
    "https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/",
]

GCP_URLS = [
    "https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform",
    "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/overview",
    "https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    "https://cloud.google.com/vertex-ai/generative-ai/docs/vector-search/overview",
    "https://cloud.google.com/vertex-ai/generative-ai/docs/rag/rag-overview",
    "https://cloud.google.com/run/docs/overview/what-is-cloud-run",
    "https://cloud.google.com/run/docs/about-instance-autoscaling",
    "https://cloud.google.com/run/docs/securing/service-identity",
    "https://cloud.google.com/sql/docs/postgres/overview",
    "https://cloud.google.com/sql/docs/postgres/connect-overview",
    "https://cloud.google.com/storage/docs/introduction",
    "https://cloud.google.com/secret-manager/docs/overview",
    "https://cloud.google.com/logging/docs/overview",
    "https://cloud.google.com/trace/docs/overview",
    "https://cloud.google.com/monitoring/docs/overview",
    "https://cloud.google.com/iam/docs/overview",
    "https://cloud.google.com/iap/docs/concepts-overview",
    "https://cloud.google.com/pubsub/docs/overview",
    "https://cloud.google.com/tasks/docs/concepts",
    "https://cloud.google.com/dlp/docs/concepts-infotypes-reference",
]

ALL_URLS = KUBERNETES_URLS + GCP_URLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEADERS = {"User-Agent": "Mozilla/5.0 (RAG-corpus-builder/1.0; research use)"}


def slugify(url: str) -> str:
    """Turn a URL into a safe filename slug."""
    path = re.sub(r"https?://[^/]+", "", url).strip("/")
    slug = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")
    return slug[:120]  # cap length


def fetch_text(url: str, timeout: int = 15) -> str | None:
    """Fetch a URL and return clean body text, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  WARN  fetch failed: {exc}", file=sys.stderr)
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove nav, header, footer, sidebar noise
    for tag in soup.select(
        "nav, header, footer, aside, script, style, "
        ".navbar, .sidebar, .toc, .feedback, .breadcrumbs, "
        "[class*='nav'], [class*='menu'], [class*='footer']"
    ):
        tag.decompose()

    # Prefer <main> or <article> if present
    main = soup.find("main") or soup.find("article") or soup.find("body")
    if main is None:
        return None

    # Collapse whitespace
    lines = [line.strip() for line in main.get_text(separator="\n").splitlines()]
    text = "\n".join(line for line in lines if line)
    return text if len(text) > 200 else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download corpus documents")
    parser.add_argument("--out", default="corpus", help="Output directory")
    parser.add_argument("--max", type=int, default=50, help="Max pages to download")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    urls = ALL_URLS[: args.max]
    saved = 0
    failed = 0

    print(f"Downloading up to {len(urls)} pages → {out_dir}/")
    print("-" * 60)

    for i, url in enumerate(urls, 1):
        slug = slugify(url)
        out_path = out_dir / f"{slug}.txt"

        if out_path.exists():
            print(f"  [{i:02d}/{len(urls)}] SKIP (exists)  {slug}.txt")
            saved += 1
            continue

        print(f"  [{i:02d}/{len(urls)}] GET  {url}")
        text = fetch_text(url)

        if text:
            out_path.write_text(f"SOURCE: {url}\n\n{text}", encoding="utf-8")
            print(f"           → saved {out_path.name}  ({len(text):,} chars)")
            saved += 1
        else:
            print(f"           → FAILED / empty")
            failed += 1

        time.sleep(0.4)  # be polite

    print("-" * 60)
    print(f"Done. {saved} saved, {failed} failed. Corpus: {out_dir}/")


if __name__ == "__main__":
    main()
