#!/usr/bin/env bash
# Copyright: lituus-io, all rights reserved.
# Author: terekete <spicyzhug@gmail.com>
#
# Run link-r's CI gates locally, in CI order: fmt, clippy, feature matrix,
# docs, every test suite (incl. the 0.2.0 golden fixture), the Python wheel
# and its pytest suite, link-r-objstore (memory, local directory and the GCS
# emulator when Docker is available), benches in test mode, short fuzz runs,
# supply-chain checks and a semver check against the previous tag.
# Expects the ../objstore-r sibling. Never touches the cloud.
set -euo pipefail
cd "$(dirname "$0")/.."
export RUSTFLAGS="${RUSTFLAGS:--D warnings}"
FUZZ_SECS="${FUZZ_SECS:-20}"
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33mskip: %s\033[0m\n' "$*"; }
have() { cargo "$1" --version >/dev/null 2>&1; }

step fmt; cargo fmt --check; (cd objstore && cargo fmt --check)
step clippy; cargo clippy -q --all-targets; cargo clippy -q --all-features --all-targets
step "feature matrix"
for f in "" hash markdown html http crawl fs github oauth robots quant parallel "crawl,github,quant"; do
  if [ -z "$f" ]; then cargo check -q --no-default-features --all-targets; else cargo check -q --no-default-features --features "$f" --lib; fi
done
step docs; RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps
step tests; cargo test -q; cargo test -q --features "github,oauth,robots,quant,parallel"
step "benches (test mode)"; cargo bench -q -- --test >/dev/null

step "python wheel + pytest"
if command -v python3 >/dev/null; then
  (cd bindings/python && { [ -d .venv ] || python3 -m venv .venv; } && . .venv/bin/activate \
    && pip install -q maturin pytest && cargo fmt --check && maturin develop -q --release && pytest -q tests/)
else warn "python3 not found"; fi

started=""
if [ -z "${OBJSTORE_FAKE_GCS:-}" ] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker rm -f linkr-ci-fakegcs >/dev/null 2>&1 || true
  docker run -d --name linkr-ci-fakegcs -p 4446:4446 fsouza/fake-gcs-server \
    -scheme http -port 4446 -public-host 127.0.0.1:4446 -external-url http://127.0.0.1:4446 >/dev/null
  started=1
  for _ in $(seq 1 30); do curl -sf "http://127.0.0.1:4446/storage/v1/b?project=test" >/dev/null && break; sleep 1; done
  export OBJSTORE_FAKE_GCS=http://127.0.0.1:4446
fi
trap '[ -n "$started" ] && docker rm -f linkr-ci-fakegcs >/dev/null 2>&1 || true' EXIT
step "link-r-objstore"
[ -n "${OBJSTORE_FAKE_GCS:-}" ] || warn "no Docker: the GCS emulator test is skipped"
(cd objstore && cargo clippy -q --all-features --all-targets && cargo test -q --all-features)

step "fuzz (${FUZZ_SECS}s per target)"
if have fuzz; then
  for t in fuzz_html_links fuzz_index_loader fuzz_url_canonicalize fuzz_index_from_bytes; do
    (cd fuzz && cargo fuzz run "$t" -- -max_total_time="$FUZZ_SECS" 2>&1 | tail -1)
  done
else warn "cargo-fuzz not installed"; fi

step "supply chain"
if have deny; then cargo deny check; (cd objstore && cargo deny check --config ../deny.toml); else warn "cargo-deny not installed"; fi
if have audit; then cargo audit; else warn "cargo-audit not installed"; fi

step "semver vs previous tag"
if have semver-checks; then
  cur="v$(cargo metadata --no-deps --format-version 1 | python3 -c 'import json,sys;print(json.load(sys.stdin)["packages"][0]["version"])')"
  prev=$(git tag --sort=-v:refname --merged HEAD | grep -v "^$cur$" | head -1)
  base="$(cd .. && pwd)/.link-r-semver-baseline"
  git worktree remove --force "$base" >/dev/null 2>&1 || rm -rf "$base"
  git worktree add -q --detach "$base" "$prev"
  cargo semver-checks --baseline-root "$base" --release-type minor
  cargo semver-checks --baseline-root "$base" --all-features --release-type minor
  git worktree remove --force "$base"
else warn "cargo-semver-checks not installed"; fi

step "all gates passed"
