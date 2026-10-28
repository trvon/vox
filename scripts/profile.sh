#!/usr/bin/env bash
# Profiling entry points for vox.
#
#   scripts/profile.sh rtf    [audio.wav]   # end-to-end RTF + WER on a fixture (release build)
#   scripts/profile.sh perf   [audio.wav]   # perf record of `vox transcribe`, top symbols
#   scripts/profile.sh samply [audio.wav]   # samply (Firefox profiler UI), if installed
#   scripts/profile.sh bench  [filter]      # Criterion micro-benchmarks
#
# With no audio argument, a ~75 s synthetic lecture is generated once with
# Kokoro (examples/synth_lecture.rs) into target/profile-fixtures/, together
# with its reference text, so WER and RTF are comparable across runs.
# Needs models (`vox download-models`).
set -euo pipefail

cd "$(dirname "$0")/.."
mode="${1:-rtf}"
fixtures=target/profile-fixtures
mkdir -p "$fixtures"
export VOX_RECORDINGS_DIR="$fixtures/recordings"

fixture() {
    if [[ -n "${1:-}" ]]; then
        echo "$1"
        return
    fi
    local wav="$fixtures/lecture.wav"
    if [[ ! -f "$wav" ]]; then
        cargo run --quiet --release --example synth_lecture -- "$wav" >&2
    fi
    echo "$wav"
}

reference_args() {
    local ref="${1%.*}.txt"
    [[ -f "$ref" ]] && echo "--reference $ref"
}

case "$mode" in
rtf)
    cargo build --quiet --release
    wav="$(fixture "${2:-}")"
    # shellcheck disable=SC2046
    ./target/release/vox transcribe "$wav" --out "$fixtures/last.md" $(reference_args "$wav")
    ;;
perf)
    command -v perf >/dev/null || { echo "perf not installed" >&2; exit 1; }
    cargo build --quiet --profile profiling
    wav="$(fixture "${2:-}")"
    # task-clock works without root and on hybrid (P/E-core) CPUs where the
    # default `cycles` event fails to open. Call graphs need
    # kernel.perf_event_paranoid <= 1; fall back to a flat profile otherwise.
    graph=(--call-graph dwarf)
    if ! perf record -e task-clock "${graph[@]}" -o /dev/null -- true >/dev/null 2>&1; then
        echo "note: call graphs unavailable (sudo sysctl kernel.perf_event_paranoid=1); flat profile" >&2
        graph=()
    fi
    perf record -e task-clock -F 999 "${graph[@]}" -o "$fixtures/perf.data" \
        ./target/profiling/vox transcribe "$wav" --out /dev/null
    perf report -i "$fixtures/perf.data" --no-children --sort dso,symbol --stdio 2>/dev/null | head -60
    echo "Full report: perf report -i $fixtures/perf.data"
    ;;
samply)
    command -v samply >/dev/null || { echo "install with: cargo install samply" >&2; exit 1; }
    cargo build --quiet --profile profiling
    wav="$(fixture "${2:-}")"
    samply record ./target/profiling/vox transcribe "$wav" --out /dev/null
    ;;
bench)
    cargo bench -- "${2:-}"
    ;;
*)
    sed -n '2,12p' "$0"
    exit 1
    ;;
esac
