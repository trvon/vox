//! Benchmarks for the long-form pipeline pieces that don't need models:
//! streaming resampling as the capture callback drives it, and transcript
//! layout/export at lecture scale.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vox::audio::StreamingResampler;
use vox::transcript::{self, LayoutOptions, Marker, TranscriptSegment};

fn sine(rate: u32, secs: f32) -> Vec<f32> {
    (0..(rate as f32 * secs) as usize)
        .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.5)
        .collect()
}

/// 1 s of audio fed in 10 ms callbacks, like a cpal input stream.
fn bench_streaming_resampler(c: &mut Criterion) {
    let mut g = c.benchmark_group("streaming_resample_1s");
    for rate in [48_000u32, 44_100] {
        let input = sine(rate, 1.0);
        let chunk = rate as usize / 100;
        g.bench_function(format!("{rate}_to_16k_10ms_chunks"), |b| {
            b.iter(|| {
                let mut rs = StreamingResampler::new(rate, 16_000);
                let mut out = Vec::with_capacity(16_100);
                for c in input.chunks(chunk) {
                    rs.process(black_box(c), &mut out);
                }
                rs.flush(&mut out);
                out
            })
        });
    }
    g.finish();
}

/// Include coefficient-table construction/destruction separately from streaming.
fn bench_resampler_setup(c: &mut Criterion) {
    let mut g = c.benchmark_group("resampler_setup_and_drop");
    for rate in [48_000u32, 44_100, 24_000, 16_001] {
        g.bench_function(format!("{rate}_to_16k"), |b| {
            b.iter(|| StreamingResampler::new(black_box(rate), black_box(16_000)))
        });
    }
    g.finish();
}

/// A 3-hour lecture: ~2,000 segments, a marker every 20 minutes.
fn lecture_segments() -> (Vec<TranscriptSegment>, Vec<Marker>) {
    let segs = (0..2_000u64)
        .map(|i| TranscriptSegment {
            seq: i + 1,
            start_ms: i * 5_400,
            end_ms: i * 5_400 + 4_000 + (i % 7) * 100,
            text: "So the key idea here is that the sampling rate bounds what we can represent."
                .into(),
            words: Vec::new(),
        })
        .collect();
    let markers = (0..9u64)
        .map(|i| Marker {
            t_ms: i * 20 * 60_000,
            label: format!("Topic {i}"),
        })
        .collect();
    (segs, markers)
}

fn bench_transcript_export(c: &mut Criterion) {
    let (segs, markers) = lecture_segments();
    let opts = LayoutOptions::default();
    let mut g = c.benchmark_group("transcript_3h");
    g.bench_function("paragraphs_sections_markdown", |b| {
        b.iter(|| {
            let paras = transcript::paragraphs(black_box(&segs), &opts);
            let secs = transcript::sections(paras, &markers, &opts);
            transcript::to_markdown(&transcript::DocMeta::default(), &secs)
        })
    });
    g.bench_function("srt", |b| b.iter(|| transcript::to_srt(black_box(&segs))));
    g.bench_function("collapse_repeats_per_segment", |b| {
        b.iter(|| {
            segs.iter()
                .map(|s| transcript::collapse_repeats(black_box(&s.text), 3).len())
                .sum::<usize>()
        })
    });
    g.finish();

    let reference = segs[..200]
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let hypothesis = reference.replace("sampling", "sample");
    c.bench_function("wer_2800_words", |b| {
        b.iter(|| transcript::word_error_rate(black_box(&reference), black_box(&hypothesis)))
    });
}

criterion_group!(
    benches,
    bench_streaming_resampler,
    bench_resampler_setup,
    bench_transcript_export
);
criterion_main!(benches);
