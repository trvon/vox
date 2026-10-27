//! Property-based invariant tests for the DSP and transcript layers.
//!
//! These pin down behaviour that must hold for *any* input, not just the
//! hand-picked cases in the unit tests: resampler length/chunking/DC/aliasing,
//! filter stability, gate/normalizer contracts, sentence splitting, and the
//! transcript layout that exports are built from.

use proptest::prelude::*;
use vox::audio::{self, StreamingResampler};
use vox::transcript::{self, LayoutOptions, Marker, TranscriptSegment};
use vox::tts::split_sentences;

const RATES: &[u32] = &[8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000];

fn rate() -> impl Strategy<Value = u32> {
    prop::sample::select(RATES)
}

fn signal(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1.0f32..1.0, 0..max_len)
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt()
}

fn tone(freq: f32, rate: u32, secs: f32) -> Vec<f32> {
    (0..(rate as f32 * secs) as usize)
        .map(|i| (i as f32 * freq * std::f32::consts::TAU / rate as f32).sin())
        .collect()
}

// --- Resampler ---------------------------------------------------------------

proptest! {
    #[test]
    fn resample_length_is_exact(x in signal(5_000), from in rate(), to in rate()) {
        let y = audio::resample(&x, from, to);
        let expected = (x.len() as u64 * to as u64).div_ceil(from as u64) as usize;
        // Rates are reduced by gcd internally; the ceil is identical.
        prop_assert_eq!(y.len(), expected);
    }

    #[test]
    fn resample_is_chunk_invariant(
        x in signal(4_000),
        from in rate(),
        to in rate(),
        chunks in prop::collection::vec(1usize..700, 1..40),
    ) {
        let one_shot = audio::resample(&x, from, to);
        let mut rs = StreamingResampler::new(from, to);
        let mut streamed = Vec::new();
        let mut rest = &x[..];
        let mut i = 0;
        while !rest.is_empty() {
            let n = chunks[i % chunks.len()].min(rest.len());
            rs.process(&rest[..n], &mut streamed);
            rest = &rest[n..];
            i += 1;
        }
        rs.flush(&mut streamed);
        // Bit-exact: same absolute positions, same taps, same arithmetic.
        prop_assert_eq!(one_shot, streamed);
    }

    #[test]
    fn resample_preserves_dc(c in -1.0f32..1.0, n in 1usize..3_000, from in rate(), to in rate()) {
        let y = audio::resample(&vec![c; n], from, to);
        for v in y {
            prop_assert!((v - c).abs() < 1e-4, "DC {c} became {v}");
        }
    }

    #[test]
    fn resample_output_is_finite_and_bounded(x in signal(3_000), from in rate(), to in rate()) {
        let peak = x.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        for v in audio::resample(&x, from, to) {
            prop_assert!(v.is_finite());
            // Lanczos ringing overshoots, but boundedly.
            prop_assert!(v.abs() <= peak * 1.6 + 1e-6, "{v} vs peak {peak}");
        }
    }

    #[test]
    fn resample_same_rate_is_identity(x in signal(2_000), r in rate()) {
        prop_assert_eq!(audio::resample(&x, r, r), x);
    }
}

#[test]
fn downsampling_rejects_aliases_and_keeps_speech_band() {
    for from in [48_000u32, 44_100] {
        let interior = |y: &[f32]| y[400..y.len() - 400].to_vec();
        // Speech band passes nearly untouched.
        for f in [300.0f32, 1_000.0, 3_000.0] {
            let y = interior(&audio::resample(&tone(f, from, 1.0), from, 16_000));
            let ratio = rms(&y) / std::f32::consts::FRAC_1_SQRT_2;
            assert!((0.9..1.1).contains(&ratio), "{from}: {f} Hz gain {ratio}");
        }
        // Content above the new Nyquist (8 kHz) must not fold back.
        for f in [11_000.0f32, 14_000.0, 20_000.0] {
            let y = interior(&audio::resample(&tone(f, from, 1.0), from, 16_000));
            let ratio = rms(&y) / std::f32::consts::FRAC_1_SQRT_2;
            assert!(ratio < 0.05, "{from}: {f} Hz aliased with gain {ratio}");
        }
    }
}

// --- Filters, gate, normalizers ---------------------------------------------

proptest! {
    #[test]
    fn highpass_is_stable(x in signal(4_000), cutoff in 20.0f64..1_000.0) {
        let mut y = x.clone();
        audio::apply_highpass_filter(&mut y, cutoff, 16_000.0);
        for v in y {
            prop_assert!(v.is_finite() && v.abs() < 4.0);
        }
    }

    #[test]
    fn highpass_removes_dc(c in -1.0f32..1.0, cutoff in 50.0f64..400.0) {
        let mut y = vec![c; 16_000];
        audio::apply_highpass_filter(&mut y, cutoff, 16_000.0);
        prop_assert!(rms(&y[8_000..]) < 1e-3);
    }

    #[test]
    fn noise_gate_is_all_or_nothing(x in signal(1_024), thr in 0.0f32..0.5) {
        prop_assume!(!x.is_empty());
        let mut y = x.clone();
        audio::apply_noise_gate(&mut y, thr);
        prop_assert!(y == x || y.iter().all(|&v| v == 0.0));
        // Idempotent and never adds energy.
        let mut z = y.clone();
        audio::apply_noise_gate(&mut z, thr);
        prop_assert_eq!(&z, &y);
        prop_assert!(rms(&y) <= rms(&x));
    }

    #[test]
    fn normalize_bounded_contract(x in signal(2_000), target in 0.1f32..1.0, max_gain in 1.0f32..30.0) {
        let peak_in = x.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        let mut y = x.clone();
        let gain = audio::normalize_bounded(&mut y, target, max_gain);
        prop_assert!((1.0..=max_gain).contains(&gain));
        let peak_out = y.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        prop_assert!(peak_out <= peak_in.max(target) * 1.0001 + 1e-7);
        for (a, b) in x.iter().zip(&y) {
            prop_assert!(a.signum() == b.signum() || *a == 0.0);
        }
    }

    #[test]
    fn peak_normalize_is_idempotent(x in signal(2_000), thr in 0.05f32..1.0) {
        let mut once = x.clone();
        audio::peak_normalize(&mut once, thr);
        let mut twice = once.clone();
        audio::peak_normalize(&mut twice, thr);
        prop_assert_eq!(once, twice);
    }
}

// --- Sentence splitting -------------------------------------------------------

proptest! {
    #[test]
    fn split_sentences_loses_nothing(text in "\\PC{0,300}") {
        let parts = split_sentences(&text);
        let strip = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        let joined: String = parts.iter().map(|p| strip(p)).collect();
        prop_assert_eq!(joined, strip(&text));
        if !text.trim().is_empty() {
            prop_assert!(parts.iter().all(|p| !p.trim().is_empty()));
        }
    }
}

// --- Transcript layout ---------------------------------------------------------

fn segments() -> impl Strategy<Value = Vec<TranscriptSegment>> {
    prop::collection::vec((0u64..20_000, 0u64..15_000, "[a-z ]{0,40}"), 0..80).prop_map(|parts| {
        let mut t = 0;
        parts
            .into_iter()
            .enumerate()
            .map(|(i, (gap, dur, text))| {
                let start = t + gap;
                t = start + dur;
                TranscriptSegment {
                    seq: i as u64 + 1,
                    start_ms: start,
                    end_ms: start + dur,
                    text,
                    words: Vec::new(),
                }
            })
            .collect()
    })
}

fn layout() -> impl Strategy<Value = LayoutOptions> {
    (0u64..10_000, 1usize..2_000, 1u64..600_000).prop_map(|(g, c, s)| LayoutOptions {
        paragraph_gap_ms: g,
        paragraph_max_chars: c,
        section_target_ms: s,
    })
}

fn words(s: &str) -> Vec<String> {
    s.split_whitespace().map(str::to_string).collect()
}

proptest! {
    #[test]
    fn paragraphs_partition_segments_in_order(segs in segments(), opts in layout()) {
        let paras = transcript::paragraphs(&segs, &opts);
        let nonempty: Vec<&TranscriptSegment> =
            segs.iter().filter(|s| !s.text.trim().is_empty()).collect();
        let seqs: Vec<u64> = paras.iter().flat_map(|p| p.segments.clone()).collect();
        prop_assert_eq!(seqs, nonempty.iter().map(|s| s.seq).collect::<Vec<_>>());

        let para_words: Vec<String> = paras.iter().flat_map(|p| words(&p.text)).collect();
        let seg_words: Vec<String> = nonempty.iter().flat_map(|s| words(&s.text)).collect();
        prop_assert_eq!(para_words, seg_words);

        for w in paras.windows(2) {
            prop_assert!(w[0].start_ms <= w[1].start_ms);
            prop_assert!(w[0].end_ms <= w[1].start_ms || opts.paragraph_gap_ms == 0 || w[0].text.len() >= opts.paragraph_max_chars);
        }
        for p in &paras {
            prop_assert!(p.start_ms <= p.end_ms);
        }
    }

    #[test]
    fn sections_partition_paragraphs(
        segs in segments(),
        opts in layout(),
        marker_times in prop::collection::vec(0u64..2_000_000, 0..6),
    ) {
        let markers: Vec<Marker> = marker_times
            .iter()
            .enumerate()
            .map(|(i, &t)| Marker { t_ms: t, label: format!("M{i}") })
            .collect();
        let paras = transcript::paragraphs(&segs, &opts);
        let secs = transcript::sections(paras.clone(), &markers, &opts);
        let flat: Vec<_> = secs.iter().flat_map(|s| s.paragraphs.clone()).collect();
        prop_assert_eq!(&flat, &paras);
        for s in &secs {
            prop_assert!(!s.paragraphs.is_empty());
            prop_assert!(s.start_ms <= s.end_ms);
            if let Some(t) = &s.title {
                prop_assert!(markers.iter().any(|m| &m.label == t && m.t_ms <= s.start_ms));
            }
        }
        // Markdown always renders every paragraph's text.
        let md = transcript::to_markdown(&transcript::DocMeta::default(), &secs);
        for p in &flat {
            prop_assert!(md.contains(&p.text));
        }
    }

    #[test]
    fn subtitles_have_one_cue_per_nonempty_segment(segs in segments()) {
        let n = segs.iter().filter(|s| !s.text.trim().is_empty()).count();
        let srt = transcript::to_srt(&segs);
        prop_assert_eq!(srt.matches(" --> ").count(), n);
        for (i, block) in srt.split("\n\n").filter(|b| !b.is_empty()).enumerate() {
            let index = block.lines().next().unwrap().to_string();
            prop_assert_eq!(index, (i + 1).to_string());
        }
        prop_assert_eq!(transcript::to_vtt(&segs).matches(" --> ").count(), n);
    }

    #[test]
    fn collapse_repeats_is_idempotent_subsequence(text in "(?:(?:ab|cd|ef|the) ?){0,40}", k in 1usize..5) {
        let once = transcript::collapse_repeats(&text, k);
        prop_assert_eq!(transcript::collapse_repeats(&once, k), once.clone());
        // Output words are an in-order subsequence of the input words.
        let input = words(&text);
        let mut it = input.iter();
        for w in words(&once) {
            prop_assert!(it.any(|x| *x == w));
        }
    }

    #[test]
    fn collapse_repeats_keeps_distinct_words(ws in prop::collection::btree_set("[a-z]{1,8}", 0..30)) {
        let text = ws.into_iter().collect::<Vec<_>>().join(" ");
        prop_assert_eq!(transcript::collapse_repeats(&text, 1), text);
    }

    #[test]
    fn wer_metric_properties(a in "[a-z ]{0,80}", b in "[a-z ]{0,80}") {
        prop_assert_eq!(transcript::word_error_rate(&a, &a).edits, 0);
        let ab = transcript::word_error_rate(&a, &b);
        let ba = transcript::word_error_rate(&b, &a);
        prop_assert_eq!(ab.edits, ba.edits);
        let (la, lb) = (words(&a).len(), words(&b).len());
        prop_assert!(ab.edits <= la.max(lb));
        prop_assert!(ab.edits >= la.abs_diff(lb));
    }
}

// --- Boundary merger ----------------------------------------------------------

fn decoded_stream() -> impl Strategy<Value = Vec<transcript::DecodedSegment>> {
    prop::collection::vec(
        (0u64..2_000, 500u64..30_000, any::<bool>(), 0usize..8),
        0..30,
    )
    .prop_map(|parts| {
        let mut t = 0;
        let mut n = 0;
        parts
            .into_iter()
            .map(|(gap, dur, forced, nwords)| {
                let start = t + gap;
                let end = start + dur;
                t = end;
                // Words spread over the padded window [start-250, end+250].
                let words: Vec<transcript::Word> = (0..nwords)
                    .map(|k| {
                        n += 1;
                        transcript::Word {
                            w: format!("w{n}"),
                            t: start.saturating_sub(250)
                                + (dur + 500) * k as u64 / nwords.max(1) as u64,
                        }
                    })
                    .collect();
                transcript::DecodedSegment {
                    start_ms: start,
                    end_ms: end,
                    forced_split: forced,
                    text: words
                        .iter()
                        .map(|w| w.w.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    words,
                }
            })
            .collect()
    })
}

proptest! {
    #[test]
    fn merger_preserves_order_and_never_invents_words(stream in decoded_stream()) {
        let mut m = transcript::BoundaryMerger::new(500);
        let mut out = Vec::new();
        for s in stream.clone() {
            out.extend(m.push(s));
        }
        out.extend(m.finish());
        // Every input segment comes out exactly once, in order.
        prop_assert_eq!(out.len(), stream.len());
        for (o, i) in out.iter().zip(&stream) {
            prop_assert_eq!((o.start_ms, o.end_ms), (i.start_ms, i.end_ms));
            // Output words are a subsequence of that segment's input words.
            let mut it = i.words.iter();
            for w in &o.words {
                prop_assert!(it.any(|x| x == w));
            }
        }
        // Word times across the merged output never go backwards at a boundary
        // that was reconciled (forced split followed by a touching segment).
        for (k, pair) in out.windows(2).enumerate() {
            let touching = stream[k].forced_split
                && pair[1].start_ms.saturating_sub(pair[0].end_ms) <= 500;
            if touching && let (Some(a), Some(b)) = (pair[0].words.last(), pair[1].words.first()) {
                prop_assert!(a.t < b.t, "{} !< {}", a.t, b.t);
            }
        }
    }
}
