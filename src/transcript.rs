//! Pure transcript post-processing: paragraphing, sectioning, and export.
//!
//! Everything here is deterministic and free of I/O so it can be property
//! tested. The structure is deliberately simple and explainable — pauses make
//! paragraphs, markers (or a time budget) make sections — and leaves the
//! rewriting into prose to the calling LLM.

use serde::{Deserialize, Serialize};

/// One transcribed VAD segment. Times are relative to the recording start.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub seq: u64,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    /// Word start times, when the STT model provides timestamps.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<Word>,
}

/// A word and its absolute start time (ms from recording start).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Word {
    pub w: String,
    pub t: u64,
}

/// Group sub-word tokens into words. sherpa marks word-initial tokens with a
/// leading space; a word's time is its first token's time.
pub fn words_from_tokens(tokens: &[(String, u64)]) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    for (tok, t) in tokens {
        let starts_word = tok.starts_with(char::is_whitespace) || words.is_empty();
        let piece = tok.trim();
        if piece.is_empty() {
            continue;
        }
        match words.last_mut() {
            Some(last) if !starts_word => last.w.push_str(piece),
            _ => words.push(Word {
                w: piece.to_string(),
                t: *t,
            }),
        }
    }
    words
}

/// A decoded segment before boundary reconciliation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedSegment {
    /// Unpadded VAD bounds.
    pub start_ms: u64,
    pub end_ms: u64,
    /// The VAD cut this segment at its max length, so the next one is contiguous.
    pub forced_split: bool,
    pub text: String,
    /// Empty when the model has no timestamps.
    pub words: Vec<Word>,
}

impl DecodedSegment {
    fn retain_words(&mut self, keep: impl Fn(u64) -> bool) {
        if self.words.is_empty() {
            return;
        }
        let before = self.words.len();
        self.words.retain(|w| keep(w.t));
        if self.words.len() != before {
            self.text = self
                .words
                .iter()
                .map(|w| w.w.as_str())
                .collect::<Vec<_>>()
                .join(" ");
        }
    }
}

/// Reconciles adjacent segments whose padded audio overlaps.
///
/// Segments are padded on both sides so the VAD doesn't clip word edges. When
/// speech runs past the VAD's max length it is force-split and the next segment
/// starts right after, so both padded windows contain the same boundary audio
/// and both transcripts may contain the same words. A force-split segment is
/// held until its successor arrives; then words are assigned by start time to
/// whichever side of the midpoint between the two segments they fall on.
/// Without word timestamps segments pass through unchanged.
pub struct BoundaryMerger {
    held: Option<DecodedSegment>,
    /// Segments closer than this (end to start) are treated as overlapping.
    overlap_ms: u64,
}

impl BoundaryMerger {
    pub fn new(overlap_ms: u64) -> Self {
        Self {
            held: None,
            overlap_ms,
        }
    }

    /// Feed the next segment; returns segments that are now final, in order.
    pub fn push(&mut self, mut seg: DecodedSegment) -> Vec<DecodedSegment> {
        let mut out = Vec::new();
        if let Some(mut prev) = self.held.take() {
            if !prev.words.is_empty()
                && !seg.words.is_empty()
                && seg.start_ms.saturating_sub(prev.end_ms) <= self.overlap_ms
            {
                let boundary = (prev.end_ms + seg.start_ms.max(prev.end_ms)) / 2;
                prev.retain_words(|t| t < boundary);
                seg.retain_words(|t| t >= boundary);
            }
            out.push(prev);
        }
        if seg.forced_split && !seg.words.is_empty() {
            self.held = Some(seg);
        } else {
            out.push(seg);
        }
        out
    }

    /// Release a held segment at end of stream.
    pub fn finish(&mut self) -> Vec<DecodedSegment> {
        self.held.take().into_iter().collect()
    }
}

/// A user/agent-placed chapter marker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Marker {
    pub t_ms: u64,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Paragraph {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    /// `seq` of every segment folded into this paragraph, in order.
    pub segments: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Section {
    pub title: Option<String>,
    pub start_ms: u64,
    pub end_ms: u64,
    pub paragraphs: Vec<Paragraph>,
}

#[derive(Debug, Clone, Copy)]
pub struct LayoutOptions {
    /// Silence between segments that starts a new paragraph.
    pub paragraph_gap_ms: u64,
    /// Soft cap on paragraph length; a paragraph is closed at the next segment
    /// boundary once it exceeds this.
    pub paragraph_max_chars: usize,
    /// Without markers, start a new section once the current one is this long.
    pub section_target_ms: u64,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            paragraph_gap_ms: 2_000,
            paragraph_max_chars: 900,
            section_target_ms: 5 * 60_000,
        }
    }
}

/// Collapse runaway repetition (a known failure mode of encoder-decoder STT
/// on noise or music): any word or two-word phrase repeated more than
/// `max_repeats` times consecutively is cut back to `max_repeats`.
pub fn collapse_repeats(text: &str, max_repeats: usize) -> String {
    let max_repeats = max_repeats.max(1);
    let words: Vec<&str> = text.split_whitespace().collect();
    let norm = |w: &str| {
        w.trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase()
    };

    let mut out: Vec<&str> = Vec::with_capacity(words.len());
    for n in [1usize, 2] {
        let src: Vec<&str> = if n == 1 {
            words.clone()
        } else {
            std::mem::take(&mut out)
        };
        out.clear();
        let mut i = 0;
        while i < src.len() {
            // Count how many times the n-gram at i repeats back-to-back.
            let mut reps = 1;
            if i + n <= src.len() {
                while i + (reps + 1) * n <= src.len()
                    && (0..n).all(|k| norm(src[i + k]) == norm(src[i + reps * n + k]))
                    && (0..n).all(|k| !norm(src[i + k]).is_empty())
                {
                    reps += 1;
                }
            }
            if reps > max_repeats {
                out.extend_from_slice(&src[i..i + max_repeats * n]);
                i += reps * n;
            } else {
                out.push(src[i]);
                i += 1;
            }
        }
    }
    out.join(" ")
}

/// Normalize raw STT output for storage. Returns `None` for empty results.
pub fn clean_segment_text(text: &str) -> Option<String> {
    let t = collapse_repeats(text.trim(), 3);
    if t.is_empty() { None } else { Some(t) }
}

/// Group segments into paragraphs separated by pauses.
pub fn paragraphs(segments: &[TranscriptSegment], opts: &LayoutOptions) -> Vec<Paragraph> {
    let mut out: Vec<Paragraph> = Vec::new();
    for seg in segments {
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        let start_new = match out.last() {
            None => true,
            Some(p) => {
                seg.start_ms.saturating_sub(p.end_ms) >= opts.paragraph_gap_ms
                    || p.text.len() >= opts.paragraph_max_chars
            }
        };
        if start_new {
            out.push(Paragraph {
                start_ms: seg.start_ms,
                end_ms: seg.end_ms.max(seg.start_ms),
                text: text.to_string(),
                segments: vec![seg.seq],
            });
        } else if let Some(p) = out.last_mut() {
            p.text.push(' ');
            p.text.push_str(text);
            p.end_ms = p.end_ms.max(seg.end_ms);
            p.segments.push(seg.seq);
        }
    }
    out
}

/// Partition paragraphs into sections. A marker opens a section (titled with
/// the marker label) at the first paragraph starting at or after it. Between
/// markers, a new untitled section starts once the current one reaches
/// `section_target_ms`.
pub fn sections(paras: Vec<Paragraph>, markers: &[Marker], opts: &LayoutOptions) -> Vec<Section> {
    let mut markers: Vec<&Marker> = markers.iter().collect();
    markers.sort_by_key(|m| m.t_ms);
    let mut next_marker = 0;

    let mut out: Vec<Section> = Vec::new();
    for p in paras {
        // Consume every marker at or before this paragraph; the last one wins.
        let mut title: Option<String> = None;
        let mut marker_hit = false;
        while next_marker < markers.len() && markers[next_marker].t_ms <= p.start_ms {
            title = Some(markers[next_marker].label.clone());
            marker_hit = true;
            next_marker += 1;
        }
        let start_new = match out.last() {
            None => true,
            Some(s) => {
                marker_hit || p.start_ms.saturating_sub(s.start_ms) >= opts.section_target_ms
            }
        };
        if start_new {
            out.push(Section {
                title,
                start_ms: p.start_ms,
                end_ms: p.end_ms,
                paragraphs: vec![p],
            });
        } else if let Some(s) = out.last_mut() {
            s.end_ms = s.end_ms.max(p.end_ms);
            s.paragraphs.push(p);
        }
    }
    out
}

/// `H:MM:SS` for >= 1 h, otherwise `MM:SS`.
pub fn format_clock(ms: u64) -> String {
    let s = ms / 1000;
    let (h, m, s) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

fn subtitle_time(ms: u64, sep: char) -> String {
    let (h, rem) = (ms / 3_600_000, ms % 3_600_000);
    let (m, rem) = (rem / 60_000, rem % 60_000);
    let (s, milli) = (rem / 1000, rem % 1000);
    format!("{h:02}:{m:02}:{s:02}{sep}{milli:03}")
}

/// SubRip subtitles, one cue per non-empty segment.
pub fn to_srt(segments: &[TranscriptSegment]) -> String {
    let mut out = String::new();
    for (i, seg) in segments
        .iter()
        .filter(|s| !s.text.trim().is_empty())
        .enumerate()
    {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            subtitle_time(seg.start_ms, ','),
            subtitle_time(seg.end_ms.max(seg.start_ms), ','),
            seg.text.trim()
        ));
    }
    out
}

/// WebVTT subtitles, one cue per non-empty segment.
pub fn to_vtt(segments: &[TranscriptSegment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for seg in segments.iter().filter(|s| !s.text.trim().is_empty()) {
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            subtitle_time(seg.start_ms, '.'),
            subtitle_time(seg.end_ms.max(seg.start_ms), '.'),
            seg.text.trim()
        ));
    }
    out
}

/// Plain text, one paragraph per line block.
pub fn to_text(paras: &[Paragraph]) -> String {
    paras
        .iter()
        .map(|p| p.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Header metadata for markdown export.
#[derive(Debug, Clone, Default)]
pub struct DocMeta {
    pub title: String,
    pub recorded_at: Option<String>,
    pub duration_ms: u64,
    pub source: Option<String>,
}

/// Markdown "drafting scaffold": sections with clock-stamped paragraphs. Meant
/// as source material an LLM (or a person) edits into a blog post or lecture
/// notes; timestamps let every quote be traced back to the audio.
pub fn to_markdown(meta: &DocMeta, secs: &[Section]) -> String {
    let mut out = format!("# {}\n\n", meta.title.trim());
    let mut facts = Vec::new();
    if let Some(at) = &meta.recorded_at {
        facts.push(format!("Recorded {at}"));
    }
    facts.push(format!("Duration {}", format_clock(meta.duration_ms)));
    let n_paras: usize = secs.iter().map(|s| s.paragraphs.len()).sum();
    facts.push(format!("{} sections, {n_paras} paragraphs", secs.len()));
    if let Some(src) = &meta.source {
        facts.push(format!("Source `{src}`"));
    }
    out.push_str(&format!("> {}\n\n", facts.join(" · ")));

    for (i, s) in secs.iter().enumerate() {
        let title = s.title.clone().unwrap_or_else(|| format!("Part {}", i + 1));
        out.push_str(&format!(
            "## {} ({}–{})\n\n",
            title,
            format_clock(s.start_ms),
            format_clock(s.end_ms)
        ));
        for p in &s.paragraphs {
            out.push_str(&format!(
                "**[{}]** {}\n\n",
                format_clock(p.start_ms),
                p.text
            ));
        }
    }
    out
}

/// Lowercased words with punctuation stripped (apostrophes kept), for WER.
pub fn normalize_words(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || c == '-')
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Word-level edit distance counts between reference and hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct WerStats {
    pub reference_words: usize,
    pub edits: usize,
}

impl WerStats {
    pub fn wer(&self) -> f64 {
        if self.reference_words == 0 {
            if self.edits == 0 { 0.0 } else { 1.0 }
        } else {
            self.edits as f64 / self.reference_words as f64
        }
    }
}

/// Word error rate via Levenshtein distance over normalized words.
pub fn word_error_rate(reference: &str, hypothesis: &str) -> WerStats {
    let r = normalize_words(reference);
    let h = normalize_words(hypothesis);
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    let mut cur = vec![0; h.len() + 1];
    for i in 1..=r.len() {
        cur[0] = i;
        for j in 1..=h.len() {
            let sub = prev[j - 1] + usize::from(r[i - 1] != h[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    WerStats {
        reference_words: r.len(),
        edits: prev[h.len()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(seq: u64, start: u64, end: u64, text: &str) -> TranscriptSegment {
        TranscriptSegment {
            seq,
            start_ms: start,
            end_ms: end,
            text: text.into(),
            words: Vec::new(),
        }
    }

    fn decoded(start: u64, end: u64, forced: bool, words: &[(&str, u64)]) -> DecodedSegment {
        let words: Vec<Word> = words
            .iter()
            .map(|(w, t)| Word {
                w: w.to_string(),
                t: *t,
            })
            .collect();
        DecodedSegment {
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
    }

    #[test]
    fn words_group_subword_tokens() {
        let toks: Vec<(String, u64)> = [
            (" Kuber", 100),
            ("net", 180),
            ("es", 220),
            (" is", 400),
            (",", 450),
        ]
        .iter()
        .map(|(s, t)| (s.to_string(), *t))
        .collect();
        let w = words_from_tokens(&toks);
        assert_eq!(
            w.iter().map(|w| w.w.as_str()).collect::<Vec<_>>(),
            ["Kubernetes", "is,"]
        );
        assert_eq!(w[0].t, 100);
        assert_eq!(w[1].t, 400);
    }

    #[test]
    fn merger_dedupes_forced_split_overlap() {
        let mut m = BoundaryMerger::new(500);
        // Forced split at 25 000 ms; padded windows overlap by 250 ms each side.
        let a = decoded(
            0,
            25_000,
            true,
            &[("the", 100), ("sampling", 24_600), ("theorem", 24_950)],
        );
        let b = decoded(
            25_100,
            30_000,
            false,
            &[("theorem", 24_900), ("says", 25_300)],
        );
        assert!(m.push(a).is_empty(), "forced segment is held");
        let out = m.push(b);
        assert_eq!(out.len(), 2);
        // Midpoint is 25 050 ms: "theorem" (24 950) stays left, its duplicate
        // in the right segment (24 900) is dropped.
        assert_eq!(out[0].text, "the sampling theorem");
        assert_eq!(out[1].text, "says");
        assert!(m.finish().is_empty());
    }

    #[test]
    fn merger_passes_through_without_timestamps_or_gap() {
        let mut m = BoundaryMerger::new(500);
        let plain = DecodedSegment {
            start_ms: 0,
            end_ms: 25_000,
            forced_split: true,
            text: "no timestamps".into(),
            words: vec![],
        };
        assert_eq!(m.push(plain.clone()), vec![plain]);
        let a = decoded(0, 25_000, true, &[("a", 24_900)]);
        let b = decoded(27_000, 28_000, false, &[("b", 26_900)]); // far apart: no trim
        assert!(m.push(a.clone()).is_empty());
        assert_eq!(m.push(b.clone()), vec![a, b]);
        let held = decoded(0, 25_000, true, &[("x", 10)]);
        m.push(held.clone());
        assert_eq!(m.finish(), vec![held]);
    }

    #[test]
    fn wer_basics() {
        assert_eq!(word_error_rate("Hello, world!", "hello world").edits, 0);
        assert_eq!(word_error_rate("a b c d", "a x c").edits, 2);
        assert!((word_error_rate("a b c d", "a b c d e").wer() - 0.25).abs() < 1e-9);
        assert_eq!(word_error_rate("", "").wer(), 0.0);
        assert_eq!(
            normalize_words("Low pass-filter, don't."),
            ["low", "pass", "filter", "don't"]
        );
    }

    #[test]
    fn collapse_repeats_cuts_runaway_words_and_bigrams() {
        assert_eq!(
            collapse_repeats("the the the the the end", 3),
            "the the the end"
        );
        assert_eq!(
            collapse_repeats("thank you thank you thank you thank you thank you", 2),
            "thank you thank you"
        );
        assert_eq!(
            collapse_repeats("a normal sentence.", 3),
            "a normal sentence."
        );
        assert_eq!(collapse_repeats("", 3), "");
    }

    #[test]
    fn paragraphs_split_on_pause_and_skip_empty() {
        let segs = vec![
            seg(1, 0, 1000, "Hello."),
            seg(2, 1500, 3000, "Still talking."),
            seg(3, 3100, 3200, "   "),
            seg(4, 6000, 7000, "New thought."),
        ];
        let p = paragraphs(&segs, &LayoutOptions::default());
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].text, "Hello. Still talking.");
        assert_eq!(p[0].segments, vec![1, 2]);
        assert_eq!(p[1].start_ms, 6000);
    }

    #[test]
    fn sections_follow_markers_then_time_budget() {
        let opts = LayoutOptions {
            section_target_ms: 10_000,
            ..LayoutOptions::default()
        };
        let segs: Vec<_> = (0..6)
            .map(|i| seg(i, i * 5000, i * 5000 + 1000, "x"))
            .collect();
        let paras = paragraphs(&segs, &opts);
        let markers = vec![Marker {
            t_ms: 14_000,
            label: "Q&A".into(),
        }];
        let s = sections(paras, &markers, &opts);
        // Budget opens 10 s; the marker opens 15 s; budget again at 25 s.
        let starts: Vec<u64> = s.iter().map(|x| x.start_ms).collect();
        assert_eq!(starts, vec![0, 10_000, 15_000, 25_000]);
        assert_eq!(s[1].title, None);
        assert_eq!(s[2].title.as_deref(), Some("Q&A"));
    }

    #[test]
    fn subtitle_formats() {
        let segs = vec![seg(1, 3_723_456, 3_724_000, "hi")];
        assert_eq!(to_srt(&segs), "1\n01:02:03,456 --> 01:02:04,000\nhi\n\n");
        assert!(to_vtt(&segs).starts_with("WEBVTT\n\n01:02:03.456 --> "));
        assert_eq!(format_clock(59_999), "00:59");
        assert_eq!(format_clock(3_600_000), "1:00:00");
    }

    #[test]
    fn markdown_has_title_sections_and_timestamps() {
        let segs = vec![seg(1, 0, 1000, "Intro."), seg(2, 65_000, 66_000, "Body.")];
        let opts = LayoutOptions::default();
        let md = to_markdown(
            &DocMeta {
                title: "Talk".into(),
                duration_ms: 66_000,
                ..Default::default()
            },
            &sections(paragraphs(&segs, &opts), &[], &opts),
        );
        assert!(md.starts_with("# Talk\n"));
        assert!(md.contains("## Part 1 (00:00–01:06)"));
        assert!(md.contains("**[01:05]** Body."));
    }
}
