use crate::config::Config;
use crate::error::{Result, VoiceError};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

pub(crate) const VAD_SAMPLE_RATE: u32 = 16000;
pub(crate) const VAD_WINDOW_SIZE: usize = 512;

/// Tuning knobs for a [`VadSession`].
///
/// Semantics follow sherpa-onnx's `VoiceActivityDetector`:
///
/// * `max_speech_secs` is *not* a hard cut. Once the buffered utterance grows past
///   this length, sherpa-onnx raises the speech threshold to 0.9 and drops
///   min-silence to 0.1 s so the next natural pause ends the segment. Setting it
///   too low (sherpa-rs defaulted to 0.5 s) chops normal speech into fragments.
/// * `buffer_secs` sizes the internal circular buffer, which is **preallocated**
///   (`std::vector<float>::resize`, i.e. zero-filled) at creation. It grows
///   automatically (doubling, no data loss) if an utterance overflows it, so it
///   only needs to cover a typical utterance — 86 400 s would be ~5.5 GB.
#[derive(Debug, Clone, Copy)]
pub struct VadOptions {
    /// Speech probability threshold (Silero), default 0.5.
    pub threshold: f32,
    /// Trailing silence required to close a segment, default 0.3 s.
    pub min_silence_secs: f32,
    /// Minimum speech duration to open a segment, default 0.25 s.
    pub min_speech_secs: f32,
    /// Utterance length after which sherpa-onnx force-splits more eagerly, default 20 s.
    pub max_speech_secs: f32,
    /// Initial internal ring buffer size in seconds, default 60 s.
    pub buffer_secs: f32,
}

impl Default for VadOptions {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_silence_secs: 0.3,
            min_speech_secs: 0.25,
            max_speech_secs: 20.0,
            buffer_secs: 60.0,
        }
    }
}

impl VadOptions {
    /// Long-form dictation / lecture capture: tolerate longer pauses and longer
    /// utterances before splitting.
    pub fn for_lecture() -> Self {
        Self {
            min_silence_secs: 0.5,
            max_speech_secs: 25.0,
            ..Self::default()
        }
    }
}

/// A detected speech segment.
#[derive(Debug, Clone)]
pub struct VadSegment {
    /// Index of the first sample at 16 kHz, counted from session creation.
    ///
    /// sherpa-onnx reports the start as an absolute position in its circular
    /// buffer (starts at 0, never rebased). In 1.13.x the C API type is `int32`,
    /// which wraps after 2^31 samples (~37.3 h at 16 kHz); we unwrap it against
    /// our own `u64` sample counter, so this stays correct as long as a segment
    /// starts within ~74 h of the most recent input. Note that sherpa-onnx 1.13.x
    /// also keeps its internal buffer head/tail in `int32` (fixed in 2.0, "Widen
    /// VAD sample positions to int64_t to fix long-stream SIGSEGV"), so sessions
    /// longer than ~37 h should be recreated periodically.
    pub start_sample: u64,
    pub samples: Vec<f32>,
}

impl VadSegment {
    pub fn start_ms(&self) -> u64 {
        self.start_sample * 1000 / VAD_SAMPLE_RATE as u64
    }

    pub fn end_ms(&self) -> u64 {
        (self.start_sample + self.samples.len() as u64) * 1000 / VAD_SAMPLE_RATE as u64
    }
}

pub struct VadSession {
    vad: VoiceActivityDetector,
    /// Total samples fed so far (used to unwrap sherpa's 32-bit start index).
    fed: u64,
}

/// Reconstruct an absolute 64-bit sample index from sherpa-onnx's start index,
/// which may be a wrapped `int32` (1.13.x) or an `int64` (2.x). Uses the low
/// 32 bits relative to `fed`, the number of samples accepted so far.
pub(crate) fn unwrap_start(fed: u64, raw: i64) -> u64 {
    let back = (fed as u32).wrapping_sub(raw as u32) as u64;
    fed.saturating_sub(back)
}

impl VadSession {
    pub fn new(config: &Config, opts: VadOptions) -> Result<Self> {
        let model_path = config.vad_model_path();
        if !model_path.exists() {
            return Err(VoiceError::Vad(format!(
                "VAD model not found at {}",
                model_path.display()
            )));
        }

        let vad_config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(model_path.to_string_lossy().into_owned()),
                threshold: opts.threshold,
                min_silence_duration: opts.min_silence_secs,
                min_speech_duration: opts.min_speech_secs,
                window_size: VAD_WINDOW_SIZE as i32,
                max_speech_duration: opts.max_speech_secs,
            },
            sample_rate: VAD_SAMPLE_RATE as i32,
            num_threads: 1,
            provider: Some("cpu".to_string()),
            debug: false,
            ..Default::default()
        };

        let buffer_secs = opts.buffer_secs.max(1.0);
        let vad = VoiceActivityDetector::create(&vad_config, buffer_secs)
            .ok_or_else(|| VoiceError::Vad("Failed to initialize Silero VAD".to_string()))?;

        Ok(Self { vad, fed: 0 })
    }

    /// Feed audio at 16 kHz. Any length is accepted; sherpa-onnx buffers partial
    /// windows internally and evaluates the model per 512-sample window.
    pub fn accept_waveform(&mut self, samples: &[f32]) {
        if !samples.is_empty() {
            self.vad.accept_waveform(samples);
            self.fed += samples.len() as u64;
        }
    }

    /// Whether the detector is currently inside a speech segment.
    pub fn is_speech(&mut self) -> bool {
        self.vad.detected()
    }

    /// Flush trailing speech into the segment queue (call at end of input).
    pub fn flush(&mut self) {
        self.vad.flush();
    }

    /// Drain all completed speech segments.
    pub fn collect_segments(&mut self) -> Vec<VadSegment> {
        let mut segments = Vec::new();
        while let Some(seg) = self.vad.front() {
            segments.push(VadSegment {
                start_sample: unwrap_start(self.fed, i64::from(seg.start())),
                samples: seg.samples().to_vec(),
            });
            drop(seg);
            self.vad.pop();
        }
        segments
    }
}

// VoiceActivityDetector is Send in sherpa-onnx; kept explicit for the public contract.
unsafe impl Send for VadSession {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vad_sample_rate_is_16khz() {
        assert_eq!(VAD_SAMPLE_RATE, 16000);
    }

    #[test]
    fn vad_window_size_matches_silero_requirement() {
        // Silero VAD requires 512-sample windows at 16kHz
        assert_eq!(VAD_WINDOW_SIZE, 512);
    }

    #[test]
    fn default_options_are_balanced() {
        let o = VadOptions::default();
        assert!((o.threshold - 0.5).abs() < f32::EPSILON);
        assert!((o.min_silence_secs - 0.3).abs() < f32::EPSILON);
        assert!((o.min_speech_secs - 0.25).abs() < f32::EPSILON);
        assert!((o.max_speech_secs - 20.0).abs() < f32::EPSILON);
        assert!((o.buffer_secs - 60.0).abs() < f32::EPSILON);
    }

    #[test]
    fn max_speech_is_not_sherpa_rs_default() {
        // sherpa-rs defaulted max_speech_duration to 0.5 s, which made sherpa-onnx
        // split every utterance after half a second.
        assert!(VadOptions::default().max_speech_secs >= 10.0);
        assert!(VadOptions::for_lecture().max_speech_secs >= 10.0);
    }

    #[test]
    fn lecture_options_tolerate_longer_pauses() {
        let o = VadOptions::for_lecture();
        assert!((o.min_silence_secs - 0.5).abs() < f32::EPSILON);
        assert!((o.max_speech_secs - 25.0).abs() < f32::EPSILON);
        assert!((o.threshold - VadOptions::default().threshold).abs() < f32::EPSILON);
    }

    #[test]
    fn unwrap_start_identity_for_small_values() {
        assert_eq!(unwrap_start(16_000, 0), 0);
        assert_eq!(unwrap_start(160_000, 32_000), 32_000);
    }

    #[test]
    fn unwrap_start_handles_i32_wraparound() {
        // 40 h at 16 kHz is past i32::MAX samples.
        let abs: u64 = 40 * 3600 * 16_000 + 1234;
        let fed = abs + 16_000 * 5; // segment started 5 s ago
        let wrapped = abs as u32 as i32; // what a 32-bit C API would return
        assert_eq!(unwrap_start(fed, i64::from(wrapped)), abs);
        // And an i64 source is passed through unchanged.
        assert_eq!(unwrap_start(fed, abs as i64), abs);
    }

    #[test]
    fn buffer_is_bounded_by_default() {
        // The ring buffer is preallocated; keep it small (60 s * 16 kHz * 4 B ≈ 3.8 MB).
        let bytes = VadOptions::default().buffer_secs * VAD_SAMPLE_RATE as f32 * 4.0;
        assert!(bytes < 8.0 * 1024.0 * 1024.0);
    }
}
