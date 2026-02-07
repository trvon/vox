use crate::config::Config;
use crate::error::{Result, VoiceError};
use sherpa_rs::silero_vad::{SileroVadConfig, SileroVad, SpeechSegment};

pub(crate) const VAD_SAMPLE_RATE: u32 = 16000;
pub(crate) const VAD_WINDOW_SIZE: usize = 512;
pub(crate) const VAD_THRESHOLD: f32 = 0.5;
pub(crate) const VAD_MIN_SILENCE: f32 = 0.5;
pub(crate) const VAD_MIN_SPEECH: f32 = 0.25;

pub struct VadSession {
    vad: SileroVad,
}

impl VadSession {
    pub fn new(config: &Config, max_speech_secs: f32) -> Result<Self> {
        let vad_config = SileroVadConfig {
            model: config.vad_model_path().to_string_lossy().to_string(),
            min_silence_duration: VAD_MIN_SILENCE,
            min_speech_duration: VAD_MIN_SPEECH,
            threshold: VAD_THRESHOLD,
            sample_rate: VAD_SAMPLE_RATE,
            window_size: VAD_WINDOW_SIZE as i32,
            ..Default::default()
        };

        let vad = SileroVad::new(vad_config, max_speech_secs)
            .map_err(|e| VoiceError::Vad(format!("Failed to initialize VAD: {e}")))?;

        Ok(Self { vad })
    }

    /// Feed a window of audio samples (must be window_size=512 samples at 16kHz)
    pub fn accept_waveform(&mut self, samples: Vec<f32>) {
        self.vad.accept_waveform(samples);
    }

    /// Check if speech has been detected
    pub fn is_speech(&mut self) -> bool {
        self.vad.is_speech()
    }

    /// Check if there are speech segments available
    pub fn is_empty(&mut self) -> bool {
        self.vad.is_empty()
    }

    /// Get the front speech segment
    pub fn front(&mut self) -> SpeechSegment {
        self.vad.front()
    }

    /// Pop the front speech segment
    pub fn pop(&mut self) {
        self.vad.pop();
    }

    /// Flush remaining audio and detect final speech segments
    pub fn flush(&mut self) {
        self.vad.flush();
    }

    /// Collect all available speech segments
    pub fn collect_segments(&mut self) -> Vec<SpeechSegment> {
        let mut segments = Vec::new();
        while !self.is_empty() {
            segments.push(self.front());
            self.pop();
        }
        segments
    }
}

// SileroVad wraps a raw pointer but sherpa-rs declares Send+Sync
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
    fn vad_threshold_is_balanced() {
        // 0.5 is the standard Silero VAD threshold
        assert!((VAD_THRESHOLD - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn vad_min_silence_allows_pauses() {
        // 0.5s minimum silence prevents premature cutoff
        assert!((VAD_MIN_SILENCE - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn vad_min_speech_filters_noise() {
        // 0.25s minimum speech duration filters clicks/noise
        assert!((VAD_MIN_SPEECH - 0.25).abs() < f32::EPSILON);
    }
}
