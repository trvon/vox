use crate::config::Config;
use crate::error::{Result, VoiceError};
use sherpa_rs::silero_vad::{SileroVadConfig, SileroVad, SpeechSegment};

pub struct VadSession {
    vad: SileroVad,
}

impl VadSession {
    pub fn new(config: &Config, max_speech_secs: f32) -> Result<Self> {
        let vad_config = SileroVadConfig {
            model: config.vad_model_path().to_string_lossy().to_string(),
            min_silence_duration: 0.5,
            min_speech_duration: 0.25,
            threshold: 0.5,
            sample_rate: 16000,
            window_size: 512,
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
