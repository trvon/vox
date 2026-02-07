use crate::error::{Result, VoiceError};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::sync::mpsc;

const TARGET_SAMPLE_RATE: u32 = 16000;
const VAD_WINDOW_SIZE: usize = 512;

/// Play audio samples through the default output device
pub async fn play_audio(samples: Vec<f32>, sample_rate: u32) -> Result<()> {
    tokio::task::spawn_blocking(move || play_audio_blocking(&samples, sample_rate))
        .await
        .map_err(|e| VoiceError::Audio(format!("Playback task failed: {e}")))?
}

fn play_audio_blocking(samples: &[f32], sample_rate: u32) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| VoiceError::Audio("No output device available".to_string()))?;

    tracing::debug!(
        device = device.name().unwrap_or_default(),
        sample_rate,
        num_samples = samples.len(),
        "Playing audio"
    );

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let samples = Arc::new(samples.to_vec());
    let position = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let finished = Arc::new(AtomicBool::new(false));

    let samples_clone = samples.clone();
    let position_clone = position.clone();
    let finished_clone = finished.clone();

    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let current_pos = position_clone.load(Ordering::Relaxed);
                let remaining = samples_clone.len().saturating_sub(current_pos);
                let to_copy = remaining.min(data.len());
                data[..to_copy]
                    .copy_from_slice(&samples_clone[current_pos..current_pos + to_copy]);
                for sample in data[to_copy..].iter_mut() {
                    *sample = 0.0;
                }
                position_clone.store(current_pos + to_copy, Ordering::Relaxed);
                if current_pos + to_copy >= samples_clone.len() {
                    finished_clone.store(true, Ordering::Relaxed);
                }
            },
            move |err| {
                tracing::error!("Audio output error: {err}");
            },
            None,
        )
        .map_err(|e| VoiceError::Audio(format!("Failed to build output stream: {e}")))?;

    stream
        .play()
        .map_err(|e| VoiceError::Audio(format!("Failed to play stream: {e}")))?;

    while !finished.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    // Small buffer to let the last samples play out
    std::thread::sleep(std::time::Duration::from_millis(100));

    drop(stream);
    tracing::debug!("Playback complete");
    Ok(())
}

/// Captured audio chunk from the microphone
pub struct AudioChunk {
    pub samples: Vec<f32>,
}

/// Handle to stop audio capture. This is Send-safe.
pub struct CaptureHandle {
    stop: Arc<AtomicBool>,
    // The thread that owns the cpal::Stream (which is not Send)
    _thread: Option<std::thread::JoinHandle<()>>,
}

// CaptureHandle is Send because it only holds Arc<AtomicBool> and JoinHandle (both Send)
// The non-Send cpal::Stream lives entirely on the spawned thread
unsafe impl Send for CaptureHandle {}

impl CaptureHandle {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Record audio from the default input device.
/// Returns a receiver of VAD-window-sized audio chunks (512 samples at 16kHz)
/// and a Send-safe stop handle.
pub fn start_capture() -> Result<(mpsc::Receiver<AudioChunk>, CaptureHandle)> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| VoiceError::Audio("No input device (microphone) available".to_string()))?;

    let supported_config = device
        .default_input_config()
        .map_err(|e| VoiceError::Audio(format!("Failed to get input config: {e}")))?;

    let device_sample_rate = supported_config.sample_rate().0;
    let device_channels = supported_config.channels() as usize;

    tracing::debug!(
        device = device.name().unwrap_or_default(),
        sample_rate = device_sample_rate,
        channels = device_channels,
        "Starting audio capture"
    );

    let (tx, rx) = mpsc::channel::<AudioChunk>(64);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    // Spawn a dedicated thread that owns the cpal::Stream (which is not Send).
    // This thread runs until the stop flag is set.
    let thread = std::thread::spawn(move || {
        let config = cpal::StreamConfig {
            channels: supported_config.channels(),
            sample_rate: supported_config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };

        let stop_inner = stop_clone.clone();
        let mut accumulator: Vec<f32> = Vec::new();

        let stream = match device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if stop_inner.load(Ordering::Relaxed) {
                    return;
                }

                // Convert to mono if needed
                let mono: Vec<f32> = if device_channels > 1 {
                    data.chunks(device_channels)
                        .map(|frame| frame.iter().sum::<f32>() / device_channels as f32)
                        .collect()
                } else {
                    data.to_vec()
                };

                // Resample to 16kHz if needed
                let resampled = if device_sample_rate != TARGET_SAMPLE_RATE {
                    resample(&mono, device_sample_rate, TARGET_SAMPLE_RATE)
                } else {
                    mono
                };

                accumulator.extend_from_slice(&resampled);

                // Send complete VAD windows
                while accumulator.len() >= VAD_WINDOW_SIZE {
                    let window: Vec<f32> = accumulator.drain(..VAD_WINDOW_SIZE).collect();
                    let chunk = AudioChunk { samples: window };
                    let _ = tx.try_send(chunk);
                }
            },
            move |err| {
                tracing::error!("Audio input error: {err}");
            },
            None,
        ) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("Failed to build input stream: {e}");
                return;
            }
        };

        if let Err(e) = stream.play() {
            tracing::error!("Failed to start capture: {e}");
            return;
        }

        // Keep the stream alive until stop is signaled
        while !stop_clone.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        drop(stream);
    });

    let handle = CaptureHandle {
        stop,
        _thread: Some(thread),
    };

    Ok((rx, handle))
}

/// Simple linear resampling
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;

        let sample = if src_idx + 1 < samples.len() {
            samples[src_idx] as f64 * (1.0 - frac) + samples[src_idx + 1] as f64 * frac
        } else if src_idx < samples.len() {
            samples[src_idx] as f64
        } else {
            0.0
        };

        output.push(sample as f32);
    }

    output
}
