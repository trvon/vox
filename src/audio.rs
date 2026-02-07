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
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    // Small buffer to let the last samples play out
    std::thread::sleep(std::time::Duration::from_millis(20));

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
        let mut accumulator: Vec<f32> = Vec::with_capacity(VAD_WINDOW_SIZE * 4);

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

/// Lanczos-3 windowed sinc resampler with precomputed kernel table.
///
/// Uses a 6-tap (a=3) Lanczos kernel which suppresses aliasing far better than
/// linear interpolation, especially for the common 48kHz→16kHz (3:1) downsampling
/// of microphone input fed to the STT engine. The kernel is precomputed into a
/// 512-entry lookup table to avoid sin() calls in the real-time audio path.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    const A: i32 = 3; // Lanczos kernel half-width

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let center = src_pos.floor() as i32;
        let mut sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;

        for j in (center - A + 1)..=(center + A) {
            let x = src_pos - j as f64;
            let w = lanczos3_table(x);
            if j >= 0 && (j as usize) < samples.len() {
                sum += samples[j as usize] as f64 * w;
                weight_sum += w;
            }
        }

        let sample = if weight_sum.abs() > 1e-10 {
            sum / weight_sum
        } else {
            0.0
        };
        output.push(sample as f32);
    }

    output
}

// Precomputed Lanczos-3 kernel lookup table.
// Covers [0, 3) with KERNEL_TABLE_SIZE entries. Symmetric, so we use abs(x).
const KERNEL_TABLE_SIZE: usize = 512;
const KERNEL_TABLE_SCALE: f64 = KERNEL_TABLE_SIZE as f64 / 3.0;

/// Build the kernel table at compile time is not possible with sin(),
/// so we use std::sync::LazyLock for one-time init.
static KERNEL_TABLE: std::sync::LazyLock<[f32; KERNEL_TABLE_SIZE]> = std::sync::LazyLock::new(|| {
    let mut table = [0.0f32; KERNEL_TABLE_SIZE];
    for (i, entry) in table.iter_mut().enumerate() {
        let x = i as f64 / KERNEL_TABLE_SCALE;
        *entry = lanczos3_exact(x) as f32;
    }
    table
});

/// Fast kernel lookup with linear interpolation between table entries.
#[inline]
fn lanczos3_table(x: f64) -> f64 {
    let ax = x.abs();
    if ax >= 3.0 {
        return 0.0;
    }
    let pos = ax * KERNEL_TABLE_SCALE;
    let idx = pos as usize;
    if idx + 1 >= KERNEL_TABLE_SIZE {
        return KERNEL_TABLE[KERNEL_TABLE_SIZE - 1] as f64;
    }
    let frac = pos - idx as f64;
    let a = KERNEL_TABLE[idx] as f64;
    let b = KERNEL_TABLE[idx + 1] as f64;
    a + (b - a) * frac
}

/// Exact Lanczos-3 kernel: sinc(x) * sinc(x/3) for |x| < 3, else 0.
/// Used only to build the lookup table.
fn lanczos3_exact(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        return 1.0;
    }
    if x.abs() >= 3.0 {
        return 0.0;
    }
    let px = std::f64::consts::PI * x;
    (px.sin() / px) * ((px / 3.0).sin() / (px / 3.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_same_rate_is_passthrough() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = resample(&input, 16000, 16000);
        assert_eq!(output, input);
    }

    #[test]
    fn resample_downsample_48k_to_16k() {
        // 3:1 ratio — 12 input samples should yield 4 output samples
        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let output = resample(&input, 48000, 16000);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn resample_upsample_8k_to_16k() {
        // 1:2 ratio — 4 input samples should yield 8 output samples
        let input = vec![0.0, 1.0, 2.0, 3.0];
        let output = resample(&input, 8000, 16000);
        assert_eq!(output.len(), 8);
        // First sample should be close to 0.0 (kernel edge effects allow small deviation)
        assert!((output[0]).abs() < 0.05);
    }

    #[test]
    fn resample_empty_input() {
        let output = resample(&[], 48000, 16000);
        assert!(output.is_empty());
    }

    #[test]
    fn resample_single_sample() {
        let input = vec![0.5];
        let output = resample(&input, 48000, 16000);
        // With 3:1 ratio, 1 sample / 3 ≈ 0 output samples (integer truncation)
        // but at least it shouldn't panic
        assert!(output.len() <= 1);
    }

    #[test]
    fn resample_output_length_matches_ratio() {
        let input: Vec<f32> = vec![0.0; 48000]; // 1 second at 48kHz
        let output = resample(&input, 48000, 16000);
        // Should be approximately 16000 samples (1 second at 16kHz)
        assert_eq!(output.len(), 16000);
    }

    #[test]
    fn lanczos3_kernel_properties() {
        // Kernel is 1.0 at origin
        assert!((lanczos3_exact(0.0) - 1.0).abs() < 1e-10);
        // Kernel is 0 at integer multiples (except 0)
        assert!(lanczos3_exact(1.0).abs() < 1e-10);
        assert!(lanczos3_exact(2.0).abs() < 1e-10);
        assert!(lanczos3_exact(-1.0).abs() < 1e-10);
        assert!(lanczos3_exact(-2.0).abs() < 1e-10);
        // Kernel is 0 outside [-3, 3]
        assert_eq!(lanczos3_exact(3.0), 0.0);
        assert_eq!(lanczos3_exact(-3.0), 0.0);
        assert_eq!(lanczos3_exact(4.0), 0.0);
        // Kernel is symmetric
        assert!((lanczos3_exact(0.5) - lanczos3_exact(-0.5)).abs() < 1e-10);
        assert!((lanczos3_exact(1.5) - lanczos3_exact(-1.5)).abs() < 1e-10);
    }

    #[test]
    fn lanczos3_table_matches_exact() {
        // Verify table lookup is close to exact computation
        for i in 0..100 {
            let x = i as f64 * 0.03; // 0.0 to 2.97
            let exact = lanczos3_exact(x);
            let table = lanczos3_table(x);
            assert!(
                (exact - table).abs() < 0.005,
                "Table mismatch at x={x}: exact={exact}, table={table}"
            );
        }
    }

    #[test]
    fn sinc_resampler_preserves_dc_signal() {
        // A constant signal should remain constant after resampling
        let input: Vec<f32> = vec![0.75; 4800];
        let output = resample(&input, 48000, 16000);
        assert_eq!(output.len(), 1600);
        // Interior samples (away from edges) should be very close to 0.75
        for &s in &output[3..output.len() - 3] {
            assert!(
                (s - 0.75).abs() < 0.01,
                "DC signal not preserved: got {s}, expected 0.75"
            );
        }
    }
}
