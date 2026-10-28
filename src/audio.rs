use crate::error::{Result, VoiceError};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::mpsc;

const TARGET_SAMPLE_RATE: u32 = 16000;

/// Second-order Butterworth high-pass biquad filter.
/// Removes low-frequency speaker bleed and room rumble from mic capture.
struct BiquadHPF {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl BiquadHPF {
    /// Create a Butterworth high-pass filter at the given cutoff frequency.
    fn highpass(cutoff_hz: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * std::f64::consts::PI * cutoff_hz / sample_rate;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0_f64.sqrt()); // Q = 1/sqrt(2) for Butterworth

        let a0 = 1.0 + alpha;
        let b0 = ((1.0 + cos_w0) / 2.0) / a0;
        let b1 = (-(1.0 + cos_w0)) / a0;
        let b2 = ((1.0 + cos_w0) / 2.0) / a0;
        let a1 = (-2.0 * cos_w0) / a0;
        let a2 = (1.0 - alpha) / a0;

        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Process a single sample through the filter (Direct Form I).
    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

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
        device = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_default(),
        sample_rate,
        num_samples = samples.len(),
        "Playing audio"
    );

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate,
        buffer_size: cpal::BufferSize::Default,
    };

    let samples = Arc::new(samples.to_vec());
    let position = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let notify = Arc::new((Mutex::new(false), Condvar::new()));

    let samples_clone = samples.clone();
    let position_clone = position.clone();
    let notify_clone = notify.clone();

    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let current_pos = position_clone.load(Ordering::Relaxed);
                let remaining = samples_clone.len().saturating_sub(current_pos);
                let to_copy = remaining.min(data.len());
                data[..to_copy].copy_from_slice(&samples_clone[current_pos..current_pos + to_copy]);
                for sample in data[to_copy..].iter_mut() {
                    *sample = 0.0;
                }
                position_clone.store(current_pos + to_copy, Ordering::Relaxed);
                if current_pos + to_copy >= samples_clone.len() {
                    let (lock, cvar) = &*notify_clone;
                    if let Ok(mut done) = lock.lock() {
                        *done = true;
                        cvar.notify_one();
                    }
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

    // Wait for playback completion via condvar (no busy-wait)
    let (lock, cvar) = &*notify;
    let _guard = cvar
        .wait_while(lock.lock().unwrap(), |done| !*done)
        .unwrap();
    // Drain for the last audio buffer to reach the DAC.
    // 25ms covers two macOS Core Audio buffer periods (~11.6ms each at 512/44.1kHz).
    std::thread::sleep(std::time::Duration::from_millis(25));

    drop(stream);
    tracing::debug!("Playback complete");
    Ok(())
}

/// Play audio in streaming mode — receives chunks from a channel and plays them
/// as they arrive through cpal. Audio starts playing as soon as the first chunk
/// is received, without waiting for full synthesis to complete.
pub async fn play_audio_streaming(
    rx: std::sync::mpsc::Receiver<Vec<f32>>,
    sample_rate: u32,
) -> Result<()> {
    tokio::task::spawn_blocking(move || play_audio_streaming_blocking(rx, sample_rate))
        .await
        .map_err(|e| VoiceError::Audio(format!("Streaming playback task failed: {e}")))?
}

fn play_audio_streaming_blocking(
    rx: std::sync::mpsc::Receiver<Vec<f32>>,
    sample_rate: u32,
) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| VoiceError::Audio("No output device available".to_string()))?;

    tracing::debug!(
        device = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_default(),
        sample_rate,
        "Starting streaming playback"
    );

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate,
        buffer_size: cpal::BufferSize::Default,
    };

    // Shared ring buffer between the receiver loop and the cpal callback
    let ring = Arc::new(Mutex::new(VecDeque::<f32>::with_capacity(
        sample_rate as usize,
    )));
    let ring_cpal = ring.clone();

    // Signal that the producer is done and ring buffer is drained
    let done = Arc::new((Mutex::new(false), Condvar::new()));
    let done_cpal = done.clone();
    let producer_finished = Arc::new(AtomicBool::new(false));
    let producer_finished_cpal = producer_finished.clone();

    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut buf = ring_cpal.lock().unwrap();
                for sample in data.iter_mut() {
                    *sample = buf.pop_front().unwrap_or(0.0);
                }
                // If producer is done and buffer is empty, signal completion
                if producer_finished_cpal.load(Ordering::Relaxed) && buf.is_empty() {
                    let (lock, cvar) = &*done_cpal;
                    if let Ok(mut finished) = lock.lock() {
                        *finished = true;
                        cvar.notify_one();
                    }
                }
            },
            move |err| {
                tracing::error!("Streaming audio output error: {err}");
            },
            None,
        )
        .map_err(|e| VoiceError::Audio(format!("Failed to build streaming output stream: {e}")))?;

    stream
        .play()
        .map_err(|e| VoiceError::Audio(format!("Failed to play streaming stream: {e}")))?;

    // Receive chunks and push into ring buffer
    while let Ok(chunk) = rx.recv() {
        let mut buf = ring.lock().unwrap();
        buf.extend(chunk.iter());
    }

    // Producer is done, signal to cpal callback
    producer_finished.store(true, Ordering::Relaxed);

    // Wait for ring buffer to drain (cpal callback sets done=true when empty)
    {
        let (lock, cvar) = &*done;
        let _guard = cvar
            .wait_while(lock.lock().unwrap(), |finished| !*finished)
            .unwrap();
    }

    // Final DAC drain
    std::thread::sleep(std::time::Duration::from_millis(25));

    drop(stream);
    tracing::debug!("Streaming playback complete");
    Ok(())
}

// --- Public DSP functions for benchmarks and testability ---

/// Peak-normalize audio to 1.0 if peak amplitude is below `threshold`.
/// Returns the peak amplitude before normalization.
pub fn peak_normalize(samples: &mut [f32], threshold: f32) -> f32 {
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 && peak < threshold {
        let gain = 1.0 / peak;
        samples.iter_mut().for_each(|s| *s *= gain);
    }
    peak
}

/// Scale toward `target_peak`, but never by more than `max_gain` and never
/// attenuating. Unlike [`peak_normalize`], a near-silent segment of room noise
/// can't be boosted 100x. Returns the gain applied.
pub fn normalize_bounded(samples: &mut [f32], target_peak: f32, max_gain: f32) -> f32 {
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak <= 0.0 || !peak.is_finite() {
        return 1.0;
    }
    let gain = (target_peak / peak).clamp(1.0, max_gain.max(1.0));
    if gain > 1.0 {
        samples.iter_mut().for_each(|s| *s *= gain);
    }
    gain
}

/// Apply a second-order Butterworth high-pass filter in-place.
pub fn apply_highpass_filter(samples: &mut [f32], cutoff_hz: f64, sample_rate: f64) {
    let mut hpf = BiquadHPF::highpass(cutoff_hz, sample_rate);
    for s in samples.iter_mut() {
        *s = hpf.process(*s as f64) as f32;
    }
}

/// RMS noise gate: zeros the window if its RMS is below `threshold`.
pub fn apply_noise_gate(window: &mut [f32], threshold: f32) {
    let rms = (window.iter().map(|s| s * s).sum::<f32>() / window.len() as f32).sqrt();
    if rms < threshold {
        window.iter_mut().for_each(|s| *s = 0.0);
    }
}

/// Captured audio chunk from the microphone
#[derive(Debug)]
pub struct AudioChunk {
    /// Absolute position of the first sample at the capture sample rate (16 kHz).
    pub start_sample: u64,
    pub samples: Vec<f32>,
}

/// Process-wide admission control, shared by every microphone consumer.
/// The permit is released only after the capture thread has stopped.
#[derive(Default)]
struct CaptureCoordinator(Arc<AtomicBool>);

impl CaptureCoordinator {
    fn acquire(&self) -> Result<CapturePermit> {
        self.0
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                VoiceError::Audio(
                    "Microphone is busy; stop the active listener, recording, or calibration first"
                        .into(),
                )
            })?;
        Ok(CapturePermit(self.0.clone()))
    }
}

struct CapturePermit(Arc<AtomicBool>);

impl Drop for CapturePermit {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

static CAPTURE_COORDINATOR: std::sync::LazyLock<CaptureCoordinator> =
    std::sync::LazyLock::new(CaptureCoordinator::default);

/// Handle to stop audio capture. This is Send-safe.
pub struct CaptureHandle {
    stop: Arc<AtomicBool>,
    dropped: Arc<std::sync::atomic::AtomicU64>,
    captured: Arc<std::sync::atomic::AtomicU64>,
    permit: Option<CapturePermit>,
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

    /// Windows dropped because the consumer fell behind (channel full).
    pub fn dropped_windows(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Stop and join capture, returning the exact end of all attempted windows,
    /// including any dropped tail windows. Queued chunks can still be drained.
    pub fn finish(&mut self) -> u64 {
        self.stop();
        if let Some(thread) = self._thread.take() {
            let _ = thread.join();
        }
        self.permit.take();
        self.captured.load(Ordering::Acquire)
    }
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Record audio from the default input device.
/// Returns a receiver of VAD-window-sized audio chunks at 16kHz and a Send-safe stop handle.
///
/// DSP parameters:
/// - `hpf_cutoff_hz`: High-pass filter cutoff frequency (e.g. 200.0)
/// - `noise_gate_rms`: RMS threshold below which windows are zeroed (e.g. 0.01)
/// - `window_size`: Number of samples per VAD window (e.g. 512)
pub fn start_capture(
    hpf_cutoff_hz: f64,
    noise_gate_rms: f32,
    window_size: usize,
) -> Result<(mpsc::Receiver<AudioChunk>, CaptureHandle)> {
    start_capture_with(hpf_cutoff_hz, noise_gate_rms, window_size, 64)
}

/// Like [`start_capture`] with an explicit channel capacity (in windows).
/// Long recordings use a large capacity so a slow consumer doesn't lose audio;
/// any drops that still happen are counted in [`CaptureHandle::dropped_windows`].
/// A `noise_gate_rms` of 0 disables the gate (use for archival capture).
pub fn start_capture_with(
    hpf_cutoff_hz: f64,
    noise_gate_rms: f32,
    window_size: usize,
    channel_capacity: usize,
) -> Result<(mpsc::Receiver<AudioChunk>, CaptureHandle)> {
    if window_size == 0 {
        return Err(VoiceError::Audio(
            "capture window_size must be positive".into(),
        ));
    }
    let permit = CAPTURE_COORDINATOR.acquire()?;
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| VoiceError::Audio("No input device (microphone) available".to_string()))?;

    let supported_config = device
        .default_input_config()
        .map_err(|e| VoiceError::Audio(format!("Failed to get input config: {e}")))?;

    let device_sample_rate = supported_config.sample_rate();
    let device_channels = supported_config.channels() as usize;

    tracing::debug!(
        device = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_default(),
        sample_rate = device_sample_rate,
        channels = device_channels,
        "Starting audio capture"
    );

    let (tx, rx) = mpsc::channel::<AudioChunk>(channel_capacity.max(1));
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();
    let dropped = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let dropped_cb = dropped.clone();
    let captured = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let captured_cb = captured.clone();
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);

    // Spawn a dedicated thread that owns the cpal::Stream (which is not Send).
    // This thread runs until the stop flag is set.
    let thread = std::thread::spawn(move || {
        let config = cpal::StreamConfig {
            channels: supported_config.channels(),
            sample_rate: supported_config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };

        let stop_inner = stop_clone.clone();
        let stop_on_error = stop_clone.clone();
        let mut accumulator: Vec<f32> = Vec::with_capacity(window_size * 4);
        let mut window_buf = vec![0.0f32; window_size];
        let mut hpf = BiquadHPF::highpass(hpf_cutoff_hz, device_sample_rate as f64);
        let mut resampler = StreamingResampler::new(device_sample_rate, TARGET_SAMPLE_RATE);
        let mut next_sample = 0u64;

        let stream = match device.build_input_stream(
            config,
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

                // Apply high-pass filter to remove speaker bleed & room rumble
                let filtered: Vec<f32> =
                    mono.iter().map(|&s| hpf.process(s as f64) as f32).collect();

                // Resample to 16kHz. The resampler is stateful so callback
                // boundaries don't create discontinuities or drop fractional samples.
                resampler.process(&filtered, &mut accumulator);

                // Send complete VAD windows using pre-allocated buffer
                while accumulator.len() >= window_size {
                    window_buf.copy_from_slice(&accumulator[..window_size]);
                    accumulator.drain(..window_size);

                    // RMS noise gate: zero out windows below threshold to prevent
                    // quiet echo/noise from triggering false VAD detections
                    if noise_gate_rms > 0.0 {
                        apply_noise_gate(&mut window_buf, noise_gate_rms);
                    }

                    let chunk = AudioChunk {
                        start_sample: next_sample,
                        samples: window_buf.clone(),
                    };
                    next_sample += window_size as u64;
                    captured_cb.store(next_sample, Ordering::Release);
                    if tx.try_send(chunk).is_err() {
                        dropped_cb.fetch_add(1, Ordering::Relaxed);
                    }
                }
            },
            move |err| {
                tracing::error!("Audio input error: {err}");
                stop_on_error.store(true, Ordering::Release);
            },
            None,
        ) {
            Ok(s) => s,
            Err(e) => {
                let _ = ready_tx.send(Err(format!("Failed to build input stream: {e}")));
                return;
            }
        };

        if let Err(e) = stream.play() {
            let _ = ready_tx.send(Err(format!("Failed to start capture: {e}")));
            return;
        }
        let _ = ready_tx.send(Ok(()));

        // Keep the stream alive until stop is signaled
        while !stop_clone.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        drop(stream);
    });

    match ready_rx.recv() {
        Ok(Ok(())) => {}
        result => {
            stop.store(true, Ordering::Relaxed);
            let _ = thread.join();
            let message = match result {
                Ok(Err(e)) => e,
                _ => "Capture thread exited during startup".into(),
            };
            return Err(VoiceError::Audio(message));
        }
    }
    let handle = CaptureHandle {
        stop,
        dropped,
        captured,
        permit: Some(permit),
        _thread: Some(thread),
    };

    Ok((rx, handle))
}

/// Eagerly initialize the resampler kernel table to avoid first-call latency.
pub fn init() {
    let _ = &*KERNEL_TABLE;
}

/// One-shot Lanczos-3 resample of a complete buffer.
///
/// Equivalent to feeding `samples` through a fresh [`StreamingResampler`] and
/// flushing it. Produces `ceil(len * to / from)` samples.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let mut rs = StreamingResampler::new(from_rate, to_rate);
    let mut out = Vec::with_capacity(rs.expected_len(samples.len()) + 1);
    rs.process(samples, &mut out);
    rs.flush(&mut out);
    out
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

// Bound each instance's coefficient table to 64 KiB (plus phase denominators).
// Common audio ratios fit; unusual co-prime rates retain the scalar LUT path.
const MAX_POLYPHASE_COEFFICIENTS: usize = 8_192;

struct PolyphaseKernel {
    weights: Box<[f64]>,
    weight_sums: Box<[f64]>,
    taps: usize,
}

impl PolyphaseKernel {
    fn new(phases: u64, scale: f64, half_width: i64) -> Option<Self> {
        let phases = usize::try_from(phases).ok()?;
        let taps = usize::try_from(half_width).ok()?.checked_mul(2)?;
        let coefficient_count = phases.checked_mul(taps)?;
        if coefficient_count > MAX_POLYPHASE_COEFFICIENTS {
            return None;
        }
        let mut weights = Vec::with_capacity(coefficient_count);
        let mut weight_sums = Vec::with_capacity(phases);
        for phase in 0..phases {
            let frac = phase as f64 / phases as f64;
            let mut weight_sum = 0.0;
            for offset in -half_width + 1..=half_width {
                let weight = lanczos3_table(((-offset) as f64 + frac) * scale);
                weights.push(weight);
                weight_sum += weight;
            }
            weight_sums.push(weight_sum);
        }
        Some(Self {
            weights: weights.into_boxed_slice(),
            weight_sums: weight_sums.into_boxed_slice(),
            taps,
        })
    }

    fn phase(&self, phase: usize) -> (&[f64], f64) {
        let start = phase * self.taps;
        (
            &self.weights[start..start + self.taps],
            self.weight_sums[phase],
        )
    }
}

/// Stateful Lanczos-3 windowed-sinc resampler for streaming audio.
///
/// - **Anti-aliased**: when downsampling, the kernel is stretched by
///   `from/to` so its cutoff sits at the *target* Nyquist (8 kHz for 16 kHz
///   output) instead of the source Nyquist. Without this, 8–24 kHz energy
///   from a 48 kHz mic folds back into the speech band.
/// - **Chunk-invariant**: output positions are tracked with exact integer
///   arithmetic (`k * from / to`) over absolute sample indices, and input is
///   buffered until every tap is available. Feeding audio in any chunking
///   yields bit-identical output to feeding it all at once, with no drift over
///   hours of capture.
/// - **Unity DC gain**: weights are normalized per output sample.
pub struct StreamingResampler {
    from: u64,
    to: u64,
    scale: f64,
    half_width: i64,
    kernel: Option<PolyphaseKernel>,
    buf: Vec<f32>,
    /// Absolute source index of `buf[0]`.
    base: u64,
    /// Absolute index of the next output sample.
    next_out: u64,
}

impl StreamingResampler {
    pub fn new(from_rate: u32, to_rate: u32) -> Self {
        assert!(
            from_rate > 0 && to_rate > 0,
            "sample rates must be non-zero"
        );
        let g = gcd(from_rate as u64, to_rate as u64);
        let (from, to) = (from_rate as u64 / g, to_rate as u64 / g);
        let scale = (to as f64 / from as f64).min(1.0);
        let half_width = (3.0 / scale).ceil() as i64;
        Self {
            from,
            to,
            scale,
            half_width,
            kernel: (from != to)
                .then(|| PolyphaseKernel::new(to, scale, half_width))
                .flatten(),
            buf: Vec::new(),
            base: 0,
            next_out: 0,
        }
    }

    /// Number of output samples a complete (flushed) input of `n` samples yields.
    pub fn expected_len(&self, n: usize) -> usize {
        (n as u64 * self.to).div_ceil(self.from) as usize
    }

    fn is_passthrough(&self) -> bool {
        self.from == self.to
    }

    #[inline]
    fn position(&self, k: u64) -> (u64, u64) {
        let num = k * self.from;
        (num / self.to, num % self.to)
    }

    /// One output sample. Taps outside the available input (stream start, or
    /// the right edge during `flush`) reuse the nearest edge sample rather than
    /// truncating the kernel: renormalizing a truncated kernel amplifies
    /// ringing (up to ~2.8x at the last sample), while edge extension keeps
    /// DC exact and overshoot at interior levels.
    #[inline]
    fn compute(&self, ip: u64, phase: u64, end: u64) -> f32 {
        let lo = ip as i64 - self.half_width + 1;
        let hi = ip as i64 + self.half_width;
        let first = self.base as i64;
        let last = end as i64 - 1;
        let (sum, weight_sum) = if let Some(kernel) = &self.kernel {
            let (weights, weight_sum) = kernel.phase(phase as usize);
            let mut sum = 0.0_f64;
            if lo >= first && hi <= last {
                // Most samples use contiguous input, with no per-tap clamping
                // or coefficient interpolation. Keep the scalar accumulation
                // order and final division identical to the fallback.
                let start = (lo - first) as usize;
                for (&sample, &weight) in self.buf[start..start + weights.len()].iter().zip(weights)
                {
                    sum += sample as f64 * weight;
                }
            } else {
                for (offset, &weight) in weights.iter().enumerate() {
                    let idx = (lo + offset as i64).clamp(first, last);
                    sum += self.buf[(idx - first) as usize] as f64 * weight;
                }
            }
            (sum, weight_sum)
        } else {
            let frac = phase as f64 / self.to as f64;
            let mut sum = 0.0_f64;
            let mut weight_sum = 0.0_f64;
            for j in lo..=hi {
                let x = (ip as i64 - j) as f64 + frac;
                let w = lanczos3_table(x * self.scale);
                let idx = j.clamp(first, last);
                sum += self.buf[(idx - first) as usize] as f64 * w;
                weight_sum += w;
            }
            (sum, weight_sum)
        };
        if weight_sum.abs() > 1e-10 {
            (sum / weight_sum) as f32
        } else {
            0.0
        }
    }

    /// Feed input and append every output sample whose taps are now available.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.is_passthrough() {
            out.extend_from_slice(input);
            return;
        }
        self.buf.extend_from_slice(input);
        let end = self.base + self.buf.len() as u64;
        loop {
            let (ip, phase) = self.position(self.next_out);
            if ip as i64 + self.half_width >= end as i64 {
                break;
            }
            out.push(self.compute(ip, phase, end));
            self.next_out += 1;
        }
        // Drop input no future output can reach.
        let (ip, _) = self.position(self.next_out);
        let keep_from = (ip as i64 - self.half_width + 1).max(self.base as i64) as u64;
        let drop = ((keep_from - self.base) as usize).min(self.buf.len());
        self.buf.drain(..drop);
        self.base += drop as u64;
    }

    /// Emit the remaining outputs (right edge uses only available samples).
    /// The resampler should not be fed again after flushing.
    pub fn flush(&mut self, out: &mut Vec<f32>) {
        if self.is_passthrough() {
            return;
        }
        let end = self.base + self.buf.len() as u64;
        loop {
            let (ip, phase) = self.position(self.next_out);
            if ip >= end {
                break;
            }
            out.push(self.compute(ip, phase, end));
            self.next_out += 1;
        }
        self.buf.clear();
        self.base = end;
    }
}

// Precomputed Lanczos-3 kernel lookup table.
// Covers [0, 3) with KERNEL_TABLE_SIZE entries. Symmetric, so we use abs(x).
const KERNEL_TABLE_SIZE: usize = 512;
const KERNEL_TABLE_SCALE: f64 = KERNEL_TABLE_SIZE as f64 / 3.0;

/// Build the kernel table at compile time is not possible with sin(),
/// so we use std::sync::LazyLock for one-time init.
static KERNEL_TABLE: std::sync::LazyLock<[f32; KERNEL_TABLE_SIZE]> =
    std::sync::LazyLock::new(|| {
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
    fn capture_permit_rejects_overlap_and_releases_on_drop() {
        let coordinator = CaptureCoordinator::default();
        let permit = coordinator.acquire().unwrap();
        assert!(coordinator.acquire().is_err());
        drop(permit);
        assert!(coordinator.acquire().is_ok());
    }

    #[test]
    fn capture_finish_joins_and_reports_tail_before_releasing_permit() {
        let coordinator = CaptureCoordinator::default();
        let permit = coordinator.acquire().unwrap();
        let captured = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let worker_captured = captured.clone();
        let mut handle = CaptureHandle {
            stop: Arc::new(AtomicBool::new(false)),
            dropped: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            captured,
            permit: Some(permit),
            _thread: Some(std::thread::spawn(move || {
                worker_captured.store(1536, Ordering::Release);
            })),
        };
        assert!(coordinator.acquire().is_err());
        assert_eq!(handle.finish(), 1536);
        assert_eq!(handle.finish(), 1536); // Idempotent.
        assert!(coordinator.acquire().is_ok());
    }

    #[test]
    fn polyphase_matches_scalar_fallback_bit_for_bit() {
        let rates = [8_000, 16_000, 22_050, 24_000, 44_100, 48_000];
        for from in rates {
            for to in rates {
                for len in [0, 1, 2, 17, 257] {
                    let input: Vec<f32> = (0..len)
                        .map(|i| ((i * 73 + 19) % 251) as f32 / 125.0 - 1.0)
                        .collect();
                    let mut scalar = StreamingResampler::new(from, to);
                    scalar.kernel = None;
                    let mut expected = Vec::new();
                    scalar.process(&input, &mut expected);
                    scalar.flush(&mut expected);
                    let expected_bits: Vec<_> = expected.into_iter().map(f32::to_bits).collect();
                    for chunk_size in [1, 7, 128] {
                        let mut cached = StreamingResampler::new(from, to);
                        let mut actual = Vec::new();
                        for chunk in input.chunks(chunk_size) {
                            cached.process(chunk, &mut actual);
                        }
                        cached.flush(&mut actual);
                        let actual_bits: Vec<_> = actual.into_iter().map(f32::to_bits).collect();
                        assert_eq!(
                            actual_bits, expected_bits,
                            "{from}->{to}, len={len}, chunk={chunk_size}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn uncommon_ratio_fallback_remains_chunk_invariant() {
        let input: Vec<f32> = (0..2_003).map(|i| (i as f32 * 0.17).sin()).collect();
        let mut fallback = StreamingResampler::new(16_000, 16_001);
        assert!(fallback.kernel.is_none());
        let expected = resample(&input, 16_000, 16_001);
        let mut actual = Vec::new();
        for chunk in input.chunks(13) {
            fallback.process(chunk, &mut actual);
        }
        fallback.flush(&mut actual);
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), fallback.expected_len(input.len()));
    }

    #[test]
    fn polyphase_cache_is_bounded_and_passthrough_needs_none() {
        for (from, to) in [(48_000, 16_000), (44_100, 16_000), (16_000, 44_100)] {
            let rs = StreamingResampler::new(from, to);
            let kernel = rs.kernel.as_ref().unwrap();
            assert!(kernel.weights.len() <= MAX_POLYPHASE_COEFFICIENTS);
            assert!(
                kernel
                    .weight_sums
                    .iter()
                    .all(|w| w.is_finite() && w.abs() > 1e-10)
            );
        }
        assert!(StreamingResampler::new(16_000, 16_000).kernel.is_none());
        for (from, to) in [(16_000, 16_001), (1, u32::MAX), (u32::MAX, 1)] {
            assert!(StreamingResampler::new(from, to).kernel.is_none());
        }
    }

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

    // --- BiquadHPF tests ---

    #[test]
    fn biquad_hpf_passes_high_frequency() {
        // 1kHz sine at 16kHz sample rate should pass through a 200Hz HPF
        let mut hpf = BiquadHPF::highpass(200.0, 16000.0);
        let freq = 1000.0;
        let sr = 16000.0;
        let input: Vec<f64> = (0..1600)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin())
            .collect();
        let output: Vec<f64> = input.iter().map(|&s| hpf.process(s)).collect();

        // After settling (skip first 100 samples), output should have significant energy
        let rms: f64 =
            (output[100..].iter().map(|s| s * s).sum::<f64>() / (output.len() - 100) as f64).sqrt();
        assert!(
            rms > 0.5,
            "1kHz signal should pass through 200Hz HPF, got RMS={rms}"
        );
    }

    #[test]
    fn biquad_hpf_attenuates_low_frequency() {
        // 50Hz sine at 16kHz sample rate should be heavily attenuated by 200Hz HPF
        let mut hpf = BiquadHPF::highpass(200.0, 16000.0);
        let freq = 50.0;
        let sr = 16000.0;
        let input: Vec<f64> = (0..16000)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin())
            .collect();
        let output: Vec<f64> = input.iter().map(|&s| hpf.process(s)).collect();

        // After settling, output RMS should be much lower than input
        let input_rms: f64 =
            (input[1000..].iter().map(|s| s * s).sum::<f64>() / (input.len() - 1000) as f64).sqrt();
        let output_rms: f64 = (output[1000..].iter().map(|s| s * s).sum::<f64>()
            / (output.len() - 1000) as f64)
            .sqrt();
        let attenuation = output_rms / input_rms;
        assert!(
            attenuation < 0.2,
            "50Hz should be attenuated by 200Hz HPF, got ratio={attenuation}"
        );
    }

    #[test]
    fn biquad_hpf_removes_dc() {
        // DC (constant) signal should be fully removed by HPF
        let mut hpf = BiquadHPF::highpass(200.0, 16000.0);
        let output: Vec<f64> = (0..1600).map(|_| hpf.process(1.0)).collect();
        // After settling, output should be near zero
        let tail_rms: f64 =
            (output[200..].iter().map(|s| s * s).sum::<f64>() / (output.len() - 200) as f64).sqrt();
        assert!(
            tail_rms < 0.01,
            "DC signal should be removed by HPF, got RMS={tail_rms}"
        );
    }

    #[test]
    fn biquad_hpf_coefficients_sane() {
        let hpf = BiquadHPF::highpass(200.0, 48000.0);
        // b0 should be close to 1.0 for a high-pass (passes high freq at unity)
        assert!(hpf.b0 > 0.9 && hpf.b0 < 1.1, "b0={}", hpf.b0);
        // b1 should be approximately -2*b0
        assert!(
            (hpf.b1 + 2.0 * hpf.b0).abs() < 0.1,
            "b1={}, expected ~{}",
            hpf.b1,
            -2.0 * hpf.b0
        );
    }

    // --- Noise gate tests ---

    #[test]
    fn noise_gate_threshold_zeros_quiet_window() {
        let mut window = vec![0.001f32; 512];
        apply_noise_gate(&mut window, 0.01);
        assert!(window.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn noise_gate_threshold_passes_loud_window() {
        let mut window = vec![0.1f32; 512];
        let original = window.clone();
        apply_noise_gate(&mut window, 0.01);
        // Should NOT zero the window
        assert_eq!(window, original);
    }

    // --- Peak normalize tests ---

    #[test]
    fn peak_normalize_amplifies_quiet_audio() {
        let mut samples = vec![0.1, -0.2, 0.15, -0.05];
        let peak = peak_normalize(&mut samples, 0.5);
        assert!((peak - 0.2).abs() < f32::EPSILON);
        // After normalization, peak should be 1.0
        let new_peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!((new_peak - 1.0).abs() < 0.001);
    }

    #[test]
    fn peak_normalize_skips_loud_audio() {
        let mut samples = vec![0.6, -0.8, 0.7];
        let original = samples.clone();
        let peak = peak_normalize(&mut samples, 0.5);
        assert!((peak - 0.8).abs() < f32::EPSILON);
        // Should not modify — peak is above threshold
        assert_eq!(samples, original);
    }

    #[test]
    fn peak_normalize_handles_silence() {
        let mut samples = vec![0.0, 0.0, 0.0];
        let peak = peak_normalize(&mut samples, 0.5);
        assert!((peak - 0.0).abs() < f32::EPSILON);
        // Should stay zero
        assert!(samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn peak_normalize_handles_empty() {
        let mut samples: Vec<f32> = vec![];
        let peak = peak_normalize(&mut samples, 0.5);
        assert!((peak - 0.0).abs() < f32::EPSILON);
    }

    // --- Public DSP function tests ---

    #[test]
    fn apply_highpass_filter_passes_high_freq() {
        // 1kHz sine at 16kHz sample rate should pass through 200Hz HPF
        let sr = 16000.0;
        let freq = 1000.0;
        let mut samples: Vec<f32> = (0..1600)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();
        let input_rms: f32 = (samples[100..].iter().map(|s| s * s).sum::<f32>()
            / (samples.len() - 100) as f32)
            .sqrt();

        apply_highpass_filter(&mut samples, 200.0, sr);

        let output_rms: f32 = (samples[100..].iter().map(|s| s * s).sum::<f32>()
            / (samples.len() - 100) as f32)
            .sqrt();
        // High freq should pass with minimal loss
        assert!(
            output_rms / input_rms > 0.8,
            "1kHz should pass, ratio={}",
            output_rms / input_rms
        );
    }

    #[test]
    fn apply_highpass_filter_attenuates_low_freq() {
        // 50Hz sine at 16kHz sample rate should be attenuated
        let sr = 16000.0;
        let freq = 50.0;
        let mut samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();

        apply_highpass_filter(&mut samples, 200.0, sr);

        let output_rms: f32 = (samples[1000..].iter().map(|s| s * s).sum::<f32>()
            / (samples.len() - 1000) as f32)
            .sqrt();
        // Input RMS of a sine is ~0.707
        assert!(
            output_rms < 0.15,
            "50Hz should be attenuated, got RMS={output_rms}"
        );
    }

    #[test]
    fn apply_noise_gate_zeros_quiet() {
        let mut window = vec![0.001f32; 512];
        apply_noise_gate(&mut window, 0.01);
        assert!(window.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn apply_noise_gate_passes_loud() {
        let mut window = vec![0.5f32; 512];
        apply_noise_gate(&mut window, 0.01);
        assert!(window.iter().all(|&s| s == 0.5));
    }

    // --- Streaming playback tests ---

    #[test]
    fn streaming_playback_handles_empty_channel() {
        // Dropping the sender immediately should not panic
        let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
        drop(tx);
        // The receiver should return immediately with no chunks
        assert!(rx.recv().is_err());
    }
}
