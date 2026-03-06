use crate::audio::{self, CaptureHandle};
use crate::config::{Config, SharedConfig};
use crate::models;
use crate::stt::SttEngine;
use crate::tts::TtsEngine;
use crate::vad::VadSession;

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Mutex as TokioMutex;

static SAY_COUNTER: AtomicU64 = AtomicU64::new(1);
const VAD_MAX_SPEECH_SECS: f32 = 86400.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboxMessage {
    pub text: String,
    pub timestamp: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
}

#[derive(Debug, Clone)]
struct QueuedSpeech {
    id: String,
    message: String,
    voice: Option<String>,
    speed: f32,
    enqueued_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SpeechQueueItemInfo {
    id: String,
    message_chars: usize,
    preview: String,
    voice: Option<String>,
    speed: f32,
    enqueued_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TtsQueueStatus {
    worker_active: bool,
    currently_speaking: Option<SpeechQueueItemInfo>,
    pending_count: usize,
    pending: Vec<SpeechQueueItemInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TtsQueueClearResult {
    cleared: usize,
    pending_count: usize,
    currently_speaking: bool,
}

impl QueuedSpeech {
    fn info(&self) -> SpeechQueueItemInfo {
        SpeechQueueItemInfo {
            id: self.id.clone(),
            message_chars: self.message.chars().count(),
            preview: preview_text(&self.message),
            voice: self.voice.clone(),
            speed: self.speed,
            enqueued_at: self.enqueued_at.clone(),
        }
    }
}

fn preview_text(text: &str) -> String {
    const MAX_PREVIEW_CHARS: usize = 80;
    let mut preview = String::new();

    for (i, ch) in text.chars().enumerate() {
        if i >= MAX_PREVIEW_CHARS {
            preview.push_str("...");
            break;
        }
        preview.push(ch);
    }

    preview
}

fn queue_push(queue: &mut VecDeque<QueuedSpeech>, item: QueuedSpeech) -> usize {
    queue.push_back(item);
    queue.len()
}

fn queue_pop(queue: &mut VecDeque<QueuedSpeech>) -> Option<QueuedSpeech> {
    queue.pop_front()
}

fn queue_status_snapshot(
    worker_active: bool,
    current: Option<&QueuedSpeech>,
    queue: &VecDeque<QueuedSpeech>,
) -> TtsQueueStatus {
    let pending_items = queue.iter().map(QueuedSpeech::info).collect::<Vec<_>>();
    TtsQueueStatus {
        worker_active,
        currently_speaking: current.map(QueuedSpeech::info),
        pending_count: pending_items.len(),
        pending: pending_items,
    }
}

fn clear_pending_queue(
    queue: &mut VecDeque<QueuedSpeech>,
    currently_speaking: bool,
) -> TtsQueueClearResult {
    let cleared = queue.len();
    queue.clear();
    TtsQueueClearResult {
        cleared,
        pending_count: 0,
        currently_speaking,
    }
}

pub(crate) fn chrono_now() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;
    let years = 1970 + days / 365;
    let remaining_days = days % 365;
    let months = remaining_days / 30 + 1;
    let day = remaining_days % 30 + 1;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        years, months, day, hours, minutes, seconds
    )
}

#[derive(Clone)]
pub(crate) struct VoiceRuntime {
    tts: Arc<Mutex<TtsEngine>>,
    tts_swap_lock: Arc<TokioMutex<()>>,
    tts_playback_lock: Arc<TokioMutex<()>>,
    stt: Arc<Mutex<SttEngine>>,
    config: SharedConfig,
    inbox: Arc<Mutex<Vec<InboxMessage>>>,
    bg_capture: Arc<Mutex<Option<CaptureHandle>>>,
    bg_active: Arc<AtomicBool>,
    tts_queue: Arc<TokioMutex<VecDeque<QueuedSpeech>>>,
    tts_current: Arc<TokioMutex<Option<QueuedSpeech>>>,
    tts_worker_active: Arc<AtomicBool>,
}

impl VoiceRuntime {
    pub(crate) fn new(tts: TtsEngine, stt: SttEngine, config: SharedConfig) -> Self {
        Self::with_shared(Arc::new(Mutex::new(tts)), Arc::new(Mutex::new(stt)), config)
    }

    pub(crate) fn with_shared(
        tts: Arc<Mutex<TtsEngine>>,
        stt: Arc<Mutex<SttEngine>>,
        config: SharedConfig,
    ) -> Self {
        Self {
            tts,
            tts_swap_lock: Arc::new(TokioMutex::new(())),
            tts_playback_lock: Arc::new(TokioMutex::new(())),
            stt,
            config,
            inbox: Arc::new(Mutex::new(Vec::new())),
            bg_capture: Arc::new(Mutex::new(None)),
            bg_active: Arc::new(AtomicBool::new(false)),
            tts_queue: Arc::new(TokioMutex::new(VecDeque::new())),
            tts_current: Arc::new(TokioMutex::new(None)),
            tts_worker_active: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn config_snapshot(&self) -> std::result::Result<Config, String> {
        self.config
            .read()
            .map_err(|e| format!("Config lock poisoned: {e}"))
            .map(|c| c.clone())
    }

    pub(crate) fn push_inbox(&self, message: InboxMessage) {
        if let Ok(mut inbox) = self.inbox.lock() {
            inbox.push(message);
        }
    }

    pub(crate) fn drain_inbox(&self) -> Vec<InboxMessage> {
        let mut inbox = self.inbox.lock().unwrap();
        inbox.drain(..).collect()
    }

    pub(crate) async fn speak(
        &self,
        text: &str,
        voice: Option<&str>,
        speed: Option<f32>,
    ) -> std::result::Result<(), String> {
        let _playback_guard = self.tts_playback_lock.lock().await;
        self.ensure_tts_model_synced().await?;

        let config = self.config_snapshot()?;
        let selected_voice = voice.unwrap_or(&config.voice).to_string();
        let selected_speed = speed.unwrap_or(config.speed);

        let tts = self.tts.clone();
        let text = text.to_string();

        let (std_tx, std_rx) = std::sync::mpsc::channel::<Vec<f32>>();

        let sample_rate = {
            let tts = tts.lock().map_err(|e| format!("TTS lock: {e}"))?;
            tts.sample_rate()
        };

        const SENTENCE_STREAMING_THRESHOLD: usize = 150;
        let use_sentence_streaming = text.len() >= SENTENCE_STREAMING_THRESHOLD
            && text
                .chars()
                .filter(|&c| c == '.' || c == '!' || c == '?')
                .count()
                >= 2;

        let t_start = std::time::Instant::now();
        let text_len_for_log = text.len();

        let producer = tokio::task::spawn_blocking(move || {
            let mut tts = tts.lock().map_err(|e| format!("TTS lock: {e}"))?;
            let sid = crate::tts::resolve_voice_id(&selected_voice);

            if use_sentence_streaming {
                tracing::debug!("Using sentence-first streaming for low latency");
                tts.synthesize_sentences_streaming(&text, sid, selected_speed, std_tx)
                    .map_err(|e| format!("TTS failed: {e}"))
            } else {
                tts.synthesize_streaming(&text, sid, selected_speed, std_tx)
                    .map_err(|e| format!("TTS failed: {e}"))
            }
        });

        let consumer = audio::play_audio_streaming(std_rx, sample_rate);

        let (prod_result, cons_result) = tokio::join!(producer, consumer);
        let t_total = t_start.elapsed();

        prod_result.map_err(|e| format!("Producer: {e}"))??;
        cons_result.map_err(|e| format!("Playback: {e}"))?;

        tracing::debug!(
            total_ms = t_total.as_millis() as u64,
            use_sentence_streaming,
            text_len = text_len_for_log,
            "TTS streaming complete"
        );

        Ok(())
    }

    pub(crate) async fn enqueue_speech(
        &self,
        message: String,
        voice: Option<String>,
        speed: f32,
    ) -> (String, usize) {
        let id = Self::new_speech_id();
        let queued = QueuedSpeech {
            id: id.clone(),
            message,
            voice,
            speed,
            enqueued_at: chrono_now(),
        };

        let pending_count = {
            let mut queue = self.tts_queue.lock().await;
            queue_push(&mut queue, queued)
        };

        self.ensure_tts_queue_worker();
        (id, pending_count)
    }

    pub(crate) async fn tts_queue_status_snapshot(&self) -> TtsQueueStatus {
        let current = { self.tts_current.lock().await.clone() };
        let queue = self.tts_queue.lock().await;
        queue_status_snapshot(
            self.tts_worker_active.load(Ordering::Relaxed),
            current.as_ref(),
            &queue,
        )
    }

    pub(crate) async fn clear_tts_queue(&self) -> TtsQueueClearResult {
        let currently_speaking = self.tts_current.lock().await.is_some();
        let mut queue = self.tts_queue.lock().await;
        clear_pending_queue(&mut queue, currently_speaking)
    }

    pub(crate) async fn record_and_transcribe(
        &self,
        silence_timeout_ms: u32,
        min_speech_ms: Option<u32>,
    ) -> std::result::Result<String, String> {
        if self.bg_active.load(Ordering::Relaxed) {
            return Err("Background listener is active. Call stop_listening first.".to_string());
        }
        let config = self.config_snapshot()?;
        let max_speech_secs = VAD_MAX_SPEECH_SECS;

        let (mut rx, capture_handle) = audio::start_capture(
            config.dsp.hpf_cutoff_hz,
            config.dsp.noise_gate_rms,
            config.dsp.noise_gate_window,
        )
        .map_err(|e| format!("Capture failed: {e}"))?;

        let mut all_speech_samples: Vec<f32> = Vec::new();
        let mut vad = VadSession::new(&config, max_speech_secs)
            .map_err(|e| format!("VAD init failed: {e}"))?;

        let mut speech_started = false;
        let base_silence_threshold = std::time::Duration::from_millis(silence_timeout_ms as u64);

        const SIGNIFICANT_SPEECH_MS: u64 = 3000;
        const EXTENDED_SILENCE_BONUS_MS: u64 = 1000;

        let mut last_speech_time = std::time::Instant::now();
        let mut first_speech_time: Option<std::time::Instant> = None;
        let mut total_speech_duration = std::time::Duration::ZERO;
        let mut last_speech_segment_start: Option<std::time::Instant> = None;

        const GRACE_WINDOW_MS: u64 = 800;
        let mut pending_end_time: Option<std::time::Instant> = None;

        loop {
            let chunk = tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv()).await;

            match chunk {
                Ok(Some(chunk)) => {
                    vad.accept_waveform(chunk.samples);

                    let is_speech = vad.is_speech();

                    if is_speech {
                        speech_started = true;
                        let now = std::time::Instant::now();
                        last_speech_time = now;

                        if first_speech_time.is_none() {
                            first_speech_time = Some(now);
                            tracing::debug!("Speech started");
                        }

                        if last_speech_segment_start.is_none() {
                            last_speech_segment_start = Some(now);
                        }

                        if pending_end_time.is_some() {
                            pending_end_time = None;
                            tracing::debug!("Speech resumed, cancelling pending end-of-turn");
                        }

                        let segments = vad.collect_segments();
                        for seg in segments {
                            all_speech_samples.extend_from_slice(&seg.samples);
                        }
                    } else {
                        if let Some(seg_start) = last_speech_segment_start.take() {
                            let segment_duration = seg_start.elapsed();
                            total_speech_duration += segment_duration;
                            tracing::debug!(
                                segment_duration_ms = segment_duration.as_millis() as u64,
                                total_speech_duration_ms = total_speech_duration.as_millis() as u64,
                                "Speech segment ended"
                            );
                        }

                        if speech_started {
                            let silence_elapsed = last_speech_time.elapsed();
                            let effective_threshold = if total_speech_duration.as_millis() as u64
                                >= SIGNIFICANT_SPEECH_MS
                            {
                                base_silence_threshold
                                    + std::time::Duration::from_millis(EXTENDED_SILENCE_BONUS_MS)
                            } else {
                                base_silence_threshold
                            };

                            if silence_elapsed >= effective_threshold {
                                if let Some(min_ms) = min_speech_ms
                                    && let Some(first) = first_speech_time
                                    && first.elapsed()
                                        < std::time::Duration::from_millis(min_ms as u64)
                                {
                                    continue;
                                }

                                match pending_end_time {
                                    None => {
                                        pending_end_time = Some(std::time::Instant::now());
                                        tracing::debug!(
                                            silence_elapsed_ms = silence_elapsed.as_millis() as u64,
                                            effective_threshold_ms =
                                                effective_threshold.as_millis() as u64,
                                            "Silence threshold reached, starting grace window"
                                        );
                                    }
                                    Some(started) => {
                                        if started.elapsed()
                                            >= std::time::Duration::from_millis(GRACE_WINDOW_MS)
                                        {
                                            tracing::debug!(
                                                total_speech_duration_ms =
                                                    total_speech_duration.as_millis() as u64,
                                                grace_window_ms = GRACE_WINDOW_MS,
                                                "Grace window expired, confirming end-of-turn"
                                            );
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(None) => {
                    tracing::debug!("Audio channel closed, stopping");
                    break;
                }
                Err(_) => {
                    if let Some(started) = pending_end_time
                        && started.elapsed() >= std::time::Duration::from_millis(GRACE_WINDOW_MS)
                    {
                        tracing::debug!(
                            total_speech_duration_ms = total_speech_duration.as_millis() as u64,
                            grace_window_ms = GRACE_WINDOW_MS,
                            "Grace window expired (timeout), confirming end-of-turn"
                        );
                        break;
                    }
                }
            }
        }

        capture_handle.stop();

        vad.flush();
        let segments = vad.collect_segments();
        for seg in segments {
            all_speech_samples.extend_from_slice(&seg.samples);
        }

        if all_speech_samples.is_empty() {
            return Ok("(no speech detected)".to_string());
        }

        audio::peak_normalize(&mut all_speech_samples, config.dsp.normalize_threshold);

        tracing::debug!(
            num_samples = all_speech_samples.len(),
            duration_secs = all_speech_samples.len() as f32 / 16000.0,
            "Transcribing captured speech"
        );

        let stt = self.stt.clone();
        let text: String = tokio::task::spawn_blocking(move || {
            let mut stt = stt.lock().map_err(|e| format!("STT lock poisoned: {e}"))?;
            stt.transcribe(16000, &all_speech_samples)
                .map_err(|e| format!("STT failed: {e}"))
        })
        .await
        .map_err(|e| format!("STT task failed: {e}"))??;

        Ok(text)
    }

    pub(crate) async fn start_background_listening(&self) -> std::result::Result<String, String> {
        if self.bg_active.load(Ordering::Relaxed) {
            return Err("Background listener is already active".to_string());
        }

        let config = self.config_snapshot()?;
        let (mut rx, capture_handle) = audio::start_capture(
            config.dsp.hpf_cutoff_hz,
            config.dsp.noise_gate_rms,
            config.dsp.noise_gate_window,
        )
        .map_err(|e| format!("Capture failed: {e}"))?;

        {
            let mut bg = self.bg_capture.lock().unwrap();
            *bg = Some(capture_handle);
        }
        self.bg_active.store(true, Ordering::Relaxed);

        let runtime = self.clone();
        let config_clone = config.clone();

        tokio::spawn(async move {
            let mut vad = match VadSession::new(&config_clone, 300.0) {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!("Background VAD init failed: {e}");
                    runtime.bg_active.store(false, Ordering::Relaxed);
                    return;
                }
            };

            let mut speech_samples: Vec<f32> = Vec::new();
            let mut speech_active = false;
            let mut last_speech = std::time::Instant::now();
            let silence_threshold = std::time::Duration::from_millis(1500);

            loop {
                if !runtime.bg_active.load(Ordering::Relaxed) {
                    break;
                }

                let chunk =
                    tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await;

                match chunk {
                    Ok(Some(chunk)) => {
                        vad.accept_waveform(chunk.samples);

                        if vad.is_speech() {
                            speech_active = true;
                            last_speech = std::time::Instant::now();
                            let segments = vad.collect_segments();
                            for seg in segments {
                                speech_samples.extend_from_slice(&seg.samples);
                            }
                        } else if speech_active
                            && last_speech.elapsed() >= silence_threshold
                            && !speech_samples.is_empty()
                        {
                            vad.flush();
                            let segments = vad.collect_segments();
                            for seg in segments {
                                speech_samples.extend_from_slice(&seg.samples);
                            }

                            audio::peak_normalize(
                                &mut speech_samples,
                                config_clone.dsp.normalize_threshold,
                            );

                            let samples = std::mem::take(&mut speech_samples);
                            let runtime_for_stt = runtime.clone();

                            tokio::task::spawn_blocking(move || {
                                let mut stt: std::sync::MutexGuard<'_, SttEngine> =
                                    match runtime_for_stt.stt.lock() {
                                        Ok(s) => s,
                                        Err(e) => {
                                            tracing::error!("STT lock failed: {e}");
                                            return;
                                        }
                                    };
                                match stt.transcribe(16000, &samples) {
                                    Ok(text) => {
                                        if !text.is_empty() && text != "(no speech detected)" {
                                            runtime_for_stt.push_inbox(InboxMessage {
                                                text,
                                                timestamp: chrono_now(),
                                                source: Some("background_listening".to_string()),
                                                conversation_id: None,
                                            });
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!("Background STT failed: {e}");
                                    }
                                }
                            });

                            speech_active = false;
                        }
                    }
                    Ok(None) => break,
                    Err(_) => {}
                }
            }
        });

        Ok("Background listening started".to_string())
    }

    pub(crate) async fn stop_background_listening(
        &self,
    ) -> std::result::Result<Vec<InboxMessage>, String> {
        if !self.bg_active.load(Ordering::Relaxed) {
            return Err("Background listener is not active".to_string());
        }

        {
            let mut bg = self.bg_capture.lock().unwrap();
            if let Some(handle) = bg.take() {
                handle.stop();
            }
        }
        self.bg_active.store(false, Ordering::Relaxed);
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        Ok(self.drain_inbox())
    }

    pub(crate) async fn reload_config(&self) -> std::result::Result<String, String> {
        Config::reload_into(&self.config).map_err(|e| e.to_string())?;
        self.ensure_tts_model_synced().await?;
        let config = self
            .config
            .read()
            .map_err(|e| format!("Config lock poisoned: {e}"))?;
        Ok(format!("Config reloaded:\n\n{}", config.display_all()))
    }

    pub(crate) async fn reset_dsp(&self) -> std::result::Result<String, String> {
        Config::reset_dsp().map_err(|e| e.to_string())?;
        Config::reload_into(&self.config).map_err(|e| e.to_string())?;
        let defaults = crate::config::DspConfig::default();
        Ok(format!(
            "DSP parameters reset to defaults and applied:\n  hpf_cutoff_hz:       {}\n  noise_gate_rms:      {}\n  noise_gate_window:   {}\n  normalize_threshold: {}",
            defaults.hpf_cutoff_hz,
            defaults.noise_gate_rms,
            defaults.noise_gate_window,
            defaults.normalize_threshold,
        ))
    }

    fn new_speech_id() -> String {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let n = SAY_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("say-{now_ms}-{n}")
    }

    fn ensure_tts_queue_worker(&self) {
        if self
            .tts_worker_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let runtime = self.clone();
            tokio::spawn(async move {
                runtime.run_tts_queue_worker().await;
            });
        }
    }

    async fn run_tts_queue_worker(self) {
        loop {
            let next = {
                let mut queue = self.tts_queue.lock().await;
                queue_pop(&mut queue)
            };

            let Some(item) = next else {
                self.tts_worker_active.store(false, Ordering::Release);

                let has_more = {
                    let queue = self.tts_queue.lock().await;
                    !queue.is_empty()
                };

                if has_more
                    && self
                        .tts_worker_active
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                {
                    continue;
                }

                break;
            };

            {
                let mut current = self.tts_current.lock().await;
                *current = Some(item.clone());
            }

            if let Err(e) = self
                .speak(&item.message, item.voice.as_deref(), Some(item.speed))
                .await
            {
                tracing::error!(speech_id = %item.id, "Queued speech failed: {e}");
                self.push_inbox(InboxMessage {
                    text: format!("queued speech {} failed: {e}", item.id),
                    timestamp: chrono_now(),
                    source: Some("tts_queue".to_string()),
                    conversation_id: None,
                });
            }

            {
                let mut current = self.tts_current.lock().await;
                *current = None;
            }
        }
    }

    async fn ensure_tts_model_synced(&self) -> std::result::Result<(), String> {
        let config = self.config_snapshot()?;
        let desired_model = config.kokoro_model;

        let current_model = {
            let tts = self.tts.lock().map_err(|e| format!("TTS lock: {e}"))?;
            tts.kokoro_model()
        };
        if current_model == desired_model {
            return Ok(());
        }

        let _guard = self.tts_swap_lock.lock().await;

        let current_model = {
            let tts = self.tts.lock().map_err(|e| format!("TTS lock: {e}"))?;
            tts.kokoro_model()
        };
        if current_model == desired_model {
            return Ok(());
        }

        tracing::info!(
            current = %current_model,
            target = %desired_model,
            "Kokoro model changed in config, preparing swap"
        );

        models::download_models_with_variant(&config, desired_model)
            .await
            .map_err(|e| format!("Model download failed: {e}"))?;

        let config_for_init = config.clone();
        let new_tts = tokio::task::spawn_blocking(move || TtsEngine::new(&config_for_init))
            .await
            .map_err(|e| format!("TTS init task failed: {e}"))?
            .map_err(|e| format!("TTS init failed: {e}"))?;

        let mut tts = self.tts.lock().map_err(|e| format!("TTS lock: {e}"))?;
        *tts = new_tts;

        tracing::info!(model = %desired_model, "Kokoro model swap complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queued(id: &str, message: &str) -> QueuedSpeech {
        QueuedSpeech {
            id: id.to_string(),
            message: message.to_string(),
            voice: Some("af_heart".to_string()),
            speed: 1.0,
            enqueued_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn chrono_now_produces_valid_format() {
        let ts = chrono_now();
        assert!(ts.ends_with('Z'));
        assert!(ts.contains('T'));
        assert_eq!(ts.len(), 20);
    }

    #[test]
    fn preview_text_truncates_long_messages() {
        let long = "a".repeat(90);
        let preview = preview_text(&long);
        assert_eq!(preview.len(), 83);
        assert!(preview.ends_with("..."));
    }

    #[test]
    fn queue_push_and_pop_preserve_fifo_order() {
        let mut q = VecDeque::new();
        assert_eq!(queue_push(&mut q, queued("one", "first")), 1);
        assert_eq!(queue_push(&mut q, queued("two", "second")), 2);

        let first = queue_pop(&mut q).unwrap();
        let second = queue_pop(&mut q).unwrap();
        assert_eq!(first.id, "one");
        assert_eq!(second.id, "two");
        assert!(queue_pop(&mut q).is_none());
    }

    #[test]
    fn queue_status_snapshot_reports_current_and_pending() {
        let mut q = VecDeque::new();
        queue_push(&mut q, queued("one", "first message"));
        queue_push(&mut q, queued("two", "second message"));
        let current = queued("cur", "currently speaking");

        let status = queue_status_snapshot(true, Some(&current), &q);
        assert!(status.worker_active);
        assert_eq!(status.pending_count, 2);
        assert_eq!(status.pending.len(), 2);
        assert_eq!(status.pending[0].id, "one");
        assert_eq!(status.currently_speaking.unwrap().id, "cur");
    }

    #[test]
    fn clear_pending_queue_clears_only_pending_items() {
        let mut q = VecDeque::new();
        queue_push(&mut q, queued("one", "first"));
        queue_push(&mut q, queued("two", "second"));

        let cleared = clear_pending_queue(&mut q, true);
        assert_eq!(cleared.cleared, 2);
        assert_eq!(cleared.pending_count, 0);
        assert!(cleared.currently_speaking);
        assert!(q.is_empty());
    }
}
