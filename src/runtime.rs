use crate::audio;
use crate::config::{Config, SharedConfig};
use crate::models;
use crate::recording::RecordingManager;
use crate::stt::SttEngine;
use crate::tts::TtsEngine;
use crate::vad::{VadOptions, VadSegment, VadSession};

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Mutex as TokioMutex;

static SAY_COUNTER: AtomicU64 = AtomicU64::new(1);

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

/// Current UTC time as RFC 3339 (`YYYY-MM-DDTHH:MM:SSZ`).
pub(crate) fn chrono_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format_utc(secs)
}

/// Format Unix seconds as RFC 3339 UTC.
pub(crate) fn format_utc(secs: u64) -> String {
    let (y, m, d) = civil_from_days((secs / 86400) as i64);
    let t = secs % 86400;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        t / 3600,
        (t % 3600) / 60,
        t % 60
    )
}

/// Days since 1970-01-01 → proleptic Gregorian (year, month, day).
/// Howard Hinnant's `civil_from_days`; exact for all i64 day counts we use.
pub(crate) fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// Completed segments arrive when VAD detection switches to silence.
struct BackgroundSpeech {
    samples: Vec<f32>,
    last_speech: std::time::Instant,
}

impl BackgroundSpeech {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            last_speech: std::time::Instant::now(),
        }
    }

    fn accept(&mut self, detected: bool, segments: Vec<VadSegment>) {
        if detected {
            self.last_speech = std::time::Instant::now();
        }
        for seg in segments {
            self.samples.extend(seg.samples);
        }
    }

    fn take_ready(&mut self, silence: std::time::Duration) -> Option<Vec<f32>> {
        if !self.samples.is_empty()
            && (self.last_speech.elapsed() >= silence || self.samples.len() >= 20 * 16_000)
        {
            Some(std::mem::take(&mut self.samples))
        } else {
            None
        }
    }
}

type BackgroundTask = tokio::task::JoinHandle<std::result::Result<(), String>>;

#[derive(Clone)]
pub(crate) struct VoiceRuntime {
    tts: Arc<Mutex<TtsEngine>>,
    tts_swap_lock: Arc<TokioMutex<()>>,
    tts_playback_lock: Arc<TokioMutex<()>>,
    stt: Arc<Mutex<SttEngine>>,
    stt_swap_lock: Arc<TokioMutex<()>>,
    config: SharedConfig,
    inbox: Arc<Mutex<Vec<InboxMessage>>>,
    bg_task: Arc<TokioMutex<Option<BackgroundTask>>>,
    bg_active: Arc<AtomicBool>,
    tts_queue: Arc<TokioMutex<VecDeque<QueuedSpeech>>>,
    tts_current: Arc<TokioMutex<Option<QueuedSpeech>>>,
    tts_worker_active: Arc<AtomicBool>,
    recordings: Arc<RecordingManager>,
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
        let recordings_root = config
            .read()
            .map(|c| c.recordings_dir())
            .unwrap_or_else(|_| crate::config::Config::default().recordings_dir());
        Self {
            recordings: Arc::new(RecordingManager::new(recordings_root, stt.clone())),
            tts,
            tts_swap_lock: Arc::new(TokioMutex::new(())),
            tts_playback_lock: Arc::new(TokioMutex::new(())),
            stt,
            stt_swap_lock: Arc::new(TokioMutex::new(())),
            config,
            inbox: Arc::new(Mutex::new(Vec::new())),
            bg_task: Arc::new(TokioMutex::new(None)),
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

    pub(crate) fn recordings(&self) -> &Arc<RecordingManager> {
        &self.recordings
    }

    pub(crate) async fn start_recording(
        &self,
        title: Option<String>,
        save_audio: bool,
    ) -> std::result::Result<crate::recording::JobView, String> {
        self.ensure_stt_model_synced().await?;
        self.recordings
            .start_live(&self.config_snapshot()?, title, save_audio)
    }

    pub(crate) async fn transcribe_file(
        &self,
        path: std::path::PathBuf,
        title: Option<String>,
    ) -> std::result::Result<
        (
            crate::recording::JobView,
            tokio::sync::watch::Receiver<bool>,
        ),
        String,
    > {
        self.ensure_stt_model_synced().await?;
        self.recordings
            .transcribe_file(&self.config_snapshot()?, path, title)
    }

    pub(crate) async fn calibrate(
        &self,
        speech_secs: u32,
        silence_secs: u32,
        dry_run: bool,
    ) -> std::result::Result<crate::calibrate::CalibrationResult, String> {
        let result = crate::calibrate::run_calibration(
            &self.config_snapshot()?,
            speech_secs,
            silence_secs,
            40,
            30,
            dry_run,
        )
        .await
        .map_err(|e| format!("Calibration failed: {e}"))?;
        if !dry_run {
            self.reload_config().await?;
        }
        Ok(result)
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

    /// Application composition shared by synchronous and inbox delivery.
    pub(crate) async fn converse(
        &self,
        message: &str,
        voice: Option<&str>,
        speed: f32,
        listen: Option<(u32, Option<u32>)>,
    ) -> std::result::Result<String, String> {
        self.speak(message, voice, Some(speed)).await?;
        let Some((silence_timeout_ms, min_speech_ms)) = listen else {
            return Ok("Message spoken successfully".into());
        };
        // Let output buffers drain before opening capture to reduce TTS echo.
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        self.record_and_transcribe(silence_timeout_ms, min_speech_ms)
            .await
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
        self.ensure_stt_model_synced().await?;
        let config = self.config_snapshot()?;

        let (mut rx, capture_handle) = audio::start_capture(
            config.dsp.hpf_cutoff_hz,
            config.dsp.noise_gate_rms,
            config.dsp.noise_gate_window,
        )
        .map_err(|e| format!("Capture failed: {e}"))?;

        let mut all_speech_samples: Vec<f32> = Vec::new();
        let mut vad = VadSession::new(&config, VadOptions::default())
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
                    vad.accept_waveform(&chunk.samples);

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
        let mut task = self.bg_task.lock().await;
        if task.is_some() {
            return Err("Background listener already started; call stop_listening first".into());
        }
        self.ensure_stt_model_synced().await?;
        let config = self.config_snapshot()?;
        let mut vad = VadSession::new(&config, VadOptions::default())
            .map_err(|e| format!("Background VAD init failed: {e}"))?;
        let (mut rx, mut capture) = audio::start_capture(
            config.dsp.hpf_cutoff_hz,
            config.dsp.noise_gate_rms,
            config.dsp.noise_gate_window,
        )
        .map_err(|e| format!("Capture failed: {e}"))?;
        self.bg_active.store(true, Ordering::Release);
        let runtime = self.clone();
        *task = Some(tokio::spawn(async move {
            let result = async {
                let mut speech = BackgroundSpeech::new();
                let silence = std::time::Duration::from_millis(1500);
                while runtime.bg_active.load(Ordering::Acquire) {
                    match tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv())
                        .await
                    {
                        Ok(Some(chunk)) => {
                            vad.accept_waveform(&chunk.samples);
                            let detected = vad.is_speech();
                            speech.accept(detected, vad.collect_segments());
                        }
                        Ok(None) => {
                            return Err("Background audio capture ended unexpectedly".into());
                        }
                        Err(_) => {}
                    }
                    if let Some(samples) = speech.take_ready(silence) {
                        runtime
                            .transcribe_background(samples, config.dsp.normalize_threshold)
                            .await?;
                    }
                }
                capture.finish();
                while let Ok(chunk) = rx.try_recv() {
                    vad.accept_waveform(&chunk.samples);
                    speech.accept(false, vad.collect_segments());
                }
                vad.flush();
                speech.accept(false, vad.collect_segments());
                if !speech.samples.is_empty() {
                    runtime
                        .transcribe_background(speech.samples, config.dsp.normalize_threshold)
                        .await?;
                }
                Ok(())
            }
            .await;
            drop(capture);
            runtime.bg_active.store(false, Ordering::Release);
            result
        }));
        Ok("Background listening started".into())
    }

    async fn transcribe_background(
        &self,
        mut samples: Vec<f32>,
        threshold: f32,
    ) -> std::result::Result<(), String> {
        audio::peak_normalize(&mut samples, threshold);
        let stt = self.stt.clone();
        let text = tokio::task::spawn_blocking(move || {
            stt.lock()
                .map_err(|e| format!("STT lock: {e}"))?
                .transcribe(16000, &samples)
                .map_err(|e| format!("STT failed: {e}"))
        })
        .await
        .map_err(|e| format!("Background STT task: {e}"))??;
        if !text.is_empty() && text != "(no speech detected)" {
            self.push_inbox(InboxMessage {
                text,
                timestamp: chrono_now(),
                source: Some("background_listening".into()),
                conversation_id: None,
            });
        }
        Ok(())
    }

    pub(crate) async fn stop_background_listening(
        &self,
    ) -> std::result::Result<Vec<InboxMessage>, String> {
        let mut task = self.bg_task.lock().await;
        let worker = task.take().ok_or("Background listener is not active")?;
        self.bg_active.store(false, Ordering::Release);
        worker
            .await
            .map_err(|e| format!("Background listener: {e}"))??;
        Ok(self.drain_inbox())
    }

    pub(crate) async fn shutdown(&self) {
        let has_background = self.bg_task.lock().await.is_some();
        if has_background && let Err(e) = self.stop_background_listening().await {
            tracing::warn!("Background listener shutdown: {e}");
        }
        self.recordings
            .shutdown(std::time::Duration::from_secs(20))
            .await;
    }

    pub(crate) async fn reload_config(&self) -> std::result::Result<String, String> {
        Config::reload_into(&self.config).map_err(|e| e.to_string())?;
        self.ensure_tts_model_synced().await?;
        self.ensure_stt_model_synced().await?;
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

    async fn ensure_stt_model_synced(&self) -> std::result::Result<(), String> {
        let _guard = self.stt_swap_lock.lock().await;
        let config = self.config_snapshot()?;
        let stt = self.stt.clone();
        let matches = tokio::task::spawn_blocking(move || {
            stt.lock()
                .map_err(|e| format!("STT lock: {e}"))
                .map(|s| s.matches_config(&config))
        })
        .await
        .map_err(|e| format!("STT check task: {e}"))??;
        if !matches {
            let config = self.config_snapshot()?;
            let replacement = tokio::task::spawn_blocking(move || SttEngine::new(&config))
                .await
                .map_err(|e| format!("STT load task: {e}"))?
                .map_err(|e| format!("STT reload: {e}"))?;
            let stt = self.stt.clone();
            tokio::task::spawn_blocking(move || {
                *stt.lock().map_err(|e| format!("STT lock: {e}"))? = replacement;
                Ok::<_, String>(())
            })
            .await
            .map_err(|e| format!("STT swap task: {e}"))??;
        }
        Ok(())
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
    fn background_collects_completed_segments_after_detection_stops() {
        let mut speech = BackgroundSpeech::new();
        speech.accept(true, Vec::new());
        speech.accept(
            false,
            vec![VadSegment {
                start_sample: 0,
                samples: vec![0.5; 512],
            }],
        );
        assert_eq!(
            speech.take_ready(std::time::Duration::ZERO).unwrap().len(),
            512
        );
        assert!(speech.take_ready(std::time::Duration::ZERO).is_none());
    }

    #[test]
    #[ignore = "requires installed Kokoro and Silero models; no microphone or playback"]
    fn background_model_short_utterance_drains_after_silence() {
        let config = Config::default();
        let mut tts = TtsEngine::new(&config).unwrap();
        let output = tts
            .synthesize(
                "The quick brown fox jumps over the lazy dog.",
                None,
                Some(1.0),
            )
            .unwrap();
        let mut pcm = vec![0.0; 8000];
        pcm.extend(audio::resample(&output.samples, output.sample_rate, 16000));
        pcm.extend(vec![0.0; 48000]);
        let mut vad = VadSession::new(&config, VadOptions::default()).unwrap();
        let mut speech = BackgroundSpeech::new();
        let mut delivered = 0;
        for chunk in pcm.chunks(crate::vad::VAD_WINDOW_SIZE) {
            vad.accept_waveform(chunk);
            let detected = vad.is_speech();
            speech.accept(detected, vad.collect_segments());
            if let Some(samples) = speech.take_ready(std::time::Duration::ZERO) {
                delivered += samples.len();
            }
        }
        assert!(
            delivered > 0,
            "short utterance must be delivered without a forced split or final flush"
        );
        assert!(vad.collect_segments().is_empty());
    }

    #[test]
    fn background_continuous_speech_is_bounded() {
        let mut speech = BackgroundSpeech::new();
        speech.accept(
            true,
            vec![VadSegment {
                start_sample: 0,
                samples: vec![0.5; 20 * 16_000],
            }],
        );
        assert!(
            speech
                .take_ready(std::time::Duration::from_secs(1500))
                .is_some()
        );
    }

    #[test]
    fn chrono_now_produces_valid_format() {
        let ts = chrono_now();
        assert!(ts.ends_with('Z'));
        assert!(ts.contains('T'));
        assert_eq!(ts.len(), 20);
    }

    #[test]
    fn format_utc_known_dates() {
        assert_eq!(format_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_utc(951_782_400), "2000-02-29T00:00:00Z"); // leap day
        assert_eq!(format_utc(1_709_164_800), "2024-02-29T00:00:00Z");
        assert_eq!(format_utc(1_793_102_400), "2026-10-27T12:00:00Z");
        assert_eq!(format_utc(4_102_444_799), "2099-12-31T23:59:59Z");
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
