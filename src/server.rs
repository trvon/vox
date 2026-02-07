use crate::audio::{self, CaptureHandle};
use crate::config::Config;
use crate::stt::SttEngine;
use crate::tts::TtsEngine;
use crate::vad::VadSession;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::task_manager::OperationProcessor;
use rmcp::{ErrorData as McpError, ServerHandler, task_handler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Mutex as TokioMutex;

/// MCP tool parameters for the `converse` tool
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ConverseParams {
    #[schemars(description = "Text to speak aloud before listening")]
    pub message: String,

    #[schemars(description = "Listen for user speech after speaking (default: true)")]
    #[serde(default = "default_true")]
    pub wait_for_response: bool,

    #[schemars(description = "TTS voice name (e.g. af_heart, am_michael)")]
    pub voice: Option<String>,

    #[schemars(description = "Speech rate multiplier (default: 1.0)")]
    #[serde(default = "default_speed")]
    pub speed: f32,

    #[schemars(description = "Maximum listen time in seconds (default: 30)")]
    #[serde(default = "default_timeout")]
    pub timeout_secs: u32,

    #[schemars(description = "Silence duration in ms before end-of-turn (default: 1000)")]
    #[serde(default = "default_silence_timeout")]
    pub silence_timeout_ms: u32,

    #[schemars(description = "Minimum speech duration in ms before accepting silence as end")]
    pub min_speech_ms: Option<u32>,
}

/// MCP tool parameters for the `say` tool
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct SayParams {
    #[schemars(description = "Text to speak aloud")]
    pub message: String,

    #[schemars(description = "TTS voice name (e.g. af_heart, am_michael)")]
    pub voice: Option<String>,

    #[schemars(description = "Speech rate multiplier (default: 1.0)")]
    #[serde(default = "default_speed")]
    pub speed: f32,
}

/// MCP tool parameters for the `listen` tool
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ListenParams {
    #[schemars(description = "Maximum listen time in seconds (default: 30)")]
    #[serde(default = "default_timeout")]
    pub timeout_secs: u32,

    #[schemars(description = "Silence duration in ms before end-of-turn (default: 1000)")]
    #[serde(default = "default_silence_timeout")]
    pub silence_timeout_ms: u32,

    #[schemars(description = "Minimum speech duration in ms before accepting silence as end")]
    pub min_speech_ms: Option<u32>,
}

/// MCP tool parameters for `start_listening` (no required params)
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct StartListeningParams {}

/// MCP tool parameters for `check_inbox` (no params)
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CheckInboxParams {}

/// MCP tool parameters for `stop_listening` (no params)
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct StopListeningParams {}

/// MCP tool parameters for `reset_dsp` (no params)
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ResetDspParams {}

/// MCP tool parameters for `calibrate`
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CalibrateParams {
    #[schemars(description = "If true, print results without saving to config (default: true)")]
    #[serde(default = "default_true")]
    pub dry_run: bool,

    #[schemars(description = "Seconds of speech to record (default: 10)")]
    pub speech_secs: Option<u32>,

    #[schemars(description = "Seconds of silence to record (default: 5)")]
    pub silence_secs: Option<u32>,
}

/// A transcribed message from the background listener
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboxMessage {
    pub text: String,
    pub timestamp: String,
}

/// Simple ISO 8601 timestamp without a chrono dependency
fn chrono_now() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    // Approximate: good enough for inbox timestamps
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;
    // Days since epoch → rough date (accurate for ordering, not calendar display)
    let years = 1970 + days / 365;
    let remaining_days = days % 365;
    let months = remaining_days / 30 + 1;
    let day = remaining_days % 30 + 1;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        years, months, day, hours, minutes, seconds
    )
}

fn default_true() -> bool {
    true
}

fn default_speed() -> f32 {
    1.0
}

fn default_timeout() -> u32 {
    30
}

fn default_silence_timeout() -> u32 {
    1500
}

#[derive(Clone)]
pub struct VoiceMcpServer {
    tts: Arc<Mutex<TtsEngine>>,
    stt: Arc<Mutex<SttEngine>>,
    config: Arc<Config>,
    processor: Arc<TokioMutex<OperationProcessor>>,
    tool_router: ToolRouter<Self>,
    inbox: Arc<Mutex<Vec<InboxMessage>>>,
    bg_capture: Arc<Mutex<Option<CaptureHandle>>>,
    bg_active: Arc<AtomicBool>,
}

impl VoiceMcpServer {
    /// Create a new server with owned engines (wraps in Arc<Mutex<_>>).
    pub fn new(tts: TtsEngine, stt: SttEngine, config: Config) -> Self {
        Self::with_shared(
            Arc::new(Mutex::new(tts)),
            Arc::new(Mutex::new(stt)),
            Arc::new(config),
        )
    }

    /// Create a new server with pre-shared engines (for daemon mode).
    pub fn with_shared(
        tts: Arc<Mutex<TtsEngine>>,
        stt: Arc<Mutex<SttEngine>>,
        config: Arc<Config>,
    ) -> Self {
        Self {
            tts,
            stt,
            config,
            processor: Arc::new(TokioMutex::new(OperationProcessor::new())),
            tool_router: Self::tool_router(),
            inbox: Arc::new(Mutex::new(Vec::new())),
            bg_capture: Arc::new(Mutex::new(None)),
            bg_active: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Synthesize and play audio using streaming — audio starts playing as chunks
    /// are generated by the TTS engine, without waiting for full synthesis.
    async fn speak(
        &self,
        text: &str,
        voice: Option<&str>,
        speed: Option<f32>,
    ) -> std::result::Result<(), String> {
        let tts = self.tts.clone();
        let text = text.to_string();
        let voice = voice.map(|v| v.to_string());
        let speed = speed.unwrap_or(1.0);

        let (std_tx, std_rx) = std::sync::mpsc::channel::<Vec<f32>>();

        // Query sample rate before spawning (quick lock)
        let sample_rate = {
            let tts = tts.lock().map_err(|e| format!("TTS lock: {e}"))?;
            tts.sample_rate()
        };

        // Producer: streaming TTS synthesis (blocking, callback pushes chunks)
        let producer = tokio::task::spawn_blocking(move || {
            let mut tts = tts.lock().map_err(|e| format!("TTS lock: {e}"))?;
            let sid = crate::tts::resolve_voice_id(voice.as_deref().unwrap_or("af_heart"));
            tts.synthesize_streaming(&text, sid, speed, std_tx)
                .map_err(|e| format!("TTS failed: {e}"))
        });

        // Consumer: streaming audio playback (plays chunks as they arrive)
        let consumer = audio::play_audio_streaming(std_rx, sample_rate);

        let (prod_result, cons_result) = tokio::join!(producer, consumer);
        prod_result.map_err(|e| format!("Producer: {e}"))??;
        cons_result.map_err(|e| format!("Playback: {e}"))?;

        Ok(())
    }

    /// Record and transcribe speech
    async fn record_and_transcribe(
        &self,
        timeout_secs: u32,
        silence_timeout_ms: u32,
        min_speech_ms: Option<u32>,
    ) -> std::result::Result<String, String> {
        if self.bg_active.load(Ordering::Relaxed) {
            return Err(
                "Background listener is active. Call stop_listening first.".to_string(),
            );
        }
        let config = self.config.clone();
        let max_speech_secs = timeout_secs as f32;

        let (mut rx, capture_handle) = audio::start_capture(
            config.dsp.hpf_cutoff_hz,
            config.dsp.noise_gate_rms,
            config.dsp.noise_gate_window,
        )
        .map_err(|e| format!("Capture failed: {e}"))?;

        let timeout_duration = std::time::Duration::from_secs(timeout_secs as u64);
        let start = std::time::Instant::now();

        let mut all_speech_samples: Vec<f32> = Vec::new();
        let mut vad = VadSession::new(&config, max_speech_secs)
            .map_err(|e| format!("VAD init failed: {e}"))?;

        let mut speech_started = false;
        let silence_threshold = std::time::Duration::from_millis(silence_timeout_ms as u64);
        let mut last_speech_time = std::time::Instant::now();
        let mut first_speech_time: Option<std::time::Instant> = None;

        loop {
            if start.elapsed() >= timeout_duration {
                tracing::debug!("Listen timeout reached");
                break;
            }

            let chunk = tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv()).await;

            match chunk {
                Ok(Some(chunk)) => {
                    vad.accept_waveform(chunk.samples);

                    if vad.is_speech() {
                        speech_started = true;
                        last_speech_time = std::time::Instant::now();
                        if first_speech_time.is_none() {
                            first_speech_time = Some(std::time::Instant::now());
                        }

                        let segments = vad.collect_segments();
                        for seg in segments {
                            all_speech_samples.extend_from_slice(&seg.samples);
                        }
                    } else if speech_started && last_speech_time.elapsed() >= silence_threshold {
                        // Check min_speech_ms: don't stop until user has spoken enough
                        if let Some(min_ms) = min_speech_ms
                            && let Some(first) = first_speech_time
                            && first.elapsed() < std::time::Duration::from_millis(min_ms as u64)
                        {
                            continue;
                        }
                        tracing::debug!("Silence detected after speech, stopping");
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => {
                    if speech_started && last_speech_time.elapsed() >= silence_threshold {
                        // Check min_speech_ms: don't stop until user has spoken enough
                        if let Some(min_ms) = min_speech_ms
                            && let Some(first) = first_speech_time
                            && first.elapsed() < std::time::Duration::from_millis(min_ms as u64)
                        {
                            continue;
                        }
                        tracing::debug!("Silence detected after speech, stopping");
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

        // Peak-normalize quiet audio to improve STT accuracy
        audio::peak_normalize(&mut all_speech_samples, config.dsp.normalize_threshold);

        tracing::debug!(
            num_samples = all_speech_samples.len(),
            duration_secs = all_speech_samples.len() as f32 / 16000.0,
            "Transcribing captured speech"
        );

        let stt = self.stt.clone();
        let text = tokio::task::spawn_blocking(move || {
            let mut stt = stt.lock().map_err(|e| format!("STT lock poisoned: {e}"))?;
            stt.transcribe(16000, &all_speech_samples)
                .map_err(|e| format!("STT failed: {e}"))
        })
        .await
        .map_err(|e| format!("STT task failed: {e}"))??;

        Ok(text)
    }
}

#[tool_router]
impl VoiceMcpServer {
    #[tool(
        name = "converse",
        description = "Speak a message aloud and listen for the user's spoken response. Returns the transcribed speech."
    )]
    async fn converse(
        &self,
        Parameters(params): Parameters<ConverseParams>,
    ) -> Result<CallToolResult, McpError> {
        self.speak(&params.message, params.voice.as_deref(), Some(params.speed))
            .await
            .map_err(|e| McpError::internal_error(e, None))?;

        if !params.wait_for_response {
            return Ok(CallToolResult::success(vec![Content::text(
                "Message spoken successfully",
            )]));
        }

        // Let macOS Core Audio drain output buffers and amplifier settle
        // before opening the mic, preventing TTS echo from contaminating capture
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

        let text = self
            .record_and_transcribe(params.timeout_secs, params.silence_timeout_ms, params.min_speech_ms)
            .await
            .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        name = "say",
        description = "Speak a message aloud through the speakers. Use for announcements or one-way communication. Returns immediately while speech plays in the background."
    )]
    async fn say(
        &self,
        Parameters(params): Parameters<SayParams>,
    ) -> Result<CallToolResult, McpError> {
        let server = self.clone();
        tokio::spawn(async move {
            if let Err(e) = server
                .speak(&params.message, params.voice.as_deref(), Some(params.speed))
                .await
            {
                tracing::error!("Background speak failed: {e}");
            }
        });

        Ok(CallToolResult::success(vec![Content::text("Speaking...")]))
    }

    #[tool(
        name = "listen",
        description = "Record speech from the microphone and transcribe it. Returns the text of what was spoken."
    )]
    async fn listen(
        &self,
        Parameters(params): Parameters<ListenParams>,
    ) -> Result<CallToolResult, McpError> {
        let text = self
            .record_and_transcribe(params.timeout_secs, params.silence_timeout_ms, params.min_speech_ms)
            .await
            .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        name = "start_listening",
        description = "Start background listening. Speech is captured, transcribed via VAD+STT, and queued in an inbox. Use check_inbox to retrieve transcriptions. Mic is held open until stop_listening is called."
    )]
    async fn start_listening(
        &self,
        Parameters(_params): Parameters<StartListeningParams>,
    ) -> Result<CallToolResult, McpError> {
        if self.bg_active.load(Ordering::Relaxed) {
            return Err(McpError::invalid_request(
                "Background listener is already active",
                None,
            ));
        }

        let config = self.config.clone();
        let (mut rx, capture_handle) = audio::start_capture(
            config.dsp.hpf_cutoff_hz,
            config.dsp.noise_gate_rms,
            config.dsp.noise_gate_window,
        )
        .map_err(|e| McpError::internal_error(format!("Capture failed: {e}"), None))?;

        // Store capture handle
        {
            let mut bg = self.bg_capture.lock().unwrap();
            *bg = Some(capture_handle);
        }
        self.bg_active.store(true, Ordering::Relaxed);

        let inbox = self.inbox.clone();
        let stt = self.stt.clone();
        let bg_active = self.bg_active.clone();
        let config_clone = config.clone();

        tokio::spawn(async move {
            let mut vad = match VadSession::new(&config_clone, 300.0) {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!("Background VAD init failed: {e}");
                    bg_active.store(false, Ordering::Relaxed);
                    return;
                }
            };

            let mut speech_samples: Vec<f32> = Vec::new();
            let mut speech_active = false;
            let mut last_speech = std::time::Instant::now();
            let silence_threshold = std::time::Duration::from_millis(1500);

            loop {
                if !bg_active.load(Ordering::Relaxed) {
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
                            // Flush remaining VAD segments
                            vad.flush();
                            let segments = vad.collect_segments();
                            for seg in segments {
                                speech_samples.extend_from_slice(&seg.samples);
                            }

                            // Normalize and transcribe
                            audio::peak_normalize(
                                &mut speech_samples,
                                config_clone.dsp.normalize_threshold,
                            );

                            let samples = std::mem::take(&mut speech_samples);
                            let stt_clone = stt.clone();
                            let inbox_clone = inbox.clone();

                            // Transcribe in a blocking task
                            tokio::task::spawn_blocking(move || {
                                let mut stt = match stt_clone.lock() {
                                    Ok(s) => s,
                                    Err(e) => {
                                        tracing::error!("STT lock failed: {e}");
                                        return;
                                    }
                                };
                                match stt.transcribe(16000, &samples) {
                                    Ok(text) => {
                                        if !text.is_empty()
                                            && text != "(no speech detected)"
                                        {
                                            let msg = InboxMessage {
                                                text,
                                                timestamp: chrono_now(),
                                            };
                                            if let Ok(mut inbox) = inbox_clone.lock() {
                                                inbox.push(msg);
                                            }
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
                    Err(_) => {} // timeout, loop
                }
            }
        });

        Ok(CallToolResult::success(vec![Content::text(
            "Background listening started",
        )]))
    }

    #[tool(
        name = "check_inbox",
        description = "Check for transcribed speech from the background listener. Returns a JSON array of messages (empty if none). Non-blocking."
    )]
    async fn check_inbox(
        &self,
        Parameters(_params): Parameters<CheckInboxParams>,
    ) -> Result<CallToolResult, McpError> {
        let messages: Vec<InboxMessage> = {
            let mut inbox = self.inbox.lock().unwrap();
            inbox.drain(..).collect()
        };

        let json = serde_json::to_string_pretty(&messages)
            .unwrap_or_else(|_| "[]".to_string());

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        name = "stop_listening",
        description = "Stop the background listener and return any remaining inbox messages."
    )]
    async fn stop_listening(
        &self,
        Parameters(_params): Parameters<StopListeningParams>,
    ) -> Result<CallToolResult, McpError> {
        if !self.bg_active.load(Ordering::Relaxed) {
            return Err(McpError::invalid_request(
                "Background listener is not active",
                None,
            ));
        }

        // Stop capture
        {
            let mut bg = self.bg_capture.lock().unwrap();
            if let Some(handle) = bg.take() {
                handle.stop();
            }
        }
        self.bg_active.store(false, Ordering::Relaxed);

        // Brief delay to let background task finish processing
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // Drain inbox
        let messages: Vec<InboxMessage> = {
            let mut inbox = self.inbox.lock().unwrap();
            inbox.drain(..).collect()
        };

        let json = serde_json::to_string_pretty(&messages)
            .unwrap_or_else(|_| "[]".to_string());

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        name = "reset_dsp",
        description = "Reset DSP audio parameters to defaults. Removes custom calibration values from config. Restart daemon to apply."
    )]
    async fn reset_dsp(
        &self,
        Parameters(_params): Parameters<ResetDspParams>,
    ) -> Result<CallToolResult, McpError> {
        Config::reset_dsp().map_err(|e| McpError::internal_error(e, None))?;
        let defaults = crate::config::DspConfig::default();
        let msg = format!(
            "DSP parameters reset to defaults:\n  \
             hpf_cutoff_hz:       {}\n  \
             noise_gate_rms:      {}\n  \
             noise_gate_window:   {}\n  \
             normalize_threshold: {}\n\n\
             Restart daemon to apply.",
            defaults.hpf_cutoff_hz,
            defaults.noise_gate_rms,
            defaults.noise_gate_window,
            defaults.normalize_threshold,
        );
        Ok(CallToolResult::success(vec![Content::text(msg)]))
    }

    #[tool(
        name = "calibrate",
        description = "Run DSP calibration using a genetic algorithm on live audio. Requires the user to read a passage aloud and record silence. Returns optimal DSP parameters. Set dry_run=false to persist results to config."
    )]
    async fn calibrate(
        &self,
        Parameters(params): Parameters<CalibrateParams>,
    ) -> Result<CallToolResult, McpError> {
        let config = (*self.config).clone();
        let speech_secs = params.speech_secs.unwrap_or(10);
        let silence_secs = params.silence_secs.unwrap_or(5);
        let dry_run = params.dry_run;

        let result = crate::calibrate::run_calibration(
            &config,
            speech_secs,
            silence_secs,
            40,  // population
            30,  // generations
            dry_run,
        )
        .await
        .map_err(|e| McpError::internal_error(format!("Calibration failed: {e}"), None))?;

        let status = if dry_run {
            "Results NOT saved (dry_run=true). Set dry_run=false to persist."
        } else {
            "Results saved to config. Restart daemon to apply."
        };

        let msg = format!(
            "Calibration complete!\n\n\
             Optimal DSP parameters:\n  \
             hpf_cutoff_hz:       {:.1}\n  \
             noise_gate_rms:      {:.3}\n  \
             noise_gate_window:   {}\n  \
             normalize_threshold: {:.2}\n\n\
             Metrics:\n  \
             SNR improvement:     {:+.2} dB\n  \
             Speech retention:    {:.2}\n  \
             Noise floor:         {:.1} dB FS\n\n\
             {}",
            result.optimal.hpf_cutoff_hz,
            result.optimal.noise_gate_rms,
            result.optimal.noise_gate_window,
            result.optimal.normalize_threshold,
            result.snr_improvement_db,
            result.speech_retention,
            result.noise_floor_db,
            status,
        );
        Ok(CallToolResult::success(vec![Content::text(msg)]))
    }
}

#[task_handler]
#[tool_handler]
#[allow(deprecated)] // task_handler macro uses deprecated type aliases in rmcp 0.14
impl ServerHandler for VoiceMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: rmcp::model::ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_tasks()
                .build(),
            server_info: rmcp::model::Implementation::from_build_env(),
            instructions: Some(
                "Vox: voice MCP server with text-to-speech and speech-to-text. \
                 Use 'say' to speak text aloud, 'listen' to capture and transcribe speech, \
                 or 'converse' for a speak-then-listen interaction."
                    .to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_true_returns_true() {
        assert!(default_true());
    }

    #[test]
    fn default_speed_returns_one() {
        assert!((default_speed() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn default_timeout_returns_30() {
        assert_eq!(default_timeout(), 30);
    }

    #[test]
    fn default_silence_timeout_returns_1500() {
        assert_eq!(default_silence_timeout(), 1500);
    }

    #[test]
    fn converse_params_defaults() {
        let json = serde_json::json!({
            "message": "Hello"
        });
        let params: ConverseParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.message, "Hello");
        assert!(params.wait_for_response);
        assert!((params.speed - 1.0).abs() < f32::EPSILON);
        assert_eq!(params.timeout_secs, 30);
        assert!(params.voice.is_none());
        assert_eq!(params.silence_timeout_ms, 1500);
        assert!(params.min_speech_ms.is_none());
    }

    #[test]
    fn converse_params_with_overrides() {
        let json = serde_json::json!({
            "message": "Test",
            "wait_for_response": false,
            "voice": "am_michael",
            "speed": 1.5,
            "timeout_secs": 60,
            "silence_timeout_ms": 500,
            "min_speech_ms": 2000
        });
        let params: ConverseParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.message, "Test");
        assert!(!params.wait_for_response);
        assert_eq!(params.voice.as_deref(), Some("am_michael"));
        assert!((params.speed - 1.5).abs() < f32::EPSILON);
        assert_eq!(params.timeout_secs, 60);
        assert_eq!(params.silence_timeout_ms, 500);
        assert_eq!(params.min_speech_ms, Some(2000));
    }

    #[test]
    fn say_params_minimal() {
        let json = serde_json::json!({
            "message": "Hello world"
        });
        let params: SayParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.message, "Hello world");
        assert!(params.voice.is_none());
        assert!((params.speed - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn say_params_with_voice_and_speed() {
        let json = serde_json::json!({
            "message": "Test",
            "voice": "af_bella",
            "speed": 2.0
        });
        let params: SayParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.voice.as_deref(), Some("af_bella"));
        assert!((params.speed - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn listen_params_default_timeout() {
        let json = serde_json::json!({});
        let params: ListenParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.timeout_secs, 30);
        assert_eq!(params.silence_timeout_ms, 1500);
        assert!(params.min_speech_ms.is_none());
    }

    #[test]
    fn listen_params_custom_timeout() {
        let json = serde_json::json!({
            "timeout_secs": 10,
            "silence_timeout_ms": 750,
            "min_speech_ms": 1500
        });
        let params: ListenParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.timeout_secs, 10);
        assert_eq!(params.silence_timeout_ms, 750);
        assert_eq!(params.min_speech_ms, Some(1500));
    }

    #[test]
    fn inbox_message_serialization() {
        let msg = InboxMessage {
            text: "Hello world".to_string(),
            timestamp: "2024-01-15T10:30:00Z".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("Hello world"));
        assert!(json.contains("2024-01-15T10:30:00Z"));

        let parsed: InboxMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.text, "Hello world");
        assert_eq!(parsed.timestamp, "2024-01-15T10:30:00Z");
    }

    #[test]
    fn inbox_message_array_serialization() {
        let msgs = vec![
            InboxMessage {
                text: "First".to_string(),
                timestamp: "2024-01-15T10:30:00Z".to_string(),
            },
            InboxMessage {
                text: "Second".to_string(),
                timestamp: "2024-01-15T10:30:05Z".to_string(),
            },
        ];
        let json = serde_json::to_string_pretty(&msgs).unwrap();
        assert!(json.contains("First"));
        assert!(json.contains("Second"));
    }

    #[test]
    fn empty_inbox_serialization() {
        let msgs: Vec<InboxMessage> = vec![];
        let json = serde_json::to_string_pretty(&msgs).unwrap();
        assert_eq!(json, "[]");
    }

    #[test]
    fn start_listening_params_deserialize() {
        let json = serde_json::json!({});
        let _params: StartListeningParams = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn check_inbox_params_deserialize() {
        let json = serde_json::json!({});
        let _params: CheckInboxParams = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn stop_listening_params_deserialize() {
        let json = serde_json::json!({});
        let _params: StopListeningParams = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn reset_dsp_params_deserialize() {
        let json = serde_json::json!({});
        let _params: ResetDspParams = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn calibrate_params_defaults() {
        let json = serde_json::json!({});
        let params: CalibrateParams = serde_json::from_value(json).unwrap();
        assert!(params.dry_run); // default true = safe
        assert!(params.speech_secs.is_none());
        assert!(params.silence_secs.is_none());
    }

    #[test]
    fn calibrate_params_with_overrides() {
        let json = serde_json::json!({
            "dry_run": false,
            "speech_secs": 15,
            "silence_secs": 8
        });
        let params: CalibrateParams = serde_json::from_value(json).unwrap();
        assert!(!params.dry_run);
        assert_eq!(params.speech_secs, Some(15));
        assert_eq!(params.silence_secs, Some(8));
    }

    #[test]
    fn chrono_now_produces_valid_format() {
        let ts = chrono_now();
        assert!(ts.ends_with('Z'));
        assert!(ts.contains('T'));
        assert_eq!(ts.len(), 20); // YYYY-MM-DDTHH:MM:SSZ
    }
}
