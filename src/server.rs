use crate::config::SharedConfig;
use crate::runtime::{InboxMessage, VoiceRuntime, chrono_now};
use crate::stt::SttEngine;
use crate::tts::TtsEngine;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::task_manager::OperationProcessor;
use rmcp::{ErrorData as McpError, ServerHandler, task_handler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
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

    #[schemars(
        description = "Silence duration in ms before end-of-turn (default: 2500 for converse)"
    )]
    #[serde(default = "default_converse_silence_timeout")]
    pub silence_timeout_ms: u32,

    #[schemars(description = "Minimum speech duration in ms before accepting silence as end")]
    pub min_speech_ms: Option<u32>,

    #[schemars(
        description = "If true, return immediately and deliver the transcription via check_inbox"
    )]
    #[serde(default)]
    pub async_mode: Option<bool>,
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

/// MCP tool parameters for the `enqueue_say` tool
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct EnqueueSayParams {
    #[schemars(description = "Text to speak aloud")]
    pub message: String,

    #[schemars(description = "TTS voice name (e.g. af_heart, am_michael)")]
    pub voice: Option<String>,

    #[schemars(description = "Speech rate multiplier (default: 1.0)")]
    #[serde(default = "default_speed")]
    pub speed: f32,
}

/// MCP tool parameters for `tts_queue_status` (no params)
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct TtsQueueStatusParams {}

/// MCP tool parameters for `tts_queue_clear` (no params)
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct TtsQueueClearParams {}

/// MCP tool parameters for the `listen` tool
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ListenParams {
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

/// MCP tool parameters for `reload_config` (no params)
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ReloadConfigParams {}

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

static CONVERSE_COUNTER: AtomicU64 = AtomicU64::new(1);
const CONVERSE_AUTO_ASYNC_THRESHOLD_CHARS: usize = 600;

fn default_true() -> bool {
    true
}

fn default_speed() -> f32 {
    1.0
}

fn default_silence_timeout() -> u32 {
    1500
}

fn default_converse_silence_timeout() -> u32 {
    2500
}

#[derive(Clone)]
pub struct VoiceMcpServer {
    runtime: VoiceRuntime,
    processor: Arc<TokioMutex<OperationProcessor>>,
    tool_router: ToolRouter<Self>,
}

impl VoiceMcpServer {
    /// Create a new server with owned engines (wraps in Arc<Mutex<_>>).
    pub fn new(tts: TtsEngine, stt: SttEngine, config: SharedConfig) -> Self {
        Self::with_shared(Arc::new(Mutex::new(tts)), Arc::new(Mutex::new(stt)), config)
    }

    /// Create a new server with pre-shared engines (for daemon mode).
    pub fn with_shared(
        tts: Arc<std::sync::Mutex<TtsEngine>>,
        stt: Arc<std::sync::Mutex<SttEngine>>,
        config: SharedConfig,
    ) -> Self {
        Self {
            runtime: VoiceRuntime::with_shared(tts, stt, config),
            processor: Arc::new(TokioMutex::new(OperationProcessor::new())),
            tool_router: Self::tool_router(),
        }
    }

    fn should_converse_async(params: &ConverseParams) -> bool {
        match params.async_mode {
            Some(v) => v,
            None => params.message.len() >= CONVERSE_AUTO_ASYNC_THRESHOLD_CHARS,
        }
    }

    fn new_conversation_id() -> String {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let n = CONVERSE_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("converse-{now_ms}-{n}")
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
        if Self::should_converse_async(&params) {
            let conversation_id = Self::new_conversation_id();
            let conversation_id_for_ack = conversation_id.clone();
            let runtime = self.runtime.clone();
            let params = params;

            tokio::spawn(async move {
                let speak_result = runtime
                    .speak(&params.message, params.voice.as_deref(), Some(params.speed))
                    .await;

                if let Err(e) = speak_result {
                    runtime.push_inbox(InboxMessage {
                        text: format!("converse failed during speak: {e}"),
                        timestamp: chrono_now(),
                        source: Some("converse".to_string()),
                        conversation_id: Some(conversation_id.clone()),
                    });
                    return;
                }

                if !params.wait_for_response {
                    runtime.push_inbox(InboxMessage {
                        text: "Message spoken successfully".to_string(),
                        timestamp: chrono_now(),
                        source: Some("converse".to_string()),
                        conversation_id: Some(conversation_id.clone()),
                    });
                    return;
                }

                tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                match runtime
                    .record_and_transcribe(params.silence_timeout_ms, params.min_speech_ms)
                    .await
                {
                    Ok(text) => {
                        runtime.push_inbox(InboxMessage {
                            text,
                            timestamp: chrono_now(),
                            source: Some("converse".to_string()),
                            conversation_id: Some(conversation_id.clone()),
                        });
                    }
                    Err(e) => {
                        runtime.push_inbox(InboxMessage {
                            text: format!("converse failed during listen/transcribe: {e}"),
                            timestamp: chrono_now(),
                            source: Some("converse".to_string()),
                            conversation_id: Some(conversation_id.clone()),
                        });
                    }
                }
            });

            let ack = serde_json::json!({
                "status": "accepted",
                "conversation_id": conversation_id_for_ack,
                "delivery": "check_inbox"
            });
            return Ok(CallToolResult::success(vec![Content::text(
                ack.to_string(),
            )]));
        }

        self.runtime
            .speak(&params.message, params.voice.as_deref(), Some(params.speed))
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
            .runtime
            .record_and_transcribe(params.silence_timeout_ms, params.min_speech_ms)
            .await
            .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        name = "say",
        description = "Queue a message to speak aloud through the speakers. Returns immediately after enqueuing."
    )]
    async fn say(
        &self,
        Parameters(params): Parameters<SayParams>,
    ) -> Result<CallToolResult, McpError> {
        let (speech_id, pending_count) = self
            .runtime
            .enqueue_speech(params.message, params.voice, params.speed)
            .await;
        let ack = serde_json::json!({
            "status": "queued",
            "speech_id": speech_id,
            "pending_count": pending_count
        });

        Ok(CallToolResult::success(vec![Content::text(
            ack.to_string(),
        )]))
    }

    #[tool(
        name = "enqueue_say",
        description = "Queue speech for ordered asynchronous playback. Returns a speech_id immediately."
    )]
    async fn enqueue_say(
        &self,
        Parameters(params): Parameters<EnqueueSayParams>,
    ) -> Result<CallToolResult, McpError> {
        let (speech_id, pending_count) = self
            .runtime
            .enqueue_speech(params.message, params.voice, params.speed)
            .await;
        let ack = serde_json::json!({
            "status": "queued",
            "speech_id": speech_id,
            "pending_count": pending_count
        });

        Ok(CallToolResult::success(vec![Content::text(
            ack.to_string(),
        )]))
    }

    #[tool(
        name = "tts_queue_status",
        description = "Get text-to-speech queue status, including current and pending items."
    )]
    async fn tts_queue_status(
        &self,
        Parameters(_params): Parameters<TtsQueueStatusParams>,
    ) -> Result<CallToolResult, McpError> {
        let status = self.runtime.tts_queue_status_snapshot().await;
        let json = serde_json::to_string_pretty(&status)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        name = "tts_queue_clear",
        description = "Clear pending text-to-speech queue items. Does not interrupt currently playing audio."
    )]
    async fn tts_queue_clear(
        &self,
        Parameters(_params): Parameters<TtsQueueClearParams>,
    ) -> Result<CallToolResult, McpError> {
        let result = self.runtime.clear_tts_queue().await;
        let json = serde_json::to_string_pretty(&result)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
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
            .runtime
            .record_and_transcribe(params.silence_timeout_ms, params.min_speech_ms)
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
        let msg = self
            .runtime
            .start_background_listening()
            .await
            .map_err(|e| McpError::invalid_request(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(msg)]))
    }

    #[tool(
        name = "check_inbox",
        description = "Check for transcribed speech from the background listener. Returns a JSON array of messages (empty if none). Non-blocking."
    )]
    async fn check_inbox(
        &self,
        Parameters(_params): Parameters<CheckInboxParams>,
    ) -> Result<CallToolResult, McpError> {
        let messages = self.runtime.drain_inbox();

        let json = serde_json::to_string_pretty(&messages).unwrap_or_else(|_| "[]".to_string());

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
        let messages = self
            .runtime
            .stop_background_listening()
            .await
            .map_err(|e| McpError::invalid_request(e, None))?;

        let json = serde_json::to_string_pretty(&messages).unwrap_or_else(|_| "[]".to_string());

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        name = "reset_dsp",
        description = "Reset DSP audio parameters to defaults. Removes custom calibration values from config."
    )]
    async fn reset_dsp(
        &self,
        Parameters(_params): Parameters<ResetDspParams>,
    ) -> Result<CallToolResult, McpError> {
        let msg = self
            .runtime
            .reset_dsp()
            .await
            .map_err(|e| McpError::internal_error(e, None))?;
        Ok(CallToolResult::success(vec![Content::text(msg)]))
    }

    #[tool(
        name = "reload_config",
        description = "Reload config from disk. Use after manually editing the config TOML file. Returns the current config values."
    )]
    async fn reload_config(
        &self,
        Parameters(_params): Parameters<ReloadConfigParams>,
    ) -> Result<CallToolResult, McpError> {
        let msg = self
            .runtime
            .reload_config()
            .await
            .map_err(|e| McpError::internal_error(e, None))?;
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
        let config = self
            .runtime
            .config_snapshot()
            .map_err(|e| McpError::internal_error(e, None))?;
        let speech_secs = params.speech_secs.unwrap_or(10);
        let silence_secs = params.silence_secs.unwrap_or(5);
        let dry_run = params.dry_run;

        let result = crate::calibrate::run_calibration(
            &config,
            speech_secs,
            silence_secs,
            40, // population
            30, // generations
            dry_run,
        )
        .await
        .map_err(|e| McpError::internal_error(format!("Calibration failed: {e}"), None))?;

        // Auto-reload config so new DSP params take effect immediately
        if !dry_run {
            self.runtime
                .reload_config()
                .await
                .map_err(|e| McpError::internal_error(e, None))?;
        }

        let status = if dry_run {
            "Results NOT saved (dry_run=true). Set dry_run=false to persist."
        } else {
            "Results saved and applied to running server."
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
                 Use 'say' or 'enqueue_say' to queue speech, 'tts_queue_status' to inspect queue state, \
                 'tts_queue_clear' to clear pending speech, 'listen' to capture and transcribe speech, \
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
        assert!(params.voice.is_none());
        assert_eq!(params.silence_timeout_ms, 2500);
        assert!(params.min_speech_ms.is_none());
    }

    #[test]
    fn converse_params_with_overrides() {
        let json = serde_json::json!({
            "message": "Test",
            "wait_for_response": false,
            "voice": "am_michael",
            "speed": 1.5,
            "silence_timeout_ms": 500,
            "min_speech_ms": 2000
        });
        let params: ConverseParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.message, "Test");
        assert!(!params.wait_for_response);
        assert_eq!(params.voice.as_deref(), Some("am_michael"));
        assert!((params.speed - 1.5).abs() < f32::EPSILON);
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
    fn enqueue_say_params_minimal() {
        let json = serde_json::json!({
            "message": "Queue this"
        });
        let params: EnqueueSayParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.message, "Queue this");
        assert!(params.voice.is_none());
        assert!((params.speed - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn enqueue_say_params_with_voice_and_speed() {
        let json = serde_json::json!({
            "message": "Queue",
            "voice": "am_michael",
            "speed": 1.25
        });
        let params: EnqueueSayParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.voice.as_deref(), Some("am_michael"));
        assert!((params.speed - 1.25).abs() < f32::EPSILON);
    }

    #[test]
    fn listen_params_defaults() {
        let json = serde_json::json!({});
        let params: ListenParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.silence_timeout_ms, 1500);
        assert!(params.min_speech_ms.is_none());
    }

    #[test]
    fn listen_params_with_overrides() {
        let json = serde_json::json!({
            "silence_timeout_ms": 750,
            "min_speech_ms": 1500
        });
        let params: ListenParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.silence_timeout_ms, 750);
        assert_eq!(params.min_speech_ms, Some(1500));
    }

    #[test]
    fn inbox_message_serialization() {
        let msg = InboxMessage {
            text: "Hello world".to_string(),
            timestamp: "2024-01-15T10:30:00Z".to_string(),
            source: None,
            conversation_id: None,
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
                source: None,
                conversation_id: None,
            },
            InboxMessage {
                text: "Second".to_string(),
                timestamp: "2024-01-15T10:30:05Z".to_string(),
                source: None,
                conversation_id: None,
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
    fn reload_config_params_deserialize() {
        let json = serde_json::json!({});
        let _params: ReloadConfigParams = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn tts_queue_status_params_deserialize() {
        let json = serde_json::json!({});
        let _params: TtsQueueStatusParams = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn tts_queue_clear_params_deserialize() {
        let json = serde_json::json!({});
        let _params: TtsQueueClearParams = serde_json::from_value(json).unwrap();
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
}
