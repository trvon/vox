use crate::audio;
use crate::config::Config;
use crate::stt::SttEngine;
use crate::tts::TtsEngine;
use crate::vad::VadSession;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

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

#[derive(Clone)]
pub struct VoiceMcpServer {
    tts: Arc<Mutex<TtsEngine>>,
    stt: Arc<Mutex<SttEngine>>,
    config: Arc<Config>,
    tool_router: ToolRouter<Self>,
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
            tool_router: Self::tool_router(),
        }
    }

    /// Synthesize and play audio, streaming sentence-by-sentence for multi-sentence text.
    async fn speak(
        &self,
        text: &str,
        voice: Option<&str>,
        speed: Option<f32>,
    ) -> std::result::Result<(), String> {
        let sentences = crate::tts::split_sentences(text);

        // Single sentence: simple path, no channel overhead
        if sentences.len() <= 2 {
            return self.speak_single(text, voice, speed).await;
        }

        // Multiple sentences: pipeline synthesis → playback
        let (tx, mut rx) =
            tokio::sync::mpsc::channel::<(Vec<f32>, u32)>(2);

        let tts = self.tts.clone();
        let voice = voice.map(|v| v.to_string());
        let sentences: Vec<String> = sentences.into_iter().map(|s| s.to_string()).collect();

        // Producer: synthesize sentences sequentially, send audio chunks
        let producer = tokio::task::spawn(async move {
            for sentence in sentences {
                let tts = tts.clone();
                let voice = voice.clone();
                let result = tokio::task::spawn_blocking(move || {
                    let mut tts = tts.lock().map_err(|e| format!("TTS lock poisoned: {e}"))?;
                    tts.synthesize(&sentence, voice.as_deref(), speed)
                        .map_err(|e| format!("TTS failed: {e}"))
                })
                .await
                .map_err(|e| format!("TTS task failed: {e}"))??;

                if tx.send((result.samples, result.sample_rate)).await.is_err() {
                    break; // consumer dropped
                }
            }
            Ok::<(), String>(())
        });

        // Consumer: play audio chunks as they arrive
        let consumer = tokio::task::spawn(async move {
            while let Some((samples, sample_rate)) = rx.recv().await {
                audio::play_audio(samples, sample_rate)
                    .await
                    .map_err(|e| format!("Playback failed: {e}"))?;
            }
            Ok::<(), String>(())
        });

        let (prod_result, cons_result) = tokio::join!(producer, consumer);
        prod_result.map_err(|e| format!("Producer task failed: {e}"))??;
        cons_result.map_err(|e| format!("Consumer task failed: {e}"))??;

        Ok(())
    }

    /// Synthesize and play a single chunk of text (no pipelining).
    async fn speak_single(
        &self,
        text: &str,
        voice: Option<&str>,
        speed: Option<f32>,
    ) -> std::result::Result<(), String> {
        let tts = self.tts.clone();
        let text = text.to_string();
        let voice = voice.map(|v| v.to_string());

        let audio = tokio::task::spawn_blocking(move || {
            let mut tts = tts.lock().map_err(|e| format!("TTS lock poisoned: {e}"))?;
            tts.synthesize(&text, voice.as_deref(), speed)
                .map_err(|e| format!("TTS failed: {e}"))
        })
        .await
        .map_err(|e| format!("TTS task failed: {e}"))??;

        audio::play_audio(audio.samples, audio.sample_rate)
            .await
            .map_err(|e| format!("Playback failed: {e}"))?;

        Ok(())
    }

    /// Record and transcribe speech
    async fn record_and_transcribe(
        &self,
        timeout_secs: u32,
    ) -> std::result::Result<String, String> {
        let config = self.config.clone();
        let max_speech_secs = timeout_secs as f32;

        let (mut rx, capture_handle) =
            audio::start_capture().map_err(|e| format!("Capture failed: {e}"))?;

        let timeout_duration = std::time::Duration::from_secs(timeout_secs as u64);
        let start = std::time::Instant::now();

        let mut all_speech_samples: Vec<f32> = Vec::new();
        let mut vad = VadSession::new(&config, max_speech_secs)
            .map_err(|e| format!("VAD init failed: {e}"))?;

        let mut speech_started = false;
        let silence_threshold = std::time::Duration::from_millis(1500);
        let mut last_speech_time = std::time::Instant::now();

        loop {
            if start.elapsed() >= timeout_duration {
                tracing::debug!("Listen timeout reached");
                break;
            }

            let chunk =
                tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await;

            match chunk {
                Ok(Some(chunk)) => {
                    vad.accept_waveform(chunk.samples);

                    if vad.is_speech() {
                        speech_started = true;
                        last_speech_time = std::time::Instant::now();

                        let segments = vad.collect_segments();
                        for seg in segments {
                            all_speech_samples.extend_from_slice(&seg.samples);
                        }
                    } else if speech_started && last_speech_time.elapsed() >= silence_threshold {
                        tracing::debug!("Silence detected after speech, stopping");
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => {
                    if speech_started && last_speech_time.elapsed() >= silence_threshold {
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
        self.speak(
            &params.message,
            params.voice.as_deref(),
            Some(params.speed),
        )
        .await
        .map_err(|e| McpError::internal_error(e, None))?;

        if !params.wait_for_response {
            return Ok(CallToolResult::success(vec![Content::text(
                "Message spoken successfully",
            )]));
        }

        let text = self
            .record_and_transcribe(params.timeout_secs)
            .await
            .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        name = "say",
        description = "Speak a message aloud through the speakers. Use for announcements or one-way communication."
    )]
    async fn say(
        &self,
        Parameters(params): Parameters<SayParams>,
    ) -> Result<CallToolResult, McpError> {
        self.speak(&params.message, params.voice.as_deref(), Some(params.speed))
            .await
            .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(
            "Message spoken successfully",
        )]))
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
            .record_and_transcribe(params.timeout_secs)
            .await
            .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

#[tool_handler]
impl ServerHandler for VoiceMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: rmcp::model::ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
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
    }

    #[test]
    fn converse_params_with_overrides() {
        let json = serde_json::json!({
            "message": "Test",
            "wait_for_response": false,
            "voice": "am_michael",
            "speed": 1.5,
            "timeout_secs": 60
        });
        let params: ConverseParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.message, "Test");
        assert!(!params.wait_for_response);
        assert_eq!(params.voice.as_deref(), Some("am_michael"));
        assert!((params.speed - 1.5).abs() < f32::EPSILON);
        assert_eq!(params.timeout_secs, 60);
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
    }

    #[test]
    fn listen_params_custom_timeout() {
        let json = serde_json::json!({
            "timeout_secs": 10
        });
        let params: ListenParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.timeout_secs, 10);
    }
}
