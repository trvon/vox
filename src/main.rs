mod audio;
mod config;
mod error;
mod models;
mod server;
mod stt;
mod tts;
mod vad;

use config::Config;
use rmcp::ServiceExt;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let config = Config::load();

    // Initialize logging to stderr (stdout is reserved for MCP transport)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| config.log_level.parse().unwrap_or_default()),
        )
        .with_writer(std::io::stderr)
        .init();

    // Check for --download-models flag
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--download-models") {
        eprintln!("Downloading models...");
        models::download_models(&config).await?;
        eprintln!("All models downloaded successfully.");
        return Ok(());
    }

    // Check if models exist, download if needed
    if !models::models_ready(&config) {
        eprintln!("Models not found. Downloading (this only happens once)...");
        models::download_models(&config).await?;
        eprintln!("Models downloaded successfully.");
    }

    // Initialize engines
    eprintln!("Initializing voice engines...");

    let tts_engine = {
        let config = config.clone();
        tokio::task::spawn_blocking(move || tts::TtsEngine::new(&config))
            .await??
    };

    let stt_engine = {
        let config = config.clone();
        tokio::task::spawn_blocking(move || stt::SttEngine::new(&config))
            .await??
    };

    eprintln!("Vox MCP server ready");

    // Create and serve MCP server
    let server = server::VoiceMcpServer::new(tts_engine, stt_engine, config);
    let service = server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;

    Ok(())
}
