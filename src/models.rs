use crate::config::Config;
use crate::error::{Result, VoiceError};
use std::path::Path;
use tokio::io::AsyncWriteExt;

const VAD_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";

const MOONSHINE_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-base-en-int8.tar.bz2";

const KOKORO_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2";

/// Check if all required models are present
pub fn models_ready(config: &Config) -> bool {
    let moonshine_dir = config.moonshine_dir();
    let kokoro_dir = config.kokoro_dir();
    let vad_path = config.vad_model_path();

    vad_path.exists()
        && moonshine_dir.join("preprocess.onnx").exists()
        && moonshine_dir.join("encode.int8.onnx").exists()
        && moonshine_dir.join("uncached_decode.int8.onnx").exists()
        && moonshine_dir.join("cached_decode.int8.onnx").exists()
        && moonshine_dir.join("tokens.txt").exists()
        && kokoro_dir.join("model.onnx").exists()
        && kokoro_dir.join("voices.bin").exists()
        && kokoro_dir.join("tokens.txt").exists()
}

/// Download all required models
pub async fn download_models(config: &Config) -> Result<()> {
    std::fs::create_dir_all(&config.model_dir)
        .map_err(|e| VoiceError::Download(format!("Failed to create model dir: {e}")))?;

    // Download VAD model
    let vad_path = config.vad_model_path();
    if !vad_path.exists() {
        download_file(VAD_URL, &vad_path).await?;
    } else {
        eprintln!("  VAD model already exists, skipping");
    }

    // Download Moonshine model
    let moonshine_dir = config.moonshine_dir();
    if !moonshine_dir.join("preprocess.onnx").exists() {
        download_and_extract_tar_bz2(MOONSHINE_URL, &config.model_dir).await?;
    } else {
        eprintln!("  Moonshine model already exists, skipping");
    }

    // Download Kokoro TTS model
    let kokoro_dir = config.kokoro_dir();
    if !kokoro_dir.join("model.onnx").exists() {
        download_and_extract_tar_bz2(KOKORO_URL, &config.model_dir).await?;
    } else {
        eprintln!("  Kokoro model already exists, skipping");
    }

    Ok(())
}

async fn download_file(url: &str, dest: &Path) -> Result<()> {
    eprintln!("  Downloading {} ...", url);

    let response = reqwest::get(url).await?;
    if !response.status().is_success() {
        return Err(VoiceError::Download(format!(
            "HTTP {} for {url}",
            response.status()
        )));
    }

    let total = response.content_length();
    let mut stream = response.bytes_stream();
    let mut file = tokio::fs::File::create(dest).await?;
    let mut downloaded: u64 = 0;

    use tokio_stream::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        if let Some(total) = total {
            eprint!(
                "\r  Progress: {:.1}%",
                (downloaded as f64 / total as f64) * 100.0
            );
        }
    }
    eprintln!();

    file.flush().await?;
    eprintln!("  Saved to {}", dest.display());
    Ok(())
}

async fn download_and_extract_tar_bz2(url: &str, dest_dir: &Path) -> Result<()> {
    eprintln!("  Downloading {} ...", url);

    let response = reqwest::get(url).await?;
    if !response.status().is_success() {
        return Err(VoiceError::Download(format!(
            "HTTP {} for {url}",
            response.status()
        )));
    }

    let total = response.content_length();
    let mut stream = response.bytes_stream();
    let mut data = Vec::new();
    let mut downloaded: u64 = 0;

    use tokio_stream::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        downloaded += chunk.len() as u64;
        data.extend_from_slice(&chunk);
        if let Some(total) = total {
            eprint!(
                "\r  Progress: {:.1}%",
                (downloaded as f64 / total as f64) * 100.0
            );
        }
    }
    eprintln!();

    // Extract tar.bz2 in a blocking task
    let dest = dest_dir.to_path_buf();
    tokio::task::spawn_blocking(move || extract_tar_bz2(&data, &dest))
        .await
        .map_err(|e| VoiceError::Download(format!("Extract task failed: {e}")))?
}

fn extract_tar_bz2(data: &[u8], dest_dir: &Path) -> Result<()> {
    use std::io::Read;

    eprintln!("  Extracting to {} ...", dest_dir.display());

    let bz2_reader = bzip2::read::BzDecoder::new(data);
    let mut archive = tar::Archive::new(bz2_reader);

    // Check if entries can be read
    for entry in archive
        .entries()
        .map_err(|e| VoiceError::Download(format!("Failed to read tar entries: {e}")))?
    {
        let mut entry =
            entry.map_err(|e| VoiceError::Download(format!("Failed to read tar entry: {e}")))?;
        let path = entry
            .path()
            .map_err(|e| VoiceError::Download(format!("Invalid path in tar: {e}")))?
            .to_path_buf();

        let dest_path = dest_dir.join(&path);

        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&dest_path)
                .map_err(|e| VoiceError::Download(format!("Failed to create dir: {e}")))?;
        } else {
            if let Some(parent) = dest_path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| VoiceError::Download(format!("Failed to create dir: {e}")))?;
            }
            let mut file = std::fs::File::create(&dest_path)
                .map_err(|e| VoiceError::Download(format!("Failed to create file: {e}")))?;
            let mut buf = Vec::new();
            entry
                .read_to_end(&mut buf)
                .map_err(|e| VoiceError::Download(format!("Failed to read entry: {e}")))?;
            std::io::Write::write_all(&mut file, &buf)
                .map_err(|e| VoiceError::Download(format!("Failed to write file: {e}")))?;
        }
    }

    eprintln!("  Extraction complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_config(model_dir: &std::path::Path) -> Config {
        Config {
            model_dir: model_dir.to_path_buf(),
            ..Config::default()
        }
    }

    #[test]
    fn models_ready_false_on_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let config = test_config(tmp.path());
        assert!(!models_ready(&config));
    }

    #[test]
    fn models_ready_true_when_all_files_exist() {
        let tmp = tempfile::tempdir().unwrap();
        let config = test_config(tmp.path());

        // Create VAD model
        fs::write(config.vad_model_path(), b"dummy").unwrap();

        // Create moonshine files
        let moonshine_dir = config.moonshine_dir();
        fs::create_dir_all(&moonshine_dir).unwrap();
        fs::write(moonshine_dir.join("preprocess.onnx"), b"dummy").unwrap();
        fs::write(moonshine_dir.join("encode.int8.onnx"), b"dummy").unwrap();
        fs::write(moonshine_dir.join("uncached_decode.int8.onnx"), b"dummy").unwrap();
        fs::write(moonshine_dir.join("cached_decode.int8.onnx"), b"dummy").unwrap();
        fs::write(moonshine_dir.join("tokens.txt"), b"dummy").unwrap();

        // Create kokoro files
        let kokoro_dir = config.kokoro_dir();
        fs::create_dir_all(&kokoro_dir).unwrap();
        fs::write(kokoro_dir.join("model.onnx"), b"dummy").unwrap();
        fs::write(kokoro_dir.join("voices.bin"), b"dummy").unwrap();
        fs::write(kokoro_dir.join("tokens.txt"), b"dummy").unwrap();

        assert!(models_ready(&config));
    }

    #[test]
    fn models_ready_false_when_vad_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let config = test_config(tmp.path());

        // Create everything except VAD
        let moonshine_dir = config.moonshine_dir();
        fs::create_dir_all(&moonshine_dir).unwrap();
        fs::write(moonshine_dir.join("preprocess.onnx"), b"dummy").unwrap();
        fs::write(moonshine_dir.join("encode.int8.onnx"), b"dummy").unwrap();
        fs::write(moonshine_dir.join("uncached_decode.int8.onnx"), b"dummy").unwrap();
        fs::write(moonshine_dir.join("cached_decode.int8.onnx"), b"dummy").unwrap();
        fs::write(moonshine_dir.join("tokens.txt"), b"dummy").unwrap();

        let kokoro_dir = config.kokoro_dir();
        fs::create_dir_all(&kokoro_dir).unwrap();
        fs::write(kokoro_dir.join("model.onnx"), b"dummy").unwrap();
        fs::write(kokoro_dir.join("voices.bin"), b"dummy").unwrap();
        fs::write(kokoro_dir.join("tokens.txt"), b"dummy").unwrap();

        assert!(!models_ready(&config));
    }

    #[test]
    fn models_ready_false_when_moonshine_encoder_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let config = test_config(tmp.path());

        fs::write(config.vad_model_path(), b"dummy").unwrap();

        let moonshine_dir = config.moonshine_dir();
        fs::create_dir_all(&moonshine_dir).unwrap();
        fs::write(moonshine_dir.join("preprocess.onnx"), b"dummy").unwrap();
        // Missing encode.int8.onnx
        fs::write(moonshine_dir.join("uncached_decode.int8.onnx"), b"dummy").unwrap();
        fs::write(moonshine_dir.join("cached_decode.int8.onnx"), b"dummy").unwrap();
        fs::write(moonshine_dir.join("tokens.txt"), b"dummy").unwrap();

        let kokoro_dir = config.kokoro_dir();
        fs::create_dir_all(&kokoro_dir).unwrap();
        fs::write(kokoro_dir.join("model.onnx"), b"dummy").unwrap();
        fs::write(kokoro_dir.join("voices.bin"), b"dummy").unwrap();
        fs::write(kokoro_dir.join("tokens.txt"), b"dummy").unwrap();

        assert!(!models_ready(&config));
    }

    #[test]
    fn models_ready_false_when_kokoro_model_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let config = test_config(tmp.path());

        fs::write(config.vad_model_path(), b"dummy").unwrap();

        let moonshine_dir = config.moonshine_dir();
        fs::create_dir_all(&moonshine_dir).unwrap();
        fs::write(moonshine_dir.join("preprocess.onnx"), b"dummy").unwrap();
        fs::write(moonshine_dir.join("encode.int8.onnx"), b"dummy").unwrap();
        fs::write(moonshine_dir.join("uncached_decode.int8.onnx"), b"dummy").unwrap();
        fs::write(moonshine_dir.join("cached_decode.int8.onnx"), b"dummy").unwrap();
        fs::write(moonshine_dir.join("tokens.txt"), b"dummy").unwrap();

        let kokoro_dir = config.kokoro_dir();
        fs::create_dir_all(&kokoro_dir).unwrap();
        // Missing model.onnx
        fs::write(kokoro_dir.join("voices.bin"), b"dummy").unwrap();
        fs::write(kokoro_dir.join("tokens.txt"), b"dummy").unwrap();

        assert!(!models_ready(&config));
    }

    #[test]
    fn moonshine_url_is_valid() {
        assert!(MOONSHINE_URL.starts_with("https://"));
        assert!(MOONSHINE_URL.contains("sherpa-onnx-moonshine-base-en-int8.tar.bz2"));
    }
}
