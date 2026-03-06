use crate::config::{Config, KokoroModel};
use crate::error::{Result, VoiceError};
use std::path::Component;
use std::path::Path;
use tokio::io::AsyncWriteExt;

const VAD_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";

const MOONSHINE_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-base-en-int8.tar.bz2";

const KOKORO_INT8_V1_1_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2";
const KOKORO_FP32_V1_1_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_1.tar.bz2";

fn kokoro_download_url(model: KokoroModel) -> &'static str {
    match model {
        KokoroModel::Int8V11 => KOKORO_INT8_V1_1_URL,
        KokoroModel::Fp32V11 => KOKORO_FP32_V1_1_URL,
    }
}

/// Check if all required models are present
pub fn models_ready(config: &Config) -> bool {
    models_ready_with_variant(config, config.kokoro_model)
}

/// Check if all required models are present for a selected Kokoro variant.
pub fn models_ready_with_variant(config: &Config, kokoro_model: KokoroModel) -> bool {
    let moonshine_dir = config.moonshine_dir();
    let kokoro_dir = config.model_dir.join(kokoro_model.model_dir_name());
    let vad_path = config.vad_model_path();

    vad_path.exists()
        && moonshine_dir.join("preprocess.onnx").exists()
        && moonshine_dir.join("encode.int8.onnx").exists()
        && moonshine_dir.join("uncached_decode.int8.onnx").exists()
        && moonshine_dir.join("cached_decode.int8.onnx").exists()
        && moonshine_dir.join("tokens.txt").exists()
        && kokoro_dir.join(kokoro_model.model_file_name()).exists()
        && kokoro_dir.join("voices.bin").exists()
        && kokoro_dir.join("tokens.txt").exists()
}

/// Download all required models
pub async fn download_models(config: &Config) -> Result<()> {
    download_models_with_variant(config, config.kokoro_model).await
}

/// Download all required models with a selected Kokoro variant.
pub async fn download_models_with_variant(
    config: &Config,
    kokoro_model: KokoroModel,
) -> Result<()> {
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
    let kokoro_dir = config.model_dir.join(kokoro_model.model_dir_name());
    if !kokoro_dir.join(kokoro_model.model_file_name()).exists() {
        download_and_extract_tar_bz2(kokoro_download_url(kokoro_model), &config.model_dir).await?;
    } else {
        eprintln!("  Kokoro model ({kokoro_model}) already exists, skipping");
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
    let mut downloaded: u64 = 0;

    let temp_name = format!(
        ".vox-download-{}-{}.tar.bz2.partial",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );
    let temp_path = dest_dir.join(temp_name);
    let mut temp_file = tokio::fs::File::create(&temp_path).await?;

    use tokio_stream::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        downloaded += chunk.len() as u64;
        temp_file.write_all(&chunk).await?;
        if let Some(total) = total {
            eprint!(
                "\r  Progress: {:.1}%",
                (downloaded as f64 / total as f64) * 100.0
            );
        }
    }
    eprintln!();
    temp_file.flush().await?;

    // Extract tar.bz2 in a blocking task (stream from file, no giant in-memory buffer)
    let dest = dest_dir.to_path_buf();
    let extract_path = temp_path.clone();
    let extract_result =
        tokio::task::spawn_blocking(move || extract_tar_bz2_file(&extract_path, &dest))
            .await
            .map_err(|e| VoiceError::Download(format!("Extract task failed: {e}")))?;

    // Best-effort cleanup of temporary archive file
    if let Err(e) = tokio::fs::remove_file(&temp_path).await {
        eprintln!(
            "  Warning: failed to remove temp file {}: {e}",
            temp_path.display()
        );
    }

    extract_result
}

fn extract_tar_bz2_file(archive_path: &Path, dest_dir: &Path) -> Result<()> {
    eprintln!("  Extracting to {} ...", dest_dir.display());

    let file = std::fs::File::open(archive_path)
        .map_err(|e| VoiceError::Download(format!("Failed to open archive: {e}")))?;
    let bz2_reader = bzip2::read::BzDecoder::new(file);
    let mut archive = tar::Archive::new(bz2_reader);

    // Stream extraction per-entry to keep memory usage low.
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

        // Prevent path traversal from archive entries.
        if has_unsafe_archive_path(&path) {
            return Err(VoiceError::Download(format!(
                "Refusing to extract unsafe path: {}",
                path.display()
            )));
        }

        let dest_path = dest_dir.join(&path);

        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&dest_path)
                .map_err(|e| VoiceError::Download(format!("Failed to create dir: {e}")))?;
        } else {
            if let Some(parent) = dest_path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| VoiceError::Download(format!("Failed to create dir: {e}")))?;
            }
            entry
                .unpack(&dest_path)
                .map_err(|e| VoiceError::Download(format!("Failed to unpack entry: {e}")))?;
        }
    }

    eprintln!("  Extraction complete");
    Ok(())
}

fn has_unsafe_archive_path(path: &Path) -> bool {
    path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

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
        fs::write(kokoro_dir.join("model.int8.onnx"), b"dummy").unwrap();
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
        fs::write(kokoro_dir.join("model.int8.onnx"), b"dummy").unwrap();
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
        fs::write(kokoro_dir.join("model.int8.onnx"), b"dummy").unwrap();
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
        // Missing model.int8.onnx
        fs::write(kokoro_dir.join("voices.bin"), b"dummy").unwrap();
        fs::write(kokoro_dir.join("tokens.txt"), b"dummy").unwrap();

        assert!(!models_ready(&config));
    }

    #[test]
    fn moonshine_url_is_valid() {
        assert!(MOONSHINE_URL.starts_with("https://"));
        assert!(MOONSHINE_URL.contains("sherpa-onnx-moonshine-base-en-int8.tar.bz2"));
    }

    #[test]
    fn kokoro_urls_are_valid() {
        assert!(KOKORO_INT8_V1_1_URL.starts_with("https://"));
        assert!(KOKORO_INT8_V1_1_URL.contains("kokoro-int8-multi-lang-v1_1.tar.bz2"));
        assert!(KOKORO_FP32_V1_1_URL.starts_with("https://"));
        assert!(KOKORO_FP32_V1_1_URL.contains("kokoro-multi-lang-v1_1.tar.bz2"));
    }

    #[test]
    fn models_ready_with_fp32_variant_checks_fp32_layout() {
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

        let kokoro_dir = config.model_dir.join("kokoro-multi-lang-v1_1");
        fs::create_dir_all(&kokoro_dir).unwrap();
        fs::write(kokoro_dir.join("model.onnx"), b"dummy").unwrap();
        fs::write(kokoro_dir.join("voices.bin"), b"dummy").unwrap();
        fs::write(kokoro_dir.join("tokens.txt"), b"dummy").unwrap();

        assert!(models_ready_with_variant(&config, KokoroModel::Fp32V11));
        assert!(!models_ready_with_variant(&config, KokoroModel::Int8V11));
    }

    #[test]
    fn extract_tar_bz2_file_extracts_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let archive_path = tmp.path().join("test.tar.bz2");

        let tar_bytes = {
            let mut tar_builder = tar::Builder::new(Vec::<u8>::new());

            let mut header = tar::Header::new_gnu();
            header.set_path("foo/bar.txt").unwrap();
            header.set_mode(0o644);
            header.set_size(5);
            header.set_cksum();
            tar_builder.append(&header, b"hello" as &[u8]).unwrap();

            tar_builder.into_inner().unwrap()
        };

        let mut encoder = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::best());
        encoder.write_all(&tar_bytes).unwrap();
        let compressed = encoder.finish().unwrap();
        fs::write(&archive_path, compressed).unwrap();

        extract_tar_bz2_file(&archive_path, tmp.path()).unwrap();

        let extracted = tmp.path().join("foo/bar.txt");
        assert_eq!(fs::read_to_string(extracted).unwrap(), "hello");
    }

    #[test]
    fn has_unsafe_archive_path_detects_parent_and_root_paths() {
        assert!(has_unsafe_archive_path(Path::new("../evil.txt")));
        assert!(has_unsafe_archive_path(Path::new("/absolute/path.txt")));
        assert!(!has_unsafe_archive_path(Path::new("safe/relative/path.txt")));
    }
}
