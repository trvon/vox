//! Long-form recording and file transcription jobs.
//!
//! A *job* is either a live microphone recording or the transcription of an
//! existing audio file. Both feed the same pipeline:
//!
//! ```text
//! 16 kHz audio ─► Segmenter (VAD + ring buffer) ─► padded speech segments ─► STT ─► transcript.jsonl
//!        └──────► audio.wav (live only)
//! ```
//!
//! Durability: each job lives in its own directory under the recordings root.
//! `transcript.jsonl` is append-only (one line per segment, flushed as written),
//! `audio.wav` has its header rewritten every few seconds so a crash leaves a
//! playable file, and `manifest.json` is replaced atomically. A job whose
//! manifest still says `recording`/`transcribing` but which is not running in
//! any process (checked with an OS file lock) is reported as `interrupted`;
//! status reads never rewrite another process's manifest. Its audio can be re-transcribed
//! with `transcribe_file`.
//!
//! Timeline: times are relative to job start in the *audio* timeline. Windows
//! the capture thread had to drop are replaced with silence and recorded as
//! gaps, so timestamps never drift.

use crate::audio::{self, StreamingResampler};
use crate::config::Config;
use crate::runtime::chrono_now;
use crate::stt::SttEngine;
use crate::transcript::{
    self, BoundaryMerger, DecodedSegment, LayoutOptions, Marker, TranscriptSegment,
};
use crate::vad::{VAD_SAMPLE_RATE, VAD_WINDOW_SIZE, VadOptions, VadSession};

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

const SR: u64 = VAD_SAMPLE_RATE as u64;
/// Audio kept on each side of a VAD segment so word edges aren't clipped.
const SEGMENT_PAD_MS: u64 = 250;
/// Padding on each side of a forced split (speech ran past the VAD's max
/// length). The ~2 s shared window lets both neighbours see the boundary
/// words; `BoundaryMerger` then de-duplicates them by timestamp.
const FORCED_SPLIT_PAD_MS: u64 = 1_000;
/// Ring buffer length: longer than max segment + padding + VAD hangover.
const RING_SECS: u64 = 90;
/// How often the WAV header and manifest are made durable during capture.
const DURABLE_EVERY_SECS: u64 = 5;
/// Archival capture: gentle rumble filter only, no noise gate.
const LECTURE_HPF_HZ: f64 = 80.0;
/// ~2 minutes of 32 ms windows; the consumer never blocks on STT, so this
/// only fills if the async runtime itself stalls.
const LIVE_CHANNEL_WINDOWS: usize = 4096;
/// Bounded gain before STT: boost quiet speech up to 20 dB, never more.
const STT_TARGET_PEAK: f32 = 0.9;
const STT_MAX_GAIN: f32 = 10.0;

static JOB_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Live,
    File,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    /// Live capture running.
    Recording,
    /// Capture finished (or file job running); STT still working.
    Transcribing,
    Completed,
    /// Stopped with `cancel`: audio kept, remaining STT skipped.
    Cancelled,
    /// Process died while the job was active.
    Interrupted,
    Failed,
}

impl JobStatus {
    pub fn is_terminal(self) -> bool {
        !matches!(self, JobStatus::Recording | JobStatus::Transcribing)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gap {
    pub at_ms: u64,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub id: String,
    pub title: String,
    pub kind: JobKind,
    pub status: JobStatus,
    pub started_at: String,
    #[serde(default)]
    pub ended_at: Option<String>,
    /// Input file for `file` jobs.
    #[serde(default)]
    pub source: Option<String>,
    /// Audio file name inside the job directory (live jobs with audio saved).
    #[serde(default)]
    pub audio_file: Option<String>,
    /// Audio captured (live) or decoded (file) so far.
    #[serde(default)]
    pub captured_ms: u64,
    /// Total input length when known (file jobs).
    #[serde(default)]
    pub total_ms: Option<u64>,
    /// End time of the latest transcribed segment.
    #[serde(default)]
    pub transcribed_through_ms: u64,
    #[serde(default)]
    pub segment_count: u64,
    /// Segments detected but not yet transcribed.
    #[serde(default)]
    pub pending_segments: u64,
    #[serde(default)]
    pub dropped_windows: u64,
    #[serde(default)]
    pub gaps: Vec<Gap>,
    /// Speech fed to STT and wall time spent in STT; their ratio is the RTF.
    #[serde(default)]
    pub speech_ms: u64,
    #[serde(default)]
    pub stt_compute_ms: u64,
    #[serde(default)]
    pub stt_errors: u64,
    #[serde(default)]
    pub last_text: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

impl Manifest {
    fn new(id: String, title: String, kind: JobKind, source: Option<String>) -> Self {
        Self {
            id,
            title,
            kind,
            status: if kind == JobKind::Live {
                JobStatus::Recording
            } else {
                JobStatus::Transcribing
            },
            started_at: chrono_now(),
            ended_at: None,
            source,
            audio_file: None,
            captured_ms: 0,
            total_ms: None,
            transcribed_through_ms: 0,
            segment_count: 0,
            pending_segments: 0,
            dropped_windows: 0,
            gaps: Vec::new(),
            speech_ms: 0,
            stt_compute_ms: 0,
            stt_errors: 0,
            last_text: None,
            error: None,
        }
    }

    /// STT real-time factor (compute / speech); < 1 means faster than real time.
    pub fn stt_rtf(&self) -> Option<f64> {
        (self.speech_ms > 0).then(|| self.stt_compute_ms as f64 / self.speech_ms as f64)
    }
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct RecordingStore {
    root: PathBuf,
    manifest_lock: Arc<Mutex<()>>,
}

/// Job ids are generated by us; reject anything else so ids can't escape the root.
pub(crate) fn validate_id(id: &str) -> Result<(), String> {
    let ok = !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if ok {
        Ok(())
    } else {
        Err(format!("invalid recording id: {id:?}"))
    }
}

pub(crate) fn new_job_id(kind: JobKind) -> String {
    let now = chrono_now(); // 2026-10-27T12:00:00Z
    let compact: String = now.chars().filter(|c| c.is_ascii_digit()).collect();
    let n = JOB_COUNTER.fetch_add(1, Ordering::Relaxed);
    let prefix = match kind {
        JobKind::Live => "rec",
        JobKind::File => "file",
    };
    format!(
        "{prefix}-{}-{}-{}{n}",
        &compact[..8],
        &compact[8..14],
        std::process::id() % 10_000
    )
}

impl RecordingStore {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            manifest_lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn dir(&self, id: &str) -> Result<PathBuf, String> {
        validate_id(id)?;
        Ok(self.root.join(id))
    }

    fn file(&self, id: &str, name: &str) -> Result<PathBuf, String> {
        Ok(self.dir(id)?.join(name))
    }

    /// Reserve a unique directory and hold an OS lock for the job's lifetime.
    pub fn create(&self, manifest: &Manifest) -> Result<std::fs::File, String> {
        let dir = self.dir(&manifest.id)?;
        std::fs::create_dir_all(&self.root).map_err(|e| format!("recordings root: {e}"))?;
        std::fs::create_dir(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let owner = std::fs::OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(dir.join("owner.lock"))
            .map_err(|e| format!("job ownership: {e}"))?;
        owner
            .try_lock()
            .map_err(|e| format!("job ownership: {e}"))?;
        self.save_manifest(manifest)?;
        Ok(owner)
    }

    fn is_owned(&self, id: &str) -> Result<bool, String> {
        let path = self.file(id, "owner.lock")?;
        let owner = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
        {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(format!("job ownership: {e}")),
        };
        match owner.try_lock() {
            Ok(()) => Ok(false), // Released when this probe's file is dropped.
            Err(std::fs::TryLockError::WouldBlock) => Ok(true),
            Err(e) => Err(format!("job ownership: {e}")),
        }
    }

    /// Read-only recovery view. Never declare a job interrupted merely because
    /// it belongs to a different CLI/daemon process.
    fn observed_manifest(&self, id: &str) -> Result<Manifest, String> {
        let mut m = self.load_manifest(id)?;
        if !m.status.is_terminal() && !self.is_owned(id)? {
            // The owner may have finalized between our first read and probe.
            m = self.load_manifest(id)?;
            if !m.status.is_terminal() {
                m.status = JobStatus::Interrupted;
                m.error
                    .get_or_insert_with(|| "process exited while job was active".into());
            }
        }
        Ok(m)
    }

    /// Atomic replace: write a temp file, fsync, rename.
    pub fn save_manifest(&self, m: &Manifest) -> Result<(), String> {
        let _guard = self
            .manifest_lock
            .lock()
            .map_err(|e| format!("manifest lock: {e}"))?;
        let path = self.file(&m.id, "manifest.json")?;
        let tmp = path.with_extension(format!(
            "json.{}.{}.tmp",
            std::process::id(),
            JOB_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let json = serde_json::to_vec_pretty(m).map_err(|e| e.to_string())?;
        let mut f = std::fs::File::create(&tmp).map_err(|e| format!("manifest: {e}"))?;
        f.write_all(&json).map_err(|e| format!("manifest: {e}"))?;
        f.sync_data().map_err(|e| format!("manifest: {e}"))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("manifest: {e}"))?;
        std::fs::File::open(path.parent().ok_or("manifest parent missing")?)
            .and_then(|dir| dir.sync_all())
            .map_err(|e| format!("manifest directory sync: {e}"))?;
        Ok(())
    }

    pub fn load_manifest(&self, id: &str) -> Result<Manifest, String> {
        let path = self.file(id, "manifest.json")?;
        let data = std::fs::read(&path).map_err(|_| format!("recording not found: {id}"))?;
        serde_json::from_slice(&data).map_err(|e| format!("corrupt manifest for {id}: {e}"))
    }

    pub fn list(&self) -> Vec<Manifest> {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Vec::new();
        };
        let mut out: Vec<Manifest> = entries
            .flatten()
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .filter_map(|id| self.load_manifest(&id).ok())
            .collect();
        out.sort_by(|a, b| b.started_at.cmp(&a.started_at).then(b.id.cmp(&a.id)));
        out
    }

    fn append_jsonl<T: Serialize>(&self, id: &str, name: &str, value: &T) -> Result<(), String> {
        let path = self.file(id, name)?;
        let mut line = serde_json::to_string(value).map_err(|e| e.to_string())?;
        line.push('\n');
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        f.write_all(line.as_bytes())
            .and_then(|_| f.flush())
            .and_then(|_| f.sync_data())
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Tolerate an incomplete final line (crash or concurrent append), but never
    /// turn an I/O failure or corruption in the middle into a complete empty transcript.
    fn read_jsonl<T: for<'de> Deserialize<'de>>(
        &self,
        id: &str,
        name: &str,
    ) -> Result<Vec<T>, String> {
        let path = self.file(id, name)?;
        let f = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        let mut lines = std::io::BufReader::new(f).lines().peekable();
        let mut out = Vec::new();
        while let Some(line) = lines.next() {
            let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
            match serde_json::from_str(&line) {
                Ok(value) => out.push(value),
                Err(e) if e.is_eof() && lines.peek().is_none() => break,
                Err(e) => return Err(format!("corrupt {}: {e}", path.display())),
            }
        }
        Ok(out)
    }

    pub fn append_segment(&self, id: &str, seg: &TranscriptSegment) -> Result<(), String> {
        self.append_jsonl(id, "transcript.jsonl", seg)
    }

    pub fn read_segments(&self, id: &str) -> Result<Vec<TranscriptSegment>, String> {
        self.read_jsonl(id, "transcript.jsonl")
    }

    pub fn append_marker(&self, id: &str, m: &Marker) -> Result<(), String> {
        self.append_jsonl(id, "markers.jsonl", m)
    }

    pub fn read_markers(&self, id: &str) -> Result<Vec<Marker>, String> {
        self.read_jsonl(id, "markers.jsonl")
    }
}

// ---------------------------------------------------------------------------
// Segmenter: VAD + ring buffer → padded segments with absolute timestamps
// ---------------------------------------------------------------------------

/// Fixed-capacity buffer of the most recent samples, addressed by absolute
/// sample index.
pub(crate) struct AudioRing {
    buf: VecDeque<f32>,
    cap: usize,
    /// Absolute index of `buf[0]`.
    start: u64,
}

impl AudioRing {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(cap),
            cap: cap.max(1),
            start: 0,
        }
    }

    pub(crate) fn end(&self) -> u64 {
        self.start + self.buf.len() as u64
    }

    pub(crate) fn push(&mut self, samples: &[f32]) {
        self.buf.extend(samples.iter().copied());
        let excess = self.buf.len().saturating_sub(self.cap);
        if excess > 0 {
            self.buf.drain(..excess);
            self.start += excess as u64;
        }
    }

    pub(crate) fn push_silence(&mut self, n: usize) {
        // Bounded by cap, so a huge gap doesn't allocate.
        let keep = n.min(self.cap);
        self.start += (n - keep) as u64;
        self.push(&vec![0.0; keep]);
    }

    /// Samples in `[from, to)` clamped to what is still buffered.
    pub(crate) fn slice(&self, from: u64, to: u64) -> Vec<f32> {
        let from = from.max(self.start);
        let to = to.min(self.end());
        if from >= to {
            return Vec::new();
        }
        let (a, b) = ((from - self.start) as usize, (to - self.start) as usize);
        self.buf.range(a..b).copied().collect()
    }
}

/// A speech segment ready for STT.
pub(crate) struct PendingSegment {
    /// Unpadded VAD bounds.
    pub start_ms: u64,
    pub end_ms: u64,
    /// Absolute time of `samples[0]` (start minus padding, clamped).
    pub audio_start_ms: u64,
    /// The VAD cut this segment at its max length.
    pub forced_split: bool,
    pub samples: Vec<f32>,
}

pub(crate) struct Segmenter {
    vad: VadSession,
    ring: AudioRing,
    total: u64,
    pad: u64,
    forced_pad: u64,
    /// Segments at least this long (samples) were force-split by the VAD.
    forced_len: u64,
    /// The previous segment was force-split, so this one starts mid-speech.
    after_forced: bool,
}

impl Segmenter {
    pub(crate) fn new(config: &Config) -> Result<Self, String> {
        let opts = VadOptions::for_lecture();
        let vad = VadSession::new(config, opts).map_err(|e| format!("VAD init failed: {e}"))?;
        Ok(Self {
            vad,
            ring: AudioRing::new((RING_SECS * SR) as usize),
            total: 0,
            pad: SEGMENT_PAD_MS * SR / 1000,
            forced_pad: FORCED_SPLIT_PAD_MS * SR / 1000,
            after_forced: false,
            forced_len: (opts.max_speech_secs * 0.98 * SR as f32) as u64,
        })
    }

    /// Samples consumed so far (16 kHz).
    pub(crate) fn total_samples(&self) -> u64 {
        self.total
    }

    pub(crate) fn push(&mut self, samples: &[f32]) -> Vec<PendingSegment> {
        self.ring.push(samples);
        self.total += samples.len() as u64;
        self.vad.accept_waveform(samples);
        self.drain()
    }

    /// Insert silence for audio we lost, keeping the VAD and timeline aligned.
    pub(crate) fn push_silence(&mut self, n: usize) -> Vec<PendingSegment> {
        self.ring.push_silence(n);
        self.total += n as u64;
        let window = vec![0.0f32; VAD_WINDOW_SIZE];
        let mut left = n;
        while left > 0 {
            let k = left.min(VAD_WINDOW_SIZE);
            self.vad.accept_waveform(&window[..k]);
            left -= k;
        }
        self.drain()
    }

    pub(crate) fn finish(&mut self) -> Vec<PendingSegment> {
        self.vad.flush();
        self.drain()
    }

    fn drain(&mut self) -> Vec<PendingSegment> {
        self.vad
            .collect_segments()
            .into_iter()
            .map(|seg| {
                let start = seg.start_sample;
                let end = start + seg.samples.len() as u64;
                let forced = end - start >= self.forced_len;
                let pad_before = if self.after_forced {
                    self.forced_pad
                } else {
                    self.pad
                };
                let pad_after = if forced { self.forced_pad } else { self.pad };
                self.after_forced = forced;
                let from = start.saturating_sub(pad_before).max(self.ring.start);
                let padded = self.ring.slice(from, end + pad_after);
                // Fall back to the VAD's own copy if the ring already moved on.
                let (audio_start, samples) = if padded.len() >= seg.samples.len() {
                    (from, padded)
                } else {
                    (start, seg.samples)
                };
                PendingSegment {
                    start_ms: start * 1000 / SR,
                    end_ms: end * 1000 / SR,
                    audio_start_ms: audio_start * 1000 / SR,
                    forced_split: forced,
                    samples,
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

struct Job {
    manifest: Mutex<Manifest>,
    /// Stop capture / decoding (finalize what we have).
    stop: CancellationToken,
    /// Skip remaining STT work.
    cancel: AtomicBool,
    done: tokio::sync::watch::Sender<bool>,
    /// Serialize snapshot + persistence, not just the final rename.
    persistence: Mutex<()>,
    owner: Mutex<Option<std::fs::File>>,
}

impl Job {
    fn create(store: &RecordingStore, manifest: Manifest) -> Result<Self, String> {
        let owner = store.create(&manifest)?;
        let (done, _) = tokio::sync::watch::channel(false);
        Ok(Self {
            manifest: Mutex::new(manifest),
            stop: CancellationToken::new(),
            cancel: AtomicBool::new(false),
            done,
            persistence: Mutex::new(()),
            owner: Mutex::new(Some(owner)),
        })
    }

    fn snapshot(&self) -> Manifest {
        self.manifest.lock().unwrap().clone()
    }

    fn update<R>(&self, f: impl FnOnce(&mut Manifest) -> R) -> R {
        f(&mut self.manifest.lock().unwrap())
    }

    fn persist(&self, store: &RecordingStore) -> Result<(), String> {
        let _guard = self
            .persistence
            .lock()
            .map_err(|e| format!("job persistence: {e}"))?;
        store.save_manifest(&self.snapshot())
    }

    fn commit_segment(
        &self,
        store: &RecordingStore,
        mut seg: TranscriptSegment,
    ) -> Result<(), String> {
        let mut m = self
            .manifest
            .lock()
            .map_err(|e| format!("job manifest: {e}"))?;
        seg.seq = m.segment_count + 1;
        store.append_segment(&m.id, &seg)?;
        m.segment_count = seg.seq;
        m.last_text = Some(seg.text);
        Ok(())
    }

    fn finish(&self, store: &RecordingStore, status: JobStatus, error: Option<String>) {
        self.update(|m| {
            m.status = status;
            m.pending_segments = 0;
            m.ended_at = Some(chrono_now());
            if error.is_some() {
                m.error = error;
            }
        });
        if let Err(e) = self.persist(store) {
            tracing::error!("final manifest save failed: {e}");
            self.update(|m| {
                m.status = JobStatus::Failed;
                m.error = Some(e);
            });
        }
        self.owner.lock().unwrap().take();
        self.done.send_replace(true);
    }
}

/// A gap belongs immediately before the chunk whose absolute position reveals
/// it, never before older audio that was already queued when overflow happened.
fn gap_before(expected: u64, next: u64) -> Result<Option<Gap>, String> {
    let missing = next
        .checked_sub(expected)
        .ok_or("capture chunks arrived out of order")?;
    Ok((missing > 0).then(|| Gap {
        at_ms: expected * 1000 / SR,
        duration_ms: missing * 1000 / SR,
    }))
}

type LiveWav = hound::WavWriter<std::io::BufWriter<std::fs::File>>;

fn write_live_audio(wav: &mut Option<LiveWav>, samples: &[f32]) -> Result<(), String> {
    if let Some(w) = wav {
        for &s in samples {
            w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                .map_err(|e| format!("audio.wav write: {e}"))?;
        }
    }
    Ok(())
}

fn push_live_chunk(
    job: &Job,
    segmenter: &mut Segmenter,
    wav: &mut Option<LiveWav>,
    chunk: audio::AudioChunk,
) -> Result<Vec<PendingSegment>, String> {
    let expected = segmenter.total_samples();
    let mut segments = Vec::new();
    if let Some(gap) = gap_before(expected, chunk.start_sample)? {
        let mut missing = chunk.start_sample - expected;
        job.update(|m| {
            m.dropped_windows += missing / VAD_WINDOW_SIZE as u64;
            m.gaps.push(gap);
        });
        let silence = [0.0; VAD_WINDOW_SIZE];
        // Feed/drain incrementally so even a large gap doesn't evict speech
        // still waiting inside the VAD or allocate a gap-sized sample buffer.
        while missing > 0 {
            let n = missing.min(VAD_WINDOW_SIZE as u64) as usize;
            write_live_audio(wav, &silence[..n])?;
            segments.extend(segmenter.push_silence(n));
            missing -= n as u64;
        }
    }
    write_live_audio(wav, &chunk.samples)?;
    segments.extend(segmenter.push(&chunk.samples));
    job.update(|m| m.captured_ms = segmenter.total_samples() * 1000 / SR);
    Ok(segments)
}

async fn dispatch_segments(
    job: &Job,
    tx: &tokio::sync::mpsc::Sender<PendingSegment>,
    segments: Vec<PendingSegment>,
) -> Result<(), String> {
    for seg in segments {
        job.update(|m| m.pending_segments += 1);
        if tx.send(seg).await.is_err() {
            job.update(|m| m.pending_segments = m.pending_segments.saturating_sub(1));
            return Err("STT worker stopped; audio is retained".into());
        }
    }
    Ok(())
}

/// Decodes segments, reconciles boundaries between force-split neighbours,
/// and appends final segments to the transcript.
struct SegmentSink<'a> {
    job: &'a Job,
    store: &'a RecordingStore,
    stt: &'a Mutex<SttEngine>,
    merger: BoundaryMerger,
}

impl<'a> SegmentSink<'a> {
    fn new(job: &'a Job, store: &'a RecordingStore, stt: &'a Mutex<SttEngine>) -> Self {
        Self {
            job,
            store,
            stt,
            // Only segments whose unpadded bounds nearly touch are reconciled.
            merger: BoundaryMerger::new(2 * SEGMENT_PAD_MS),
        }
    }

    fn handle(&mut self, seg: PendingSegment) -> Result<(), String> {
        if let Some(decoded) = self.decode(seg)? {
            for done in self.merger.push(decoded) {
                self.emit(done)?;
            }
        }
        self.job.persist(self.store)
    }

    fn finish(&mut self) -> Result<(), String> {
        for done in self.merger.finish() {
            self.emit(done)?;
        }
        self.job.persist(self.store)
    }

    fn decode(&self, seg: PendingSegment) -> Result<Option<DecodedSegment>, String> {
        let speech_ms = seg.end_ms.saturating_sub(seg.start_ms);
        if self.job.cancel.load(Ordering::Relaxed) {
            self.job
                .update(|m| m.pending_segments = m.pending_segments.saturating_sub(1));
            return Ok(None);
        }
        let mut samples = seg.samples;
        audio::normalize_bounded(&mut samples, STT_TARGET_PEAK, STT_MAX_GAIN);

        let t0 = std::time::Instant::now();
        let result = self
            .stt
            .lock()
            .map_err(|e| format!("STT lock poisoned: {e}"))
            .and_then(|mut s| {
                s.transcribe_timed(SR as u32, &samples)
                    .map_err(|e| e.to_string())
            });
        let compute_ms = t0.elapsed().as_millis() as u64;

        self.job.update(|m| {
            m.pending_segments = m.pending_segments.saturating_sub(1);
            m.speech_ms += speech_ms;
            m.stt_compute_ms += compute_ms;
            m.transcribed_through_ms = m.transcribed_through_ms.max(seg.end_ms);
        });

        match result {
            Ok(out) => {
                let tokens: Vec<(String, u64)> = out
                    .tokens
                    .into_iter()
                    .map(|t| (t.text, seg.audio_start_ms + t.start_ms))
                    .collect();
                Ok(Some(DecodedSegment {
                    start_ms: seg.start_ms,
                    end_ms: seg.end_ms,
                    forced_split: seg.forced_split,
                    text: out.text,
                    words: transcript::words_from_tokens(&tokens),
                }))
            }
            Err(e) => {
                tracing::warn!(id = %self.job.snapshot().id, "segment STT failed: {e}");
                self.job.update(|m| m.stt_errors += 1);
                Err(format!("segment STT failed: {e}"))
            }
        }
    }

    fn emit(&self, seg: DecodedSegment) -> Result<(), String> {
        let Some(text) = transcript::clean_segment_text(&seg.text) else {
            return Ok(());
        };
        // Keep word times only while they still describe the stored text.
        let joined = seg
            .words
            .iter()
            .map(|w| w.w.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let words = if transcript::normalize_words(&joined) == transcript::normalize_words(&text) {
            seg.words
        } else {
            Vec::new()
        };
        self.job.commit_segment(
            self.store,
            TranscriptSegment {
                seq: 0, // Assigned only when the append succeeds.
                start_ms: seg.start_ms,
                end_ms: seg.end_ms,
                text,
                words,
            },
        )
    }
}

/// Live view of a job for status tools.
#[derive(Debug, Clone, Serialize)]
pub struct JobView {
    #[serde(flatten)]
    pub manifest: Manifest,
    pub active: bool,
    pub dir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_path: Option<String>,
    pub transcript_path: String,
    /// Audio not yet transcribed (live latency / file progress indicator).
    pub lag_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stt_rtf: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptPage {
    pub recording_id: String,
    pub status: JobStatus,
    pub segments: Vec<TranscriptSegment>,
    /// Pass as `cursor` to fetch the next page. Absent when the transcript is
    /// complete and fully read.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<u64>,
    pub complete: bool,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema, clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Markdown,
    Text,
    Srt,
    Vtt,
    Json,
}

impl ExportFormat {
    fn extension(self) -> &'static str {
        match self {
            ExportFormat::Markdown => "md",
            ExportFormat::Text => "txt",
            ExportFormat::Srt => "srt",
            ExportFormat::Vtt => "vtt",
            ExportFormat::Json => "json",
        }
    }
}

/// Page through segments with `seq > cursor`.
pub(crate) fn page_segments(
    all: &[TranscriptSegment],
    cursor: Option<u64>,
    limit: usize,
    finished: bool,
) -> (Vec<TranscriptSegment>, Option<u64>, bool) {
    let after = cursor.unwrap_or(0);
    let rest: Vec<&TranscriptSegment> = all.iter().filter(|s| s.seq > after).collect();
    let page: Vec<TranscriptSegment> = rest
        .iter()
        .take(limit.max(1))
        .map(|s| (*s).clone())
        .collect();
    let more_now = rest.len() > page.len();
    let last = page.last().map(|s| s.seq).unwrap_or(after);
    let complete = finished && !more_now;
    let next = if complete { None } else { Some(last) };
    (page, next, complete)
}

pub struct RecordingManager {
    store: RecordingStore,
    stt: Arc<Mutex<SttEngine>>,
    jobs: Mutex<HashMap<String, Arc<Job>>>,
    live_id: Mutex<Option<String>>,
}

impl RecordingManager {
    pub fn new(root: PathBuf, stt: Arc<Mutex<SttEngine>>) -> Self {
        Self {
            store: RecordingStore::new(root),
            stt,
            jobs: Mutex::new(HashMap::new()),
            live_id: Mutex::new(None),
        }
    }

    fn job(&self, id: &str) -> Option<Arc<Job>> {
        self.jobs.lock().unwrap().get(id).cloned()
    }

    fn register(&self, manifest: Manifest) -> Result<Arc<Job>, String> {
        let id = manifest.id.clone();
        let job = Arc::new(Job::create(&self.store, manifest)?);
        let mut jobs = self.jobs.lock().unwrap();
        // History lives on disk; don't retain completed jobs forever.
        jobs.retain(|_, j| !j.snapshot().status.is_terminal());
        jobs.insert(id, job.clone());
        Ok(job)
    }

    /// Current manifest: live state for running jobs, otherwise from disk
    /// (non-terminal jobs not running here are marked interrupted).
    pub fn manifest(&self, id: &str) -> Result<Manifest, String> {
        if let Some(job) = self.job(id) {
            return Ok(job.snapshot());
        }
        self.store.observed_manifest(id)
    }

    pub fn view(&self, id: &str) -> Result<JobView, String> {
        let m = self.manifest(id)?;
        let active = !m.status.is_terminal();
        let dir = self.store.dir(id)?;
        let lag_ms = m.captured_ms.saturating_sub(m.transcribed_through_ms);
        let progress = m
            .total_ms
            .filter(|t| *t > 0)
            .map(|t| (m.captured_ms as f64 / t as f64).min(1.0));
        Ok(JobView {
            active,
            audio_path: m
                .audio_file
                .as_ref()
                .map(|f| dir.join(f).to_string_lossy().into_owned()),
            transcript_path: dir.join("transcript.jsonl").to_string_lossy().into_owned(),
            dir: dir.to_string_lossy().into_owned(),
            lag_ms,
            progress,
            stt_rtf: m.stt_rtf(),
            manifest: m,
        })
    }

    pub fn list(&self, limit: usize) -> Vec<JobView> {
        self.store
            .list()
            .into_iter()
            .take(limit)
            .filter_map(|m| self.view(&m.id).ok())
            .collect()
    }

    /// The running live recording, if any.
    pub fn live_id(&self) -> Option<String> {
        self.live_id.lock().unwrap().clone()
    }

    fn resolve(&self, id: Option<&str>) -> Result<String, String> {
        match id {
            Some(id) => {
                validate_id(id)?;
                Ok(id.to_string())
            }
            None => self
                .live_id()
                .ok_or_else(|| "no live recording is active; pass recording_id".to_string()),
        }
    }

    pub fn start_live(
        self: &Arc<Self>,
        config: &Config,
        title: Option<String>,
        save_audio: bool,
    ) -> Result<JobView, String> {
        let mut live = self.live_id.lock().unwrap();
        if let Some(id) = live.as_ref() {
            return Err(format!("recording {id} is already active; stop it first"));
        }
        let id = new_job_id(JobKind::Live);
        let title = title.unwrap_or_else(|| format!("Recording {}", chrono_now()));
        let mut manifest = Manifest::new(id.clone(), title, JobKind::Live, None);
        if save_audio {
            manifest.audio_file = Some("audio.wav".into());
        }

        // Open the mic before registering so a missing device fails cleanly.
        let (rx, capture) =
            audio::start_capture_with(LECTURE_HPF_HZ, 0.0, VAD_WINDOW_SIZE, LIVE_CHANNEL_WINDOWS)
                .map_err(|e| format!("Capture failed: {e}"))?;
        let segmenter = Segmenter::new(config)?;

        let job = self.register(manifest)?;
        *live = Some(id.clone());
        drop(live);

        let this = self.clone();
        tokio::spawn(async move {
            let result = this
                .run_live(job.clone(), rx, capture, segmenter, save_audio)
                .await;
            let cancelled = job.cancel.load(Ordering::Relaxed);
            match result {
                Ok(()) if cancelled => job.finish(&this.store, JobStatus::Cancelled, None),
                Ok(()) => job.finish(&this.store, JobStatus::Completed, None),
                Err(e) => {
                    tracing::error!(id = %job.snapshot().id, "recording failed: {e}");
                    job.finish(&this.store, JobStatus::Failed, Some(e));
                }
            }
            let mut live = this.live_id.lock().unwrap();
            if live.as_deref() == Some(job.snapshot().id.as_str()) {
                *live = None;
            }
        });

        self.view(&id)
    }

    async fn run_live(
        &self,
        job: Arc<Job>,
        mut rx: tokio::sync::mpsc::Receiver<audio::AudioChunk>,
        mut capture: audio::CaptureHandle,
        mut segmenter: Segmenter,
        save_audio: bool,
    ) -> Result<(), String> {
        let id = job.snapshot().id;
        let dir = self.store.dir(&id)?;
        let mut wav = if save_audio {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: SR as u32,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            Some(
                hound::WavWriter::create(dir.join("audio.wav"), spec)
                    .map_err(|e| format!("audio.wav: {e}"))?,
            )
        } else {
            None
        };

        // Bound decoded-audio backlog. Backpressure can fill the capture channel,
        // whose absolute positions let us account for losses without timestamp drift.
        let (seg_tx, mut seg_rx) = tokio::sync::mpsc::channel::<PendingSegment>(8);
        let worker = {
            let job = job.clone();
            let store = self.store.clone();
            let stt = self.stt.clone();
            tokio::task::spawn_blocking(move || {
                let mut sink = SegmentSink::new(&job, &store, &stt);
                while let Some(seg) = seg_rx.blocking_recv() {
                    sink.handle(seg)?;
                }
                sink.finish()
            })
        };

        let mut last_durable = 0u64;
        let mut result: Result<(), String> = async {
            loop {
                let chunk = tokio::select! {
                    _ = job.stop.cancelled() => break,
                    c = rx.recv() => c.ok_or("audio capture ended unexpectedly; saved audio is retained")?,
                };
                let segments = push_live_chunk(&job, &mut segmenter, &mut wav, chunk)?;
                dispatch_segments(&job, &seg_tx, segments).await?;
                let total = segmenter.total_samples();
                if total - last_durable >= DURABLE_EVERY_SECS * SR {
                    last_durable = total;
                    if let Some(w) = wav.as_mut() {
                        w.flush().map_err(|e| format!("audio.wav flush: {e}"))?;
                    }
                    job.persist(&self.store)?;
                }
            }
            Ok(())
        }.await;

        // Always stop/join capture, drain retained audio, finalize WAV and join
        // the worker, even when STT or persistence fails. No detached writer may
        // append transcript data after the job becomes terminal.
        let capture_end = capture.finish();
        let mut tail = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            tail.push(chunk);
        }
        tail.push(audio::AudioChunk {
            start_sample: capture_end,
            samples: Vec::new(),
        });
        for chunk in tail {
            match push_live_chunk(&job, &mut segmenter, &mut wav, chunk) {
                Ok(segments) if result.is_ok() => {
                    result = dispatch_segments(&job, &seg_tx, segments).await;
                }
                Ok(_) => {}
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        if result.is_ok() {
            result = dispatch_segments(&job, &seg_tx, segmenter.finish()).await;
        }
        if let Some(w) = wav.take()
            && let Err(e) = w.finalize()
        {
            result = Err(format!("audio.wav finalize: {e}"));
        }
        job.update(|m| {
            m.captured_ms = segmenter.total_samples() * 1000 / SR;
            m.status = JobStatus::Transcribing;
        });
        if let Err(e) = job.persist(&self.store) {
            result = Err(e);
        }
        drop(seg_tx);
        let worker_result = worker.await.map_err(|e| format!("STT worker: {e}"))?;
        job.update(|m| m.pending_segments = 0);
        worker_result?;
        result
    }

    /// Stop a job. `cancel = false` finalizes (waits for pending STT);
    /// `cancel = true` skips remaining STT. Audio is never deleted.
    pub async fn stop(
        &self,
        id: Option<&str>,
        cancel: bool,
        wait: bool,
    ) -> Result<JobView, String> {
        let id = self.resolve(id)?;
        let Some(job) = self.job(&id) else {
            let view = self.view(&id)?;
            return if view.manifest.status.is_terminal() {
                Ok(view)
            } else {
                Err(format!("recording {id} is owned by another process"))
            };
        };
        if job.snapshot().status.is_terminal() {
            return self.view(&id);
        }
        if cancel {
            job.cancel.store(true, Ordering::Relaxed);
        }
        job.stop.cancel();
        if wait {
            let mut rx = job.done.subscribe();
            let _ = rx.wait_for(|d| *d).await;
        }
        self.view(&id)
    }

    pub fn mark(&self, id: Option<&str>, label: String) -> Result<Marker, String> {
        let id = self.resolve(id)?;
        let job = self
            .job(&id)
            .filter(|j| !j.snapshot().status.is_terminal())
            .ok_or_else(|| format!("recording {id} is not active"))?;
        let marker = Marker {
            t_ms: job.snapshot().captured_ms,
            label,
        };
        self.store.append_marker(&id, &marker)?;
        Ok(marker)
    }

    pub fn transcribe_file(
        self: &Arc<Self>,
        config: &Config,
        path: PathBuf,
        title: Option<String>,
    ) -> Result<(JobView, tokio::sync::watch::Receiver<bool>), String> {
        let path = expand_tilde(&path);
        if !path.is_file() {
            return Err(format!("file not found: {}", path.display()));
        }
        let source = AudioSource::open(&path)?;
        let id = new_job_id(JobKind::File);
        let title = title.unwrap_or_else(|| {
            path.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| id.clone())
        });
        let mut manifest = Manifest::new(
            id.clone(),
            title,
            JobKind::File,
            Some(path.to_string_lossy().into_owned()),
        );
        manifest.total_ms = source.total_ms();
        let segmenter = Segmenter::new(config)?;
        let job = self.register(manifest)?;
        let done_rx = job.done.subscribe();

        let this = self.clone();
        tokio::task::spawn_blocking(move || {
            let result = this.run_file(&job, source, segmenter);
            let cancelled = job.cancel.load(Ordering::Relaxed) || job.stop.is_cancelled();
            match result {
                Ok(()) if cancelled => job.finish(&this.store, JobStatus::Cancelled, None),
                Ok(()) => job.finish(&this.store, JobStatus::Completed, None),
                Err(e) => job.finish(&this.store, JobStatus::Failed, Some(e)),
            }
        });

        Ok((self.view(&id)?, done_rx))
    }

    fn run_file(
        &self,
        job: &Job,
        mut source: AudioSource,
        mut seg: Segmenter,
    ) -> Result<(), String> {
        let mut resampler = StreamingResampler::new(source.sample_rate(), SR as u32);
        let mut buf16 = Vec::new();
        let mut last_persist = 0u64;
        let mut sink = SegmentSink::new(job, &self.store, &self.stt);
        let mut handle = |segs: Vec<PendingSegment>| -> Result<(), String> {
            job.update(|m| m.pending_segments += segs.len() as u64);
            for s in segs {
                sink.handle(s)?;
            }
            Ok(())
        };
        while let Some(block) = source.next_block()? {
            if job.stop.is_cancelled() {
                job.cancel.store(true, Ordering::Relaxed);
                break;
            }
            buf16.clear();
            resampler.process(&block, &mut buf16);
            handle(seg.push(&buf16))?;
            let total = seg.total_samples();
            job.update(|m| m.captured_ms = total * 1000 / SR);
            if total - last_persist >= 30 * SR {
                last_persist = total;
                job.persist(&self.store)?;
            }
        }
        buf16.clear();
        resampler.flush(&mut buf16);
        handle(seg.push(&buf16))?;
        handle(seg.finish())?;
        sink.finish()?;
        let total = seg.total_samples();
        job.update(|m| {
            m.captured_ms = total * 1000 / SR;
            m.total_ms = Some(m.total_ms.unwrap_or(m.captured_ms).max(m.captured_ms));
        });
        Ok(())
    }

    pub fn transcript_page(
        &self,
        id: &str,
        cursor: Option<u64>,
        limit: usize,
    ) -> Result<TranscriptPage, String> {
        let m = self.manifest(id)?;
        let all = self.store.read_segments(id)?;
        let (segments, next_cursor, complete) =
            page_segments(&all, cursor, limit, m.status.is_terminal());
        Ok(TranscriptPage {
            recording_id: id.to_string(),
            status: m.status,
            segments,
            next_cursor,
            complete,
        })
    }

    /// Render an export and write it into the job directory.
    pub fn export(
        &self,
        id: &str,
        format: ExportFormat,
        layout: &LayoutOptions,
    ) -> Result<(PathBuf, String), String> {
        let m = self.manifest(id)?;
        let segs = self.store.read_segments(id)?;
        let markers = self.store.read_markers(id)?;
        let paras = transcript::paragraphs(&segs, layout);
        let content = match format {
            ExportFormat::Markdown => transcript::to_markdown(
                &transcript::DocMeta {
                    title: m.title.clone(),
                    recorded_at: Some(m.started_at.clone()),
                    duration_ms: m.captured_ms,
                    source: m.source.clone(),
                },
                &transcript::sections(paras, &markers, layout),
            ),
            ExportFormat::Text => transcript::to_text(&paras),
            ExportFormat::Srt => transcript::to_srt(&segs),
            ExportFormat::Vtt => transcript::to_vtt(&segs),
            ExportFormat::Json => serde_json::to_string_pretty(&serde_json::json!({
                "recording": m,
                "markers": markers,
                "sections": transcript::sections(paras, &markers, layout),
            }))
            .map_err(|e| e.to_string())?,
        };
        let path = self
            .store
            .dir(id)?
            .join(format!("transcript.{}", format.extension()));
        std::fs::write(&path, &content).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok((path, content))
    }

    /// Stop every running job and wait (bounded) for them to finalize.
    pub async fn shutdown(&self, timeout: std::time::Duration) {
        let jobs: Vec<Arc<Job>> = self
            .jobs
            .lock()
            .unwrap()
            .values()
            .filter(|j| !j.snapshot().status.is_terminal())
            .cloned()
            .collect();
        for j in &jobs {
            j.stop.cancel();
        }
        let wait_all = async {
            for j in &jobs {
                let mut rx = j.done.subscribe();
                let _ = rx.wait_for(|d| *d).await;
            }
        };
        if tokio::time::timeout(timeout, wait_all).await.is_err() {
            tracing::warn!("recording shutdown timed out; manifests will show interrupted");
        }
    }
}

fn expand_tilde(p: &Path) -> PathBuf {
    if let Ok(rest) = p.strip_prefix("~")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    p.to_path_buf()
}

// ---------------------------------------------------------------------------
// Audio file decoding
// ---------------------------------------------------------------------------

/// Streams mono f32 blocks from a file: WAV natively, anything else through
/// `ffmpeg` (decoded and resampled to 16 kHz) when it is installed.
pub(crate) enum AudioSource {
    Wav {
        reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
        channels: usize,
        rate: u32,
        total_ms: u64,
    },
    Ffmpeg {
        child: std::process::Child,
        stdout: std::process::ChildStdout,
    },
}

const BLOCK_FRAMES: usize = 8192;

impl AudioSource {
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        let is_wav = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("wav"));
        if is_wav && let Ok(reader) = hound::WavReader::open(path) {
            let spec = reader.spec();
            let total_ms = reader.duration() as u64 * 1000 / spec.sample_rate.max(1) as u64;
            return Ok(AudioSource::Wav {
                channels: spec.channels.max(1) as usize,
                rate: spec.sample_rate,
                total_ms,
                reader,
            });
        }
        let mut child = std::process::Command::new("ffmpeg")
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(path)
            .args(["-f", "f32le", "-ac", "1", "-ar", "16000", "-"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| {
                format!(
                    "{} is not a readable WAV and ffmpeg is unavailable ({e}). Install ffmpeg or convert to WAV.",
                    path.display()
                )
            })?;
        let stdout = child.stdout.take().ok_or("ffmpeg stdout unavailable")?;
        Ok(AudioSource::Ffmpeg { child, stdout })
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        match self {
            AudioSource::Wav { rate, .. } => *rate,
            AudioSource::Ffmpeg { .. } => SR as u32,
        }
    }

    pub(crate) fn total_ms(&self) -> Option<u64> {
        match self {
            AudioSource::Wav { total_ms, .. } => Some(*total_ms),
            AudioSource::Ffmpeg { .. } => None,
        }
    }

    pub(crate) fn next_block(&mut self) -> Result<Option<Vec<f32>>, String> {
        match self {
            AudioSource::Wav {
                reader, channels, ..
            } => {
                let spec = reader.spec();
                let n = BLOCK_FRAMES * *channels;
                let interleaved: Vec<f32> = match spec.sample_format {
                    hound::SampleFormat::Float => reader
                        .samples::<f32>()
                        .take(n)
                        .collect::<Result<_, _>>()
                        .map_err(|e| format!("wav decode: {e}"))?,
                    hound::SampleFormat::Int => {
                        let scale = 1.0 / (1u64 << (spec.bits_per_sample.max(1) - 1)) as f32;
                        reader
                            .samples::<i32>()
                            .take(n)
                            .map(|s| s.map(|v| v as f32 * scale))
                            .collect::<Result<_, _>>()
                            .map_err(|e| format!("wav decode: {e}"))?
                    }
                };
                if interleaved.is_empty() {
                    return Ok(None);
                }
                Ok(Some(downmix(&interleaved, *channels)))
            }
            AudioSource::Ffmpeg { child, stdout } => {
                let mut bytes = vec![0u8; BLOCK_FRAMES * 4];
                let mut filled = 0;
                while filled < bytes.len() {
                    match stdout.read(&mut bytes[filled..]) {
                        Ok(0) => break,
                        Ok(k) => filled += k,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                        Err(e) => return Err(format!("ffmpeg read: {e}")),
                    }
                }
                if filled == 0 {
                    let status = child.wait().map_err(|e| format!("ffmpeg: {e}"))?;
                    if !status.success() {
                        return Err(format!("ffmpeg failed to decode input ({status})"));
                    }
                    return Ok(None);
                }
                filled -= filled % 4;
                Ok(Some(
                    bytes[..filled]
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect(),
                ))
            }
        }
    }
}

impl Drop for AudioSource {
    fn drop(&mut self) {
        if let AudioSource::Ffmpeg { child, .. } = self {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub(crate) fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks(channels)
        .map(|f| f.iter().sum::<f32>() / f.len() as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(seq: u64) -> TranscriptSegment {
        TranscriptSegment {
            seq,
            start_ms: seq * 1000,
            end_ms: seq * 1000 + 500,
            text: format!("s{seq}"),
            words: Vec::new(),
        }
    }

    #[test]
    fn ids_are_valid_and_unique() {
        let a = new_job_id(JobKind::Live);
        let b = new_job_id(JobKind::File);
        assert!(validate_id(&a).is_ok() && validate_id(&b).is_ok());
        assert_ne!(a, b);
        assert!(a.starts_with("rec-") && b.starts_with("file-"));
    }

    #[test]
    fn validate_id_rejects_traversal() {
        for bad in ["", "..", "../x", "a/b", "a b", "x\0", &"a".repeat(65)] {
            assert!(validate_id(bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn ring_slices_by_absolute_index_and_clamps() {
        let mut r = AudioRing::new(10);
        r.push(&(0..8).map(|i| i as f32).collect::<Vec<_>>());
        r.push(&(8..15).map(|i| i as f32).collect::<Vec<_>>());
        assert_eq!(r.end(), 15);
        assert_eq!(r.slice(0, 7), vec![5.0, 6.0]); // 0..5 evicted
        assert_eq!(r.slice(12, 100), vec![12.0, 13.0, 14.0]);
        assert!(r.slice(14, 14).is_empty());
        r.push_silence(1_000_000);
        assert_eq!(r.end(), 1_000_015);
        assert_eq!(r.slice(0, u64::MAX).len(), 10);
    }

    #[test]
    fn paging_walks_everything_once() {
        let all: Vec<_> = (1..=7).map(seg).collect();
        let mut cursor = None;
        let mut seen = Vec::new();
        loop {
            let (page, next, complete) = page_segments(&all, cursor, 3, true);
            seen.extend(page.iter().map(|s| s.seq));
            if complete {
                assert!(next.is_none());
                break;
            }
            cursor = next;
        }
        assert_eq!(seen, (1..=7).collect::<Vec<_>>());
    }

    #[test]
    fn paging_live_job_keeps_cursor_when_caught_up() {
        let all: Vec<_> = (1..=2).map(seg).collect();
        let (page, next, complete) = page_segments(&all, Some(2), 10, false);
        assert!(page.is_empty() && !complete);
        assert_eq!(next, Some(2));
    }

    #[test]
    fn store_roundtrip_and_torn_line_tolerance() {
        let tmp = tempfile::tempdir().unwrap();
        let store = RecordingStore::new(tmp.path().to_path_buf());
        let m = Manifest::new("rec-1".into(), "T".into(), JobKind::Live, None);
        store.create(&m).unwrap();
        store.append_segment("rec-1", &seg(1)).unwrap();
        store.append_segment("rec-1", &seg(2)).unwrap();
        // Simulate a crash mid-write.
        let p = store.dir("rec-1").unwrap().join("transcript.jsonl");
        let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
        f.write_all(b"{\"seq\":3,\"start_").unwrap();
        assert_eq!(store.read_segments("rec-1").unwrap().len(), 2);
        assert_eq!(store.load_manifest("rec-1").unwrap().title, "T");
        assert_eq!(store.list().len(), 1);
        assert!(store.dir("../etc").is_err());
    }

    #[test]
    fn downmix_averages_frames() {
        assert_eq!(downmix(&[1.0, 3.0, -1.0, 1.0], 2), vec![2.0, 0.0]);
        assert_eq!(downmix(&[0.5, 0.25], 1), vec![0.5, 0.25]);
    }

    #[test]
    fn wav_source_reads_and_downmixes() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("t.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 8000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..8000 {
            w.write_sample(i16::MAX / 2).unwrap();
            w.write_sample(0i16).unwrap();
        }
        w.finalize().unwrap();
        let mut src = AudioSource::open(&path).unwrap();
        assert_eq!(src.sample_rate(), 8000);
        assert_eq!(src.total_ms(), Some(1000));
        let mut n = 0;
        while let Some(b) = src.next_block().unwrap() {
            assert!(b.iter().all(|s| (s - 0.25).abs() < 1e-3));
            n += b.len();
        }
        assert_eq!(n, 8000);
    }

    fn test_job(store: &RecordingStore) -> Job {
        Job::create(
            store,
            Manifest::new("rec-test".into(), "Test".into(), JobKind::Live, None),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn completion_is_retained_without_subscribers() {
        let tmp = tempfile::tempdir().unwrap();
        let store = RecordingStore::new(tmp.path().into());
        let job = test_job(&store);
        assert_eq!(job.done.receiver_count(), 0);
        job.finish(&store, JobStatus::Completed, None);
        let mut late = job.done.subscribe();
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            late.wait_for(|done| *done),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(!store.is_owned("rec-test").unwrap());
    }

    #[test]
    fn foreign_owner_is_not_interrupted_and_reads_do_not_write() {
        let tmp = tempfile::tempdir().unwrap();
        let store = RecordingStore::new(tmp.path().into());
        let job = test_job(&store);
        let observer = RecordingStore::new(tmp.path().into());
        let path = tmp.path().join("rec-test/manifest.json");
        let before = std::fs::read(&path).unwrap();
        assert_eq!(
            observer.observed_manifest("rec-test").unwrap().status,
            JobStatus::Recording
        );
        assert_eq!(std::fs::read(&path).unwrap(), before);
        drop(job); // Simulate owner exit without finalizing.
        assert_eq!(
            observer.observed_manifest("rec-test").unwrap().status,
            JobStatus::Interrupted
        );
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn concurrent_persists_keep_valid_latest_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let store = RecordingStore::new(tmp.path().into());
        let job = Arc::new(test_job(&store));
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..30 {
                        job.update(|m| m.captured_ms += 1);
                        job.persist(&store).unwrap();
                        store.load_manifest("rec-test").unwrap();
                    }
                });
            }
        });
        assert_eq!(store.load_manifest("rec-test").unwrap().captured_ms, 120);
    }

    #[test]
    fn failed_append_does_not_commit_segment_count_or_text() {
        let tmp = tempfile::tempdir().unwrap();
        let store = RecordingStore::new(tmp.path().into());
        let job = test_job(&store);
        std::fs::create_dir(tmp.path().join("rec-test/transcript.jsonl")).unwrap();
        let error = job.commit_segment(&store, seg(1)).unwrap_err();
        assert_eq!(job.snapshot().segment_count, 0);
        assert!(job.snapshot().last_text.is_none());
        job.finish(&store, JobStatus::Failed, Some(error));
        assert_eq!(
            store.load_manifest("rec-test").unwrap().status,
            JobStatus::Failed
        );
    }

    #[test]
    fn failed_final_manifest_is_not_reported_completed() {
        let tmp = tempfile::tempdir().unwrap();
        let store = RecordingStore::new(tmp.path().into());
        let job = test_job(&store);
        let path = tmp.path().join("rec-test/manifest.json");
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(path).unwrap();
        job.finish(&store, JobStatus::Completed, None);
        assert_eq!(job.snapshot().status, JobStatus::Failed);
        assert!(job.snapshot().error.is_some());
        assert!(*job.done.subscribe().borrow());
    }

    #[test]
    fn transcript_read_errors_are_not_silently_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let store = RecordingStore::new(tmp.path().into());
        let _job = test_job(&store);
        let path = tmp.path().join("rec-test/transcript.jsonl");
        std::fs::write(&path, b"corrupt middle line\n{}\n").unwrap();
        assert!(store.read_segments("rec-test").is_err());
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(store.read_segments("rec-test").is_err());
        assert!(store.read_segments("../bad").is_err());
    }

    #[test]
    fn duplicate_job_directory_is_never_overwritten() {
        let tmp = tempfile::tempdir().unwrap();
        let store = RecordingStore::new(tmp.path().into());
        let job = test_job(&store);
        assert!(store.create(&job.snapshot()).is_err());
        assert!(store.is_owned("rec-test").unwrap());
    }

    #[tokio::test]
    async fn overflow_gaps_follow_queued_audio_and_include_dropped_tail() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(2);
        let chunk = |start_sample| audio::AudioChunk {
            start_sample,
            samples: vec![0.0; 512],
        };
        tx.try_send(chunk(0)).unwrap();
        tx.try_send(chunk(512)).unwrap();
        assert!(tx.try_send(chunk(1024)).is_err());
        let mut expected = 0;
        for _ in 0..2 {
            let c = rx.recv().await.unwrap();
            assert!(gap_before(expected, c.start_sample).unwrap().is_none());
            expected = c.start_sample + c.samples.len() as u64;
        }
        tx.try_send(chunk(1536)).unwrap();
        let c = rx.recv().await.unwrap();
        assert_eq!(
            gap_before(expected, c.start_sample).unwrap(),
            Some(Gap {
                at_ms: 64,
                duration_ms: 32
            })
        );
        expected = c.start_sample + 512;
        // Capture's final attempted position accounts for a lost final window,
        // even though no later successful chunk exists.
        assert_eq!(
            gap_before(expected, 2560).unwrap(),
            Some(Gap {
                at_ms: 128,
                duration_ms: 32
            })
        );
        assert!(gap_before(expected, 0).is_err());
    }

    #[tokio::test]
    async fn failed_worker_dispatch_does_not_leave_phantom_pending_segment() {
        let tmp = tempfile::tempdir().unwrap();
        let store = RecordingStore::new(tmp.path().into());
        let job = test_job(&store);
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let pending = PendingSegment {
            start_ms: 0,
            end_ms: 1,
            audio_start_ms: 0,
            forced_split: false,
            samples: vec![0.0; 16],
        };
        assert!(dispatch_segments(&job, &tx, vec![pending]).await.is_err());
        assert_eq!(job.snapshot().pending_segments, 0);
    }

    #[test]
    fn terminal_statuses() {
        assert!(!JobStatus::Recording.is_terminal());
        assert!(!JobStatus::Transcribing.is_terminal());
        for s in [
            JobStatus::Completed,
            JobStatus::Cancelled,
            JobStatus::Interrupted,
            JobStatus::Failed,
        ] {
            assert!(s.is_terminal());
        }
    }
}
