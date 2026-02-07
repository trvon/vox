//! DSP parameter sweep tool — brute-force search for optimal DSP values.
//!
//! Run: `cargo run --example dsp_sweep > results.csv`
//!
//! Tests 252 parameter combinations against synthetic signals and outputs
//! CSV with objective quality metrics. Top-5 combos printed to stderr.

const SAMPLE_RATE: f64 = 16000.0;
const NUM_SAMPLES: usize = 16000; // 1 second

// --- Synthetic test signals ---

/// Speech-like signal: 150Hz fundamental + 8 harmonics with decreasing amplitude
fn speech_signal(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64 / SAMPLE_RATE;
            let mut sample = 0.0_f64;
            for h in 1..=8 {
                let amp = 1.0 / h as f64;
                sample += amp * (2.0 * std::f64::consts::PI * 150.0 * h as f64 * t).sin();
            }
            sample as f32
        })
        .collect()
}

/// Deterministic white noise via LCG (no rand dependency)
fn white_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Mix signal with noise at given SNR in dB
fn mix_snr(signal: &[f32], noise: &[f32], snr_db: f32) -> Vec<f32> {
    let sig_power: f32 = signal.iter().map(|s| s * s).sum::<f32>() / signal.len() as f32;
    let noise_power: f32 = noise.iter().map(|s| s * s).sum::<f32>() / noise.len() as f32;
    let target_noise_power = sig_power / 10.0_f32.powf(snr_db / 10.0);
    let scale = if noise_power > 0.0 {
        (target_noise_power / noise_power).sqrt()
    } else {
        0.0
    };
    signal
        .iter()
        .zip(noise.iter())
        .map(|(&s, &n)| s + n * scale)
        .collect()
}

// --- Objective metrics ---

/// SNR in dB: compare processed signal against clean reference
fn snr_db(processed: &[f32], reference: &[f32]) -> f32 {
    let len = processed.len().min(reference.len());
    let signal_power: f32 = reference[..len].iter().map(|s| s * s).sum::<f32>() / len as f32;
    let noise_power: f32 = processed[..len]
        .iter()
        .zip(reference[..len].iter())
        .map(|(&p, &r)| (p - r) * (p - r))
        .sum::<f32>()
        / len as f32;
    if noise_power > 0.0 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        120.0
    }
}

/// Noise floor in dB FS
fn noise_floor_db(samples: &[f32]) -> f32 {
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    if rms > 0.0 {
        20.0 * rms.log10()
    } else {
        -120.0
    }
}

/// Goertzel algorithm: magnitude at a single frequency
fn goertzel_magnitude(samples: &[f32], freq_hz: f64, sample_rate: f64) -> f64 {
    let n = samples.len();
    let k = (freq_hz * n as f64 / sample_rate).round();
    let w = 2.0 * std::f64::consts::PI * k / n as f64;
    let coeff = 2.0 * w.cos();
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    for &sample in samples {
        let s0 = sample as f64 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    (power.abs() / (n as f64)).sqrt()
}

/// Sub-HPF rejection ratio: energy at low freqs vs speech band
fn rejection_ratio(samples: &[f32]) -> f64 {
    let low = goertzel_magnitude(samples, 60.0, SAMPLE_RATE)
        + goertzel_magnitude(samples, 100.0, SAMPLE_RATE);
    let high = goertzel_magnitude(samples, 300.0, SAMPLE_RATE)
        + goertzel_magnitude(samples, 1000.0, SAMPLE_RATE)
        + goertzel_magnitude(samples, 3000.0, SAMPLE_RATE);
    if high > 0.0 { low / high } else { 0.0 }
}

/// Speech band retention: energy at speech freqs before/after processing
fn speech_retention(original: &[f32], processed: &[f32]) -> f64 {
    let freqs = [300.0, 1000.0, 3000.0];
    let orig_energy: f64 = freqs
        .iter()
        .map(|&f| goertzel_magnitude(original, f, SAMPLE_RATE))
        .sum();
    let proc_energy: f64 = freqs
        .iter()
        .map(|&f| goertzel_magnitude(processed, f, SAMPLE_RATE))
        .sum();
    if orig_energy > 0.0 {
        proc_energy / orig_energy
    } else {
        0.0
    }
}

/// Apply the full DSP chain with given parameters
fn apply_chain(
    samples: &mut Vec<f32>,
    hpf_cutoff: f64,
    noise_gate_rms: f32,
    normalize_threshold: f32,
) {
    vox::audio::apply_highpass_filter(samples, hpf_cutoff, SAMPLE_RATE);
    for chunk in samples.chunks_mut(512) {
        vox::audio::apply_noise_gate(chunk, noise_gate_rms);
    }
    vox::audio::peak_normalize(samples, normalize_threshold);
}

#[derive(Clone)]
struct SweepResult {
    hpf_cutoff: f64,
    noise_gate_rms: f32,
    normalize_threshold: f32,
    snr_improvement_db: f32,
    noise_floor: f32,
    retention: f64,
    rejection: f64,
}

fn main() {
    // Generate test signals
    let clean_speech = speech_signal(NUM_SAMPLES);
    let noise = white_noise(NUM_SAMPLES, 42);
    let mixed_20db = mix_snr(&clean_speech, &noise, 20.0);
    let mixed_10db = mix_snr(&clean_speech, &noise, 10.0);

    let baseline_snr_20 = snr_db(&mixed_20db, &clean_speech);
    let baseline_snr_10 = snr_db(&mixed_10db, &clean_speech);

    // Parameter grid (7 * 6 * 6 = 252 combos)
    let hpf_cutoffs = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0];
    let noise_gates = [0.001f32, 0.005, 0.01, 0.02, 0.05, 0.1];
    let norm_thresholds = [0.1f32, 0.2, 0.3, 0.5, 0.7, 1.0];

    // CSV header
    println!(
        "hpf_cutoff,noise_gate_rms,normalize_threshold,snr_improvement_db,noise_floor_db,speech_retention,rejection_ratio"
    );

    let mut results: Vec<SweepResult> = Vec::new();

    for &hpf in &hpf_cutoffs {
        for &gate in &noise_gates {
            for &norm in &norm_thresholds {
                // Test on 20dB SNR mix
                let mut processed_20 = mixed_20db.clone();
                apply_chain(&mut processed_20, hpf, gate, norm);

                // Test on 10dB SNR mix
                let mut processed_10 = mixed_10db.clone();
                apply_chain(&mut processed_10, hpf, gate, norm);

                let snr_after_20 = snr_db(&processed_20, &clean_speech);
                let snr_after_10 = snr_db(&processed_10, &clean_speech);

                // Average SNR improvement across both test conditions
                let snr_improvement =
                    ((snr_after_20 - baseline_snr_20) + (snr_after_10 - baseline_snr_10)) / 2.0;

                let nf = noise_floor_db(&processed_20);
                let ret = speech_retention(&mixed_20db, &processed_20);
                let rej = rejection_ratio(&processed_20);

                println!("{hpf},{gate},{norm},{snr_improvement:.2},{nf:.2},{ret:.4},{rej:.6}");

                results.push(SweepResult {
                    hpf_cutoff: hpf,
                    noise_gate_rms: gate,
                    normalize_threshold: norm,
                    snr_improvement_db: snr_improvement,
                    noise_floor: nf,
                    retention: ret,
                    rejection: rej,
                });
            }
        }
    }

    // Sort by composite score: high retention + low rejection + positive SNR improvement
    results.sort_by(|a, b| {
        let score_a = a.retention - a.rejection as f64 + a.snr_improvement_db as f64 * 0.1;
        let score_b = b.retention - b.rejection as f64 + b.snr_improvement_db as f64 * 0.1;
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    eprintln!("\n=== Top 5 Parameter Combinations ===");
    eprintln!(
        "{:<12} {:<16} {:<20} {:<18} {:<14} {:<16} {:<14}",
        "HPF (Hz)",
        "Gate RMS",
        "Norm Threshold",
        "SNR Improv (dB)",
        "Noise (dB)",
        "Retention",
        "Rejection"
    );
    eprintln!("{}", "-".repeat(110));
    for r in results.iter().take(5) {
        eprintln!(
            "{:<12.0} {:<16} {:<20} {:<18.2} {:<14.2} {:<16.4} {:<14.6}",
            r.hpf_cutoff,
            r.noise_gate_rms,
            r.normalize_threshold,
            r.snr_improvement_db,
            r.noise_floor,
            r.retention,
            r.rejection,
        );
    }
}
