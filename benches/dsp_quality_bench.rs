use criterion::{Criterion, black_box, criterion_group, criterion_main};

const SAMPLE_RATE: f64 = 16000.0;

/// Generate a speech-like signal: 150Hz fundamental + harmonics
fn speech_signal(num_samples: usize) -> Vec<f32> {
    let fundamental = 150.0;
    let harmonics = 8;
    (0..num_samples)
        .map(|i| {
            let t = i as f64 / SAMPLE_RATE;
            let mut sample = 0.0_f64;
            for h in 1..=harmonics {
                let amp = 1.0 / h as f64;
                sample += amp
                    * (2.0 * std::f64::consts::PI * fundamental * h as f64 * t).sin();
            }
            sample as f32
        })
        .collect()
}

/// Deterministic white noise via LCG (no rand dependency)
fn white_noise(num_samples: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..num_samples)
        .map(|_| {
            // LCG: Numerical Recipes constants
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map to [-1, 1]
            ((state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Mix signal with noise at a given SNR in dB
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

/// Compute RMS in dB FS
fn noise_floor_db(samples: &[f32]) -> f32 {
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    if rms > 0.0 {
        20.0 * rms.log10()
    } else {
        -120.0
    }
}

/// Goertzel algorithm: compute magnitude at a single frequency
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

fn bench_hpf_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("hpf_quality");

    let n = 16000; // 1 second
    let speech = speech_signal(n);
    let noise = white_noise(n, 42);
    let mixed = mix_snr(&speech, &noise, 20.0);

    group.bench_function("throughput_1s", |b| {
        b.iter(|| {
            let mut s = mixed.clone();
            vox::audio::apply_highpass_filter(black_box(&mut s), 200.0, SAMPLE_RATE);
        })
    });

    // Measure quality: sub-HPF rejection
    group.bench_function("rejection_ratio", |b| {
        b.iter(|| {
            let mut s = mixed.clone();
            vox::audio::apply_highpass_filter(&mut s, 200.0, SAMPLE_RATE);
            let low = goertzel_magnitude(&s, 60.0, SAMPLE_RATE);
            let high = goertzel_magnitude(&s, 1000.0, SAMPLE_RATE);
            black_box(if high > 0.0 { low / high } else { 0.0 })
        })
    });

    group.finish();
}

fn bench_noise_gate_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("noise_gate_quality");

    let quiet: Vec<f32> = vec![0.001; 512];
    let loud = speech_signal(512);

    group.bench_function("quiet_window", |b| {
        b.iter(|| {
            let mut s = quiet.clone();
            vox::audio::apply_noise_gate(black_box(&mut s), 0.01);
            black_box(noise_floor_db(&s))
        })
    });

    group.bench_function("loud_window", |b| {
        b.iter(|| {
            let mut s = loud.clone();
            vox::audio::apply_noise_gate(black_box(&mut s), 0.01);
            black_box(noise_floor_db(&s))
        })
    });

    group.finish();
}

fn bench_normalize_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_quality");

    let quiet: Vec<f32> = speech_signal(24000)
        .iter()
        .map(|s| s * 0.1)
        .collect();

    group.bench_function("quiet_normalize", |b| {
        b.iter(|| {
            let mut s = quiet.clone();
            let peak = vox::audio::peak_normalize(black_box(&mut s), 0.5);
            black_box(peak)
        })
    });

    group.finish();
}

fn bench_full_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_chain");

    let n = 16000;
    let speech = speech_signal(n);
    let noise = white_noise(n, 42);
    let mixed = mix_snr(&speech, &noise, 20.0);

    group.bench_function("hpf_gate_normalize_1s", |b| {
        b.iter(|| {
            let mut s = mixed.clone();
            // HPF
            vox::audio::apply_highpass_filter(&mut s, 200.0, SAMPLE_RATE);
            // Noise gate in windows
            for chunk in s.chunks_mut(512) {
                vox::audio::apply_noise_gate(chunk, 0.01);
            }
            // Normalize
            vox::audio::peak_normalize(&mut s, 0.5);
            black_box(&s);
        })
    });

    // Quality metric: SNR improvement through full chain
    group.bench_function("snr_improvement", |b| {
        b.iter(|| {
            let mut s = mixed.clone();
            let before = noise_floor_db(&s);
            vox::audio::apply_highpass_filter(&mut s, 200.0, SAMPLE_RATE);
            for chunk in s.chunks_mut(512) {
                vox::audio::apply_noise_gate(chunk, 0.01);
            }
            vox::audio::peak_normalize(&mut s, 0.5);
            let after = noise_floor_db(&s);
            black_box(before - after)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hpf_quality,
    bench_noise_gate_quality,
    bench_normalize_quality,
    bench_full_chain
);
criterion_main!(benches);
