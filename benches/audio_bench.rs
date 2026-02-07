use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_resample(c: &mut Criterion) {
    // 1 second of audio at various sample rates
    let samples_48k: Vec<f32> = (0..48000)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 48000.0).sin())
        .collect();
    let samples_8k: Vec<f32> = (0..8000)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 8000.0).sin())
        .collect();
    let samples_16k: Vec<f32> = (0..16000)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16000.0).sin())
        .collect();

    let mut group = c.benchmark_group("resample");
    group.bench_function("48k_to_16k", |b| {
        b.iter(|| vox::audio::resample(black_box(&samples_48k), 48000, 16000))
    });
    group.bench_function("8k_to_16k", |b| {
        b.iter(|| vox::audio::resample(black_box(&samples_8k), 8000, 16000))
    });
    group.bench_function("passthrough_16k", |b| {
        b.iter(|| vox::audio::resample(black_box(&samples_16k), 16000, 16000))
    });
    group.finish();
}

fn bench_highpass_filter(c: &mut Criterion) {
    let samples_16k: Vec<f32> = (0..16000)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16000.0).sin())
        .collect();
    let samples_48k: Vec<f32> = (0..48000)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 48000.0).sin())
        .collect();

    let mut group = c.benchmark_group("highpass_filter");
    group.bench_function("1s_16khz", |b| {
        b.iter(|| {
            let mut s = samples_16k.clone();
            vox::audio::apply_highpass_filter(black_box(&mut s), 200.0, 16000.0);
        })
    });
    group.bench_function("1s_48khz", |b| {
        b.iter(|| {
            let mut s = samples_48k.clone();
            vox::audio::apply_highpass_filter(black_box(&mut s), 200.0, 48000.0);
        })
    });
    group.finish();
}

fn bench_noise_gate(c: &mut Criterion) {
    let quiet: Vec<f32> = vec![0.001; 512];
    let loud: Vec<f32> = vec![0.1; 512];

    let mut group = c.benchmark_group("noise_gate");
    group.bench_function("quiet_512", |b| {
        b.iter(|| {
            let mut s = quiet.clone();
            vox::audio::apply_noise_gate(black_box(&mut s), vox::audio::NOISE_GATE_RMS);
        })
    });
    group.bench_function("loud_512", |b| {
        b.iter(|| {
            let mut s = loud.clone();
            vox::audio::apply_noise_gate(black_box(&mut s), vox::audio::NOISE_GATE_RMS);
        })
    });
    group.finish();
}

fn bench_peak_normalize(c: &mut Criterion) {
    let quiet_1s: Vec<f32> = (0..24000)
        .map(|i| 0.1 * (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 24000.0).sin())
        .collect();
    let loud_1s: Vec<f32> = (0..24000)
        .map(|i| 0.8 * (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 24000.0).sin())
        .collect();
    let quiet_5s: Vec<f32> = (0..120000)
        .map(|i| 0.1 * (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 24000.0).sin())
        .collect();

    let mut group = c.benchmark_group("peak_normalize");
    group.bench_function("quiet_1s", |b| {
        b.iter(|| {
            let mut s = quiet_1s.clone();
            vox::audio::peak_normalize(black_box(&mut s), 0.5);
        })
    });
    group.bench_function("loud_1s_noop", |b| {
        b.iter(|| {
            let mut s = loud_1s.clone();
            vox::audio::peak_normalize(black_box(&mut s), 0.5);
        })
    });
    group.bench_function("quiet_5s", |b| {
        b.iter(|| {
            let mut s = quiet_5s.clone();
            vox::audio::peak_normalize(black_box(&mut s), 0.5);
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_resample,
    bench_highpass_filter,
    bench_noise_gate,
    bench_peak_normalize
);
criterion_main!(benches);
