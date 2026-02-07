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

criterion_group!(benches, bench_resample);
criterion_main!(benches);
