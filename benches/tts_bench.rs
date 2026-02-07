use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_split_sentences(c: &mut Criterion) {
    let short = "Hello world.";
    let medium = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth one.";
    let long = "The quick brown fox jumps over the lazy dog. \
                She sells seashells by the seashore. \
                Peter Piper picked a peck of pickled peppers. \
                How much wood would a woodchuck chuck. \
                The rain in Spain falls mainly on the plain. \
                Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.";

    let mut group = c.benchmark_group("split_sentences");
    group.bench_function("1_sentence", |b| {
        b.iter(|| vox::tts::split_sentences(black_box(short)))
    });
    group.bench_function("5_sentences", |b| {
        b.iter(|| vox::tts::split_sentences(black_box(medium)))
    });
    group.bench_function("paragraph", |b| {
        b.iter(|| vox::tts::split_sentences(black_box(long)))
    });
    group.finish();
}

fn bench_voice_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("voice_resolution");
    group.bench_function("known_voice", |b| {
        b.iter(|| vox::tts::resolve_voice_id(black_box("af_heart")))
    });
    group.bench_function("last_voice", |b| {
        b.iter(|| vox::tts::resolve_voice_id(black_box("bm_lewis")))
    });
    group.bench_function("numeric_fallback", |b| {
        b.iter(|| vox::tts::resolve_voice_id(black_box("15")))
    });
    group.bench_function("unknown_voice", |b| {
        b.iter(|| vox::tts::resolve_voice_id(black_box("nonexistent")))
    });
    group.finish();
}

criterion_group!(benches, bench_split_sentences, bench_voice_resolution);
criterion_main!(benches);
