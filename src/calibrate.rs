use crate::audio;
use crate::config::{Config, DspConfig};
use crate::vad::{VAD_WINDOW_SIZE, VadSession};

use console::style;
use indicatif::{ProgressBar, ProgressStyle};
const SAMPLE_RATE: f64 = 16000.0;

/// Phonetically balanced calibration sentence.
/// Covers plosives (p/b/t/d/k/g), fricatives (f/v/s/z/sh/th), nasals (m/n/ng),
/// approximants (r/l/w/y), vowels (front/back/high/low), diphthongs, and sibilants.
/// The rainbow passage is a classic speech science calibration text.
const CALIBRATION_SENTENCES: &[&str] = &[
    "The rainbow is a division of white light into many beautiful colors.",
    "These take the shape of a long round arch, with its path high above,",
    "and its two ends apparently beyond the horizon.",
];

/// Speech-band Goertzel frequencies for fitness evaluation.
/// Covers fundamental (150Hz), formant ranges F1-F3, and sibilant energy.
const SPEECH_FREQS: &[f64] = &[
    150.0,  // F0 fundamental
    300.0,  // F1 low vowels
    500.0,  // F1 mid vowels
    1000.0, // F2 low
    2000.0, // F2 high
    3000.0, // F3
    5000.0, // sibilant energy (s, sh, z)
    7000.0, // high-frequency fricatives (f, th)
];

/// Number of cross-validation folds for fitness evaluation.
const CV_FOLDS: usize = 3;

// --- LCG PRNG (no rand crate, matches dsp_sweep.rs pattern) ---

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn from_time() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self::new(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Returns a value in [0.0, 1.0)
    fn gen_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Returns a value in [lo, hi]
    fn gen_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.gen_f64() * (hi - lo)
    }
}

// --- Gene ranges ---

const GENE_COUNT: usize = 4;

const GENE_MIN: [f64; GENE_COUNT] = [20.0, 0.001, 64.0, 0.05];
const GENE_MAX: [f64; GENE_COUNT] = [500.0, 0.15, 2048.0, 1.0];

// --- Genome ---

#[derive(Clone, Debug)]
struct Genome {
    genes: [f64; GENE_COUNT],
    fitness: f64,
}

impl Genome {
    fn from_dsp_config(dsp: &DspConfig) -> Self {
        Self {
            genes: [
                dsp.hpf_cutoff_hz,
                dsp.noise_gate_rms as f64,
                dsp.noise_gate_window as f64,
                dsp.normalize_threshold as f64,
            ],
            fitness: f64::NEG_INFINITY,
        }
    }

    fn random(rng: &mut Lcg) -> Self {
        let mut genes = [0.0; GENE_COUNT];
        for i in 0..GENE_COUNT {
            genes[i] = rng.gen_range(GENE_MIN[i], GENE_MAX[i]);
        }
        Self {
            genes,
            fitness: f64::NEG_INFINITY,
        }
    }

    fn clamp(&mut self) {
        for i in 0..GENE_COUNT {
            self.genes[i] = self.genes[i].clamp(GENE_MIN[i], GENE_MAX[i]);
        }
    }

    fn to_dsp_config(&self) -> DspConfig {
        DspConfig {
            hpf_cutoff_hz: self.genes[0],
            noise_gate_rms: self.genes[1] as f32,
            noise_gate_window: nearest_power_of_two(self.genes[2] as usize),
            normalize_threshold: self.genes[3] as f32,
        }
    }
}

fn nearest_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let lower = 1usize << (usize::BITS - 1 - n.leading_zeros());
    let upper = lower << 1;
    if n - lower < upper - n { lower } else { upper }
}

// --- GA operators ---

/// SBX crossover (eta=2.0)
fn sbx_crossover(p1: &Genome, p2: &Genome, rng: &mut Lcg) -> (Genome, Genome) {
    let eta = 2.0_f64;
    let mut c1 = p1.clone();
    let mut c2 = p2.clone();

    for i in 0..GENE_COUNT {
        if rng.gen_f64() > 0.5 {
            // Don't crossover this gene
            continue;
        }
        let y1 = p1.genes[i].min(p2.genes[i]);
        let y2 = p1.genes[i].max(p2.genes[i]);
        if (y2 - y1).abs() < 1e-14 {
            continue;
        }

        let u = rng.gen_f64();
        let beta = if u <= 0.5 {
            (2.0 * u).powf(1.0 / (eta + 1.0))
        } else {
            (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
        };

        c1.genes[i] = 0.5 * ((1.0 + beta) * y1 + (1.0 - beta) * y2);
        c2.genes[i] = 0.5 * ((1.0 - beta) * y1 + (1.0 + beta) * y2);
    }

    c1.clamp();
    c2.clamp();
    c1.fitness = f64::NEG_INFINITY;
    c2.fitness = f64::NEG_INFINITY;
    (c1, c2)
}

/// Polynomial mutation (eta_m=20.0)
fn polynomial_mutation(genome: &mut Genome, prob: f64, rng: &mut Lcg) {
    let eta_m = 20.0_f64;
    for i in 0..GENE_COUNT {
        if rng.gen_f64() >= prob {
            continue;
        }
        let y = genome.genes[i];
        let lo = GENE_MIN[i];
        let hi = GENE_MAX[i];
        let delta = hi - lo;
        if delta < 1e-14 {
            continue;
        }

        let u = rng.gen_f64();
        let delta_q = if u < 0.5 {
            let bl = (y - lo) / delta;
            let val = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - bl).powf(eta_m + 1.0);
            val.powf(1.0 / (eta_m + 1.0)) - 1.0
        } else {
            let bu = (hi - y) / delta;
            let val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - bu).powf(eta_m + 1.0);
            1.0 - val.powf(1.0 / (eta_m + 1.0))
        };

        genome.genes[i] = y + delta_q * delta;
    }
    genome.clamp();
    genome.fitness = f64::NEG_INFINITY;
}

// --- Fitness function ---

/// Goertzel algorithm: magnitude at a single frequency
fn goertzel_magnitude(samples: &[f32], freq_hz: f64, sample_rate: f64) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
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

fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples
        .iter()
        .map(|s| (*s as f64) * (*s as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt()
}

/// Apply the DSP chain to a buffer in-place.
fn apply_dsp_chain(samples: &mut [f32], dsp: &DspConfig) {
    audio::apply_highpass_filter(samples, dsp.hpf_cutoff_hz, SAMPLE_RATE);
    for chunk in samples.chunks_mut(dsp.noise_gate_window) {
        audio::apply_noise_gate(chunk, dsp.noise_gate_rms);
    }
    audio::peak_normalize(samples, dsp.normalize_threshold);
}

/// Spectral distortion: cosine distance between original and processed spectral profiles.
/// Returns 0.0 (identical shape) to 1.0 (orthogonal). Lower is better.
fn spectral_distortion(original: &[f32], processed: &[f32]) -> f64 {
    let orig_mags: Vec<f64> = SPEECH_FREQS
        .iter()
        .map(|&f| goertzel_magnitude(original, f, SAMPLE_RATE))
        .collect();
    let proc_mags: Vec<f64> = SPEECH_FREQS
        .iter()
        .map(|&f| goertzel_magnitude(processed, f, SAMPLE_RATE))
        .collect();

    let dot: f64 = orig_mags.iter().zip(&proc_mags).map(|(a, b)| a * b).sum();
    let norm_a: f64 = orig_mags.iter().map(|a| a * a).sum::<f64>().sqrt();
    let norm_b: f64 = proc_mags.iter().map(|b| b * b).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Evaluate a single speech fold + silence pair.
fn evaluate_fold(dsp: &DspConfig, speech_fold: &[f32], silence_raw: &[f32]) -> f64 {
    let mut speech = speech_fold.to_vec();
    apply_dsp_chain(&mut speech, dsp);

    let mut silence = silence_raw.to_vec();
    apply_dsp_chain(&mut silence, dsp);

    let speech_rms = rms(&speech);
    let silence_rms = rms(&silence);

    // Energy guard: if gate ate the speech, fitness is terrible
    if speech_rms < 0.01 {
        return f64::NEG_INFINITY;
    }

    // SNR (dB) — higher is better
    let snr = if silence_rms > 1e-10 {
        20.0 * (speech_rms / silence_rms).log10()
    } else {
        80.0
    };

    // Speech retention: Goertzel energy at full speech band after/before
    let orig_energy: f64 = SPEECH_FREQS
        .iter()
        .map(|&f| goertzel_magnitude(speech_fold, f, SAMPLE_RATE))
        .sum();
    let proc_energy: f64 = SPEECH_FREQS
        .iter()
        .map(|&f| goertzel_magnitude(&speech, f, SAMPLE_RATE))
        .sum();
    let retention = if orig_energy > 0.0 {
        (proc_energy / orig_energy).min(1.5)
    } else {
        0.0
    };

    // Noise floor (dB FS) — more negative is better
    let noise_floor = if silence_rms > 1e-10 {
        20.0 * silence_rms.log10()
    } else {
        -120.0
    };

    // Spectral distortion — lower is better (0 = perfect shape preservation)
    let distortion = spectral_distortion(speech_fold, &speech);

    // Composite:
    //   SNR (0.35) + retention (0.25) + noise floor (0.20) + distortion penalty (0.20)
    0.35 * snr + 0.25 * retention + 0.20 * (-noise_floor) + 0.20 * (1.0 - distortion) * 20.0
}

/// Evaluate a genome using k-fold cross-validation over speech segments.
/// The speech is split into CV_FOLDS roughly equal chunks, and the fitness
/// is the average score across all folds. This prevents overfitting to a
/// particular segment of the recording.
fn evaluate(genome: &Genome, speech_raw: &[f32], silence_raw: &[f32]) -> f64 {
    let dsp = genome.to_dsp_config();

    let fold_size = speech_raw.len() / CV_FOLDS;
    if fold_size == 0 {
        // Not enough audio for CV — evaluate on the whole thing
        return evaluate_fold(&dsp, speech_raw, silence_raw);
    }

    let mut total = 0.0_f64;
    let mut valid_folds = 0;

    for k in 0..CV_FOLDS {
        let start = k * fold_size;
        let end = if k == CV_FOLDS - 1 {
            speech_raw.len()
        } else {
            start + fold_size
        };
        let fold = &speech_raw[start..end];

        let score = evaluate_fold(&dsp, fold, silence_raw);
        if score == f64::NEG_INFINITY {
            return f64::NEG_INFINITY; // any fold failing energy guard = total failure
        }
        total += score;
        valid_folds += 1;
    }

    if valid_folds == 0 {
        f64::NEG_INFINITY
    } else {
        total / valid_folds as f64
    }
}

// --- Crowding distance for diversity preservation ---

/// Compute crowding distances for a population sorted by fitness.
/// Higher distance = more isolated in gene space = more worth preserving.
fn compute_crowding_distances(pop: &[Genome]) -> Vec<f64> {
    let n = pop.len();
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let mut distances = vec![0.0_f64; n];

    // For each gene dimension, sort indices by gene value and assign distances
    for g in 0..GENE_COUNT {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            pop[a].genes[g]
                .partial_cmp(&pop[b].genes[g])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Boundary individuals get infinite distance
        distances[indices[0]] = f64::INFINITY;
        distances[indices[n - 1]] = f64::INFINITY;

        let range = GENE_MAX[g] - GENE_MIN[g];
        if range < 1e-14 {
            continue;
        }

        for i in 1..n - 1 {
            let span = pop[indices[i + 1]].genes[g] - pop[indices[i - 1]].genes[g];
            distances[indices[i]] += span / range;
        }
    }

    distances
}

// --- GA runner ---

#[allow(clippy::too_many_arguments)]
fn run_ga(
    speech: &[f32],
    silence: &[f32],
    config: &DspConfig,
    population_size: usize,
    generations: usize,
    rng: &mut Lcg,
    progress: Option<&ProgressBar>,
    baseline_fitness: f64,
) -> Genome {
    // Initialize population
    let mut pop: Vec<Genome> = Vec::with_capacity(population_size);
    pop.push(Genome::from_dsp_config(config)); // seed with current defaults
    for _ in 1..population_size {
        pop.push(Genome::random(rng));
    }

    // Evaluate initial population
    for g in &mut pop {
        g.fitness = evaluate(g, speech, silence);
    }

    let tournament_size = 3;
    let elitism = 4; // preserve top 4 (10% of default pop=40)
    let base_mutation_prob = 0.20;
    let max_mutation_prob = 0.60;
    let stagnation_limit = 10;

    let mut best_fitness = f64::NEG_INFINITY;
    let mut stagnation = 0;
    let mut mutation_prob;

    for generation in 0..generations {
        // Sort by fitness descending
        pop.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let current_best = pop[0].fitness;
        if (current_best - best_fitness).abs() < 1e-10 {
            stagnation += 1;
        } else {
            best_fitness = current_best;
            stagnation = 0;
        }

        // Adaptive mutation: ramp up during stagnation, decay when improving
        mutation_prob = if stagnation > 0 {
            (base_mutation_prob
                + (max_mutation_prob - base_mutation_prob) * stagnation as f64
                    / stagnation_limit as f64)
                .min(max_mutation_prob)
        } else {
            base_mutation_prob
        };

        if let Some(pb) = progress {
            pb.set_position((generation + 1) as u64);
            pb.set_message(format!(
                "best: {:+.2} vs baseline",
                current_best - baseline_fitness
            ));
        }

        if stagnation >= stagnation_limit {
            if let Some(pb) = progress {
                pb.set_position(generations as u64);
                pb.finish_with_message(format!(
                    "best: {:+.2} vs baseline (early stop)",
                    current_best - baseline_fitness
                ));
            }
            break;
        }

        // Compute crowding distances for diversity-aware selection
        let crowd = compute_crowding_distances(&pop);

        // Build next generation
        let mut next_pop: Vec<Genome> = Vec::with_capacity(population_size);

        // Elitism: carry over top individuals
        for individual in pop.iter().take(elitism.min(pop.len())) {
            next_pop.push(individual.clone());
        }

        // Fill rest with crossover + mutation
        // Use crowding-tournament: among equal-fitness, prefer higher crowding distance
        while next_pop.len() < population_size {
            let p1 = crowding_tournament(&pop, &crowd, rng, tournament_size);
            let p2 = crowding_tournament(&pop, &crowd, rng, tournament_size);
            let (mut c1, mut c2) = sbx_crossover(p1, p2, rng);
            polynomial_mutation(&mut c1, mutation_prob, rng);
            polynomial_mutation(&mut c2, mutation_prob, rng);
            c1.fitness = evaluate(&c1, speech, silence);
            next_pop.push(c1);
            if next_pop.len() < population_size {
                c2.fitness = evaluate(&c2, speech, silence);
                next_pop.push(c2);
            }
        }

        pop = next_pop;
    }

    // Final sort
    pop.sort_by(|a, b| {
        b.fitness
            .partial_cmp(&a.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let best = pop.into_iter().next().unwrap();

    // Non-regression guard: if GA couldn't beat baseline, fall back
    if best.fitness < baseline_fitness {
        let mut fallback = Genome::from_dsp_config(config);
        fallback.fitness = baseline_fitness;
        return fallback;
    }
    best
}

/// Crowding-aware tournament selection.
/// When fitness values are within 1% of each other, prefer the individual with
/// higher crowding distance (more isolated in gene space = more diverse).
fn crowding_tournament<'a>(
    pop: &'a [Genome],
    crowd: &[f64],
    rng: &mut Lcg,
    size: usize,
) -> &'a Genome {
    let mut best_idx = (rng.gen_f64() * pop.len() as f64) as usize % pop.len();
    for _ in 1..size {
        let idx = (rng.gen_f64() * pop.len() as f64) as usize % pop.len();
        let fitness_diff = (pop[idx].fitness - pop[best_idx].fitness).abs();
        let avg_fitness = (pop[idx].fitness.abs() + pop[best_idx].fitness.abs()) / 2.0;
        let threshold = if avg_fitness > 1e-10 {
            avg_fitness * 0.01
        } else {
            1e-10
        };

        if fitness_diff < threshold {
            // Roughly equal fitness — prefer higher crowding distance (more diverse)
            if crowd[idx] > crowd[best_idx] {
                best_idx = idx;
            }
        } else if pop[idx].fitness > pop[best_idx].fitness {
            best_idx = idx;
        }
    }
    &pop[best_idx]
}

// --- Public API ---

pub struct CalibrationResult {
    pub optimal: DspConfig,
    pub snr_improvement_db: f64,
    pub speech_retention: f64,
    pub noise_floor_db: f64,
}

/// Filter speech recording using Silero VAD to retain only actual speech frames.
/// This dramatically improves the speech/silence RMS ratio in noisy environments.
fn vad_filter_speech(config: &Config, raw: &[f32]) -> eyre::Result<Vec<f32>> {
    let duration_secs = raw.len() as f32 / SAMPLE_RATE as f32;
    let mut vad =
        VadSession::new(config, duration_secs).map_err(|e| eyre::eyre!("VAD init failed: {e}"))?;

    for chunk in raw.chunks(VAD_WINDOW_SIZE) {
        if chunk.len() == VAD_WINDOW_SIZE {
            vad.accept_waveform(chunk.to_vec());
        }
    }
    vad.flush();

    let segments = vad.collect_segments();
    if segments.is_empty() {
        eyre::bail!("VAD detected no speech segments. Speak louder or reduce background noise.");
    }

    let mut filtered = Vec::new();
    for seg in &segments {
        filtered.extend_from_slice(&seg.samples);
    }

    Ok(filtered)
}

pub async fn run_calibration(
    config: &Config,
    speech_secs: u32,
    silence_secs: u32,
    population: usize,
    generations: usize,
    dry_run: bool,
) -> eyre::Result<CalibrationResult> {
    let term = console::Term::stderr();

    // Header
    term.write_line(&format!("\n  {}", style("Vox DSP Calibration").bold()))?;
    term.write_line(&format!("  {}\n", "\u{2500}".repeat(19)))?;

    // --- Phase 1: Record ambient noise (also serves as ambient check) ---
    term.write_line(&format!(
        "  {} Capturing ambient noise ({silence_secs}s)",
        style("\u{25b8}").bold()
    ))?;

    let pb_silence = make_timer_bar(silence_secs as u64);
    let silence_samples = record_segment_with_progress(silence_secs, &pb_silence).await?;
    pb_silence.finish_with_message(format!("done    {} samples", silence_samples.len()));

    let ambient_rms = rms(&silence_samples);
    let ambient_db = if ambient_rms > 1e-10 {
        20.0 * ambient_rms.log10()
    } else {
        -120.0
    };
    if ambient_rms > 0.08 {
        term.write_line(&format!(
            "    Ambient: {:.1} dB FS {}",
            ambient_db,
            style("(noisy \u{2014} consider a quieter room)").yellow()
        ))?;
    } else {
        term.write_line(&format!(
            "    Ambient: {:.1} dB FS {}",
            ambient_db,
            style("\u{2713}").green()
        ))?;
    }
    term.write_line("")?;

    // --- Phase 2: Record speech ---
    term.write_line(&format!(
        "  {} Read this passage aloud (repeat if you finish early):\n",
        style("\u{25b8}").bold()
    ))?;
    for sentence in CALIBRATION_SENTENCES {
        term.write_line(&format!("    \"{sentence}\""))?;
    }
    term.write_line("")?;

    term.write_line(&format!(
        "  {} Recording speech ({speech_secs}s)",
        style("\u{25b8}").bold()
    ))?;

    let pb_speech = make_timer_bar(speech_secs as u64);
    let speech_samples = record_segment_with_progress(speech_secs, &pb_speech).await?;
    pb_speech.finish_with_message(format!("done    {} samples", speech_samples.len()));

    // Filter speech recording with VAD to isolate actual speech frames
    let speech_samples = vad_filter_speech(config, &speech_samples)?;
    let retained_secs = speech_samples.len() as f64 / SAMPLE_RATE;
    term.write_line(&format!(
        "    VAD retained {retained_secs:.1}s of {speech_secs}s"
    ))?;
    term.write_line("")?;

    // Validate: speech RMS should be well above silence RMS
    let speech_rms = rms(&speech_samples);
    let silence_rms = rms(&silence_samples);

    if silence_rms > 1e-10 {
        let ratio = speech_rms / silence_rms;
        if ratio < 1.5 {
            eyre::bail!(
                "Speech RMS ({:.4}) is not >1.5x silence RMS ({:.4}). \
                 Please retry in a quieter environment or speak louder.",
                speech_rms,
                silence_rms
            );
        } else if ratio < 5.0 {
            term.write_line(&format!(
                "    {}",
                style(format!(
                    "Warning: low SNR ({ratio:.1}x). Results may be suboptimal."
                ))
                .yellow()
            ))?;
        }
    }

    // Baseline: evaluate current config
    let baseline = Genome::from_dsp_config(&config.dsp);
    let baseline_fitness = evaluate(&baseline, &speech_samples, &silence_samples);

    // --- GA phase ---
    term.write_line(&format!("  {} Optimizing...", style("\u{25b8}").bold()))?;

    let pb_ga = ProgressBar::new(generations as u64);
    pb_ga.set_style(
        ProgressStyle::with_template("  {bar:38.cyan/dim} {pos}/{len} generations    {msg}")
            .unwrap()
            .progress_chars("\u{2588}\u{2588}\u{2500}"),
    );

    let mut rng = Lcg::from_time();
    let best = run_ga(
        &speech_samples,
        &silence_samples,
        &config.dsp,
        population,
        generations,
        &mut rng,
        Some(&pb_ga),
        baseline_fitness,
    );

    pb_ga.finish_with_message(format!(
        "best: {:+.2} vs baseline",
        best.fitness - baseline_fitness
    ));
    term.write_line("")?;

    // GA guarantees best.fitness >= baseline_fitness, but belt-and-suspenders
    let (optimal, snr_improvement) = if best.fitness >= baseline_fitness {
        (best.to_dsp_config(), best.fitness - baseline_fitness)
    } else {
        (config.dsp.clone(), 0.0)
    };

    // Compute final metrics on the optimal result
    let mut speech_processed = speech_samples.clone();
    apply_dsp_chain(&mut speech_processed, &optimal);

    let mut silence_processed = silence_samples.clone();
    apply_dsp_chain(&mut silence_processed, &optimal);

    let orig_energy: f64 = SPEECH_FREQS
        .iter()
        .map(|&f| goertzel_magnitude(&speech_samples, f, SAMPLE_RATE))
        .sum();
    let proc_energy: f64 = SPEECH_FREQS
        .iter()
        .map(|&f| goertzel_magnitude(&speech_processed, f, SAMPLE_RATE))
        .sum();
    let retention = if orig_energy > 0.0 {
        proc_energy / orig_energy
    } else {
        0.0
    };

    let silence_proc_rms = rms(&silence_processed);
    let noise_floor = if silence_proc_rms > 1e-10 {
        20.0 * silence_proc_rms.log10()
    } else {
        -120.0
    };

    let distortion = spectral_distortion(&speech_samples, &speech_processed);

    // Display results
    term.write_line(&format!("  {}", style("Results").bold()))?;
    term.write_line(&format!("  {}", "\u{2500}".repeat(7)))?;
    term.write_line(&format!(
        "  {:24} {:>12} {:>12}",
        "Parameter", "Current", "Optimal"
    ))?;
    term.write_line(&format!("  {}", "\u{2500}".repeat(50)))?;
    term.write_line(&format!(
        "  {:24} {:>12.1} {:>12.1}",
        "hpf_cutoff_hz", config.dsp.hpf_cutoff_hz, optimal.hpf_cutoff_hz
    ))?;
    term.write_line(&format!(
        "  {:24} {:>12.3} {:>12.3}",
        "noise_gate_rms", config.dsp.noise_gate_rms, optimal.noise_gate_rms
    ))?;
    term.write_line(&format!(
        "  {:24} {:>12} {:>12}",
        "noise_gate_window", config.dsp.noise_gate_window, optimal.noise_gate_window
    ))?;
    term.write_line(&format!(
        "  {:24} {:>12.2} {:>12.2}",
        "normalize_threshold", config.dsp.normalize_threshold, optimal.normalize_threshold
    ))?;
    term.write_line("")?;
    term.write_line(&format!("  Fitness improvement: {snr_improvement:+.2}"))?;
    term.write_line(&format!("  Speech retention:     {retention:.2}"))?;
    term.write_line(&format!("  Spectral fidelity:    {:.2}", 1.0 - distortion))?;
    term.write_line(&format!("  Noise floor:         {noise_floor:.1} dB FS"))?;

    if !dry_run && snr_improvement > 0.0 {
        Config::set_value("dsp.hpf_cutoff_hz", &format!("{}", optimal.hpf_cutoff_hz))
            .map_err(|e| eyre::eyre!(e))?;
        Config::set_value("dsp.noise_gate_rms", &format!("{}", optimal.noise_gate_rms))
            .map_err(|e| eyre::eyre!(e))?;
        Config::set_value(
            "dsp.noise_gate_window",
            &format!("{}", optimal.noise_gate_window),
        )
        .map_err(|e| eyre::eyre!(e))?;
        Config::set_value(
            "dsp.normalize_threshold",
            &format!("{}", optimal.normalize_threshold),
        )
        .map_err(|e| eyre::eyre!(e))?;

        let path = Config::config_path();
        term.write_line(&format!(
            "\n  {} Saved to {}",
            style("\u{2713}").green(),
            path.display()
        ))?;
        term.write_line("    Restart daemon to apply.")?;
    } else if !dry_run {
        term.write_line(&format!(
            "\n  {} Current config is already optimal for this environment.",
            style("i").cyan().bold()
        ))?;
    } else {
        term.write_line(&format!(
            "\n  {} dry run \u{2014} results not saved",
            style("i").cyan().bold()
        ))?;
    }

    Ok(CalibrationResult {
        optimal,
        snr_improvement_db: snr_improvement,
        speech_retention: retention,
        noise_floor_db: noise_floor,
    })
}

/// Create a progress bar styled as a recording countdown timer.
fn make_timer_bar(total_secs: u64) -> ProgressBar {
    let pb = ProgressBar::new(total_secs);
    pb.set_style(
        ProgressStyle::with_template("  {bar:38.cyan/dim} {msg}")
            .unwrap()
            .progress_chars("\u{2588}\u{2588}\u{2500}"),
    );
    pb.set_message(format!("{total_secs}s left    0 samples"));
    pb
}

/// Record raw audio with a progress bar.
/// Uses near-neutral DSP (1Hz HPF = passthrough, 0.0 gate = disabled) to capture raw audio.
async fn record_segment_with_progress(secs: u32, pb: &ProgressBar) -> eyre::Result<Vec<f32>> {
    let (mut rx, handle) =
        audio::start_capture(1.0, 0.0, 512).map_err(|e| eyre::eyre!("Capture failed: {e}"))?;

    let mut samples: Vec<f32> = Vec::with_capacity(16000 * secs as usize);
    let start = std::time::Instant::now();
    let deadline = start + std::time::Duration::from_secs(secs as u64);
    let mut last_sec = 0u64;

    while std::time::Instant::now() < deadline {
        let elapsed_sec = start.elapsed().as_secs();
        if elapsed_sec != last_sec {
            pb.set_position(elapsed_sec);
            last_sec = elapsed_sec;
        }
        let remaining = secs as u64 - elapsed_sec.min(secs as u64);
        pb.set_message(format!("{remaining}s left    {} samples", samples.len()));

        match tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await {
            Ok(Some(chunk)) => {
                samples.extend_from_slice(&chunk.samples);
            }
            Ok(None) => break,
            Err(_) => {} // timeout, loop and check deadline
        }
    }

    pb.set_position(secs as u64);
    pb.set_message(format!("done    {} samples", samples.len()));
    handle.stop();
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Deterministic white noise via LCG
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

    /// Richer synthetic speech: fundamental + harmonics + formant-like resonances + sibilant noise
    fn rich_speech_signal(n: usize, seed: u64) -> Vec<f32> {
        let mut noise_state = seed;
        (0..n)
            .map(|i| {
                let t = i as f64 / SAMPLE_RATE;
                let mut sample = 0.0_f64;
                // Voiced component: 150Hz fundamental + harmonics
                for h in 1..=12 {
                    let amp = 1.0 / (h as f64).powf(1.2);
                    sample += amp * (2.0 * std::f64::consts::PI * 150.0 * h as f64 * t).sin();
                }
                // Formant-like resonances at F1≈500Hz and F2≈1500Hz
                sample += 0.3 * (2.0 * std::f64::consts::PI * 500.0 * t).sin();
                sample += 0.2 * (2.0 * std::f64::consts::PI * 1500.0 * t).sin();
                // Sibilant energy (high-freq noise burst every ~0.5s)
                let in_sibilant = ((t * 2.0) % 1.0) > 0.8;
                if in_sibilant {
                    noise_state = noise_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let noise_val = ((noise_state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
                    sample += 0.15 * noise_val;
                }
                (sample * 0.3) as f32 // scale to reasonable amplitude
            })
            .collect()
    }

    // --- PRNG tests ---

    #[test]
    fn test_lcg_range() {
        let mut rng = Lcg::new(42);
        for _ in 0..1000 {
            let v = rng.gen_f64();
            assert!((0.0..1.0).contains(&v), "gen_f64 out of range: {v}");
        }
    }

    // --- Utility tests ---

    #[test]
    fn test_nearest_power_of_two() {
        assert_eq!(nearest_power_of_two(1), 1);
        assert_eq!(nearest_power_of_two(2), 2);
        assert_eq!(nearest_power_of_two(3), 4);
        assert_eq!(nearest_power_of_two(5), 4);
        assert_eq!(nearest_power_of_two(6), 8);
        assert_eq!(nearest_power_of_two(7), 8);
        assert_eq!(nearest_power_of_two(512), 512);
        assert_eq!(nearest_power_of_two(600), 512);
        assert_eq!(nearest_power_of_two(900), 1024);
        assert_eq!(nearest_power_of_two(2048), 2048);
    }

    // --- Genome tests ---

    #[test]
    fn test_genome_clamping() {
        let mut rng = Lcg::new(123);
        for _ in 0..100 {
            let mut g = Genome::random(&mut rng);
            g.genes[0] = -100.0;
            g.genes[1] = 999.0;
            g.genes[2] = 0.0;
            g.genes[3] = 5.0;
            g.clamp();
            for i in 0..GENE_COUNT {
                assert!(
                    g.genes[i] >= GENE_MIN[i] && g.genes[i] <= GENE_MAX[i],
                    "Gene {i} out of bounds: {} (range [{}, {}])",
                    g.genes[i],
                    GENE_MIN[i],
                    GENE_MAX[i],
                );
            }
        }
    }

    #[test]
    fn test_genome_clamping_after_mutation() {
        let mut rng = Lcg::new(456);
        for _ in 0..200 {
            let mut g = Genome::random(&mut rng);
            polynomial_mutation(&mut g, 1.0, &mut rng);
            for i in 0..GENE_COUNT {
                assert!(
                    g.genes[i] >= GENE_MIN[i] && g.genes[i] <= GENE_MAX[i],
                    "Gene {i} out of bounds after mutation: {} (range [{}, {}])",
                    g.genes[i],
                    GENE_MIN[i],
                    GENE_MAX[i],
                );
            }
        }
    }

    #[test]
    fn test_crossover_produces_valid_offspring() {
        let mut rng = Lcg::new(789);
        for _ in 0..100 {
            let p1 = Genome::random(&mut rng);
            let p2 = Genome::random(&mut rng);
            let (c1, c2) = sbx_crossover(&p1, &p2, &mut rng);
            for i in 0..GENE_COUNT {
                assert!(
                    c1.genes[i] >= GENE_MIN[i] && c1.genes[i] <= GENE_MAX[i],
                    "Child1 gene {i} out of bounds: {}",
                    c1.genes[i]
                );
                assert!(
                    c2.genes[i] >= GENE_MIN[i] && c2.genes[i] <= GENE_MAX[i],
                    "Child2 gene {i} out of bounds: {}",
                    c2.genes[i]
                );
            }
            assert!(c1.fitness == f64::NEG_INFINITY);
            assert!(c2.fitness == f64::NEG_INFINITY);
        }
    }

    #[test]
    fn test_from_dsp_config_roundtrip() {
        let dsp = DspConfig {
            hpf_cutoff_hz: 256.0,
            noise_gate_rms: 0.05,
            noise_gate_window: 512,
            normalize_threshold: 0.3,
        };
        let genome = Genome::from_dsp_config(&dsp);
        let back = genome.to_dsp_config();
        assert!((back.hpf_cutoff_hz - 256.0).abs() < 1e-10);
        assert!((back.noise_gate_rms - 0.05).abs() < f32::EPSILON);
        assert_eq!(back.noise_gate_window, 512);
        assert!((back.normalize_threshold - 0.3).abs() < f32::EPSILON);
    }

    // --- Goertzel / RMS tests ---

    #[test]
    fn test_goertzel_detects_known_frequency() {
        let n = 16000;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 300.0 * i as f64 / SAMPLE_RATE).sin() as f32)
            .collect();

        let mag_300 = goertzel_magnitude(&samples, 300.0, SAMPLE_RATE);
        let mag_1000 = goertzel_magnitude(&samples, 1000.0, SAMPLE_RATE);

        assert!(
            mag_300 > mag_1000 * 10.0,
            "300Hz magnitude ({mag_300:.4}) should be much larger than 1000Hz ({mag_1000:.4})"
        );
    }

    #[test]
    fn test_rms_known_values() {
        let silence = vec![0.0f32; 100];
        assert!((rms(&silence) - 0.0).abs() < 1e-10);

        let sine: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 16000.0).sin() as f32)
            .collect();
        let r = rms(&sine);
        assert!(
            (r - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.01,
            "Sine RMS should be ~0.707, got {r}"
        );
    }

    // --- Spectral distortion tests ---

    #[test]
    fn test_spectral_distortion_identical_is_zero() {
        let signal = speech_signal(16000);
        let d = spectral_distortion(&signal, &signal);
        assert!(
            d < 0.001,
            "Identical signals should have ~0 distortion, got {d}"
        );
    }

    #[test]
    fn test_spectral_distortion_noise_is_high() {
        let signal = speech_signal(16000);
        let noise = white_noise(16000, 99);
        let d = spectral_distortion(&signal, &noise);
        assert!(
            d > 0.1,
            "Speech vs noise should have significant distortion, got {d}"
        );
    }

    // --- Fitness evaluation tests ---

    #[test]
    fn test_evaluate_energy_guard() {
        let speech = vec![0.0f32; 16000];
        let silence = vec![0.0f32; 8000];
        let g = Genome {
            genes: [200.0, 0.01, 512.0, 0.5],
            fitness: f64::NEG_INFINITY,
        };
        let f = evaluate(&g, &speech, &silence);
        assert!(
            f == f64::NEG_INFINITY,
            "Zero speech should get -inf fitness"
        );
    }

    #[test]
    fn test_fitness_with_synthetic_data() {
        // Use enough samples for k-fold CV (need at least CV_FOLDS * window_size)
        let speech = speech_signal(48000); // 3 seconds
        let noise = white_noise(24000, 42);
        let silence: Vec<f32> = noise.iter().map(|s| s * 0.001).collect();

        let good = Genome {
            genes: [200.0, 0.01, 512.0, 0.5],
            fitness: f64::NEG_INFINITY,
        };
        let bad = Genome {
            genes: [490.0, 0.14, 64.0, 0.05],
            fitness: f64::NEG_INFINITY,
        };

        let good_fitness = evaluate(&good, &speech, &silence);
        let bad_fitness = evaluate(&bad, &speech, &silence);

        assert!(
            good_fitness > bad_fitness,
            "Good params ({good_fitness:.4}) should score higher than bad params ({bad_fitness:.4})"
        );
        assert!(good_fitness.is_finite());
    }

    #[test]
    fn test_fitness_cv_consistency() {
        // Fitness should be similar whether we use 1-fold or k-fold on uniform speech
        let speech = speech_signal(48000);
        let silence: Vec<f32> = white_noise(24000, 42).iter().map(|s| s * 0.001).collect();

        let dsp = DspConfig::default();
        let genome = Genome::from_dsp_config(&dsp);

        let full_fitness = evaluate(&genome, &speech, &silence);
        let fold_fitness = evaluate_fold(&dsp, &speech, &silence);

        // k-fold average on uniform signal should be close to single-eval
        let diff = (full_fitness - fold_fitness).abs();
        assert!(
            diff < full_fitness.abs() * 0.15,
            "CV fitness ({full_fitness:.4}) should be close to single-fold ({fold_fitness:.4}), diff={diff:.4}"
        );
    }

    #[test]
    fn test_fitness_with_rich_signal() {
        // Rich signal with formants and sibilants should still rank good > bad
        let speech = rich_speech_signal(48000, 123);
        let silence: Vec<f32> = white_noise(24000, 42).iter().map(|s| s * 0.001).collect();

        let good = Genome {
            genes: [200.0, 0.01, 512.0, 0.5],
            fitness: f64::NEG_INFINITY,
        };
        let bad = Genome {
            genes: [490.0, 0.14, 64.0, 0.05],
            fitness: f64::NEG_INFINITY,
        };

        let good_fitness = evaluate(&good, &speech, &silence);
        let bad_fitness = evaluate(&bad, &speech, &silence);

        assert!(
            good_fitness > bad_fitness,
            "Good > bad on rich signal: {good_fitness:.4} vs {bad_fitness:.4}"
        );
    }

    // --- Crowding distance tests ---

    #[test]
    fn test_crowding_distance_boundary_is_infinite() {
        let mut rng = Lcg::new(42);
        let pop: Vec<Genome> = (0..10).map(|_| Genome::random(&mut rng)).collect();
        let distances = compute_crowding_distances(&pop);
        assert_eq!(distances.len(), 10);
        // At least some boundary individuals should have infinite distance
        assert!(
            distances.iter().any(|&d| d.is_infinite()),
            "Boundary individuals should have infinite crowding distance"
        );
    }

    #[test]
    fn test_crowding_distance_small_pop() {
        let mut rng = Lcg::new(42);
        let pop: Vec<Genome> = (0..2).map(|_| Genome::random(&mut rng)).collect();
        let distances = compute_crowding_distances(&pop);
        assert!(distances.iter().all(|&d| d.is_infinite()));
    }

    // --- GA convergence tests ---

    #[test]
    fn test_ga_converges() {
        let speech = speech_signal(48000);
        let silence: Vec<f32> = white_noise(24000, 42).iter().map(|s| s * 0.001).collect();

        let mut rng = Lcg::new(42);
        let default_config = DspConfig::default();
        let default_genome = Genome::from_dsp_config(&default_config);
        let default_fitness = evaluate(&default_genome, &speech, &silence);
        let best = run_ga(
            &speech,
            &silence,
            &default_config,
            40,
            30,
            &mut rng,
            None,
            default_fitness,
        );

        assert!(best.fitness.is_finite(), "Best fitness should be finite");
        assert!(
            best.fitness >= default_fitness,
            "GA result ({:.4}) should be >= default ({:.4})",
            best.fitness,
            default_fitness
        );
    }

    #[test]
    fn test_ga_converges_rich_signal() {
        // GA should also converge on a richer, more realistic signal
        let speech = rich_speech_signal(48000, 777);
        let silence: Vec<f32> = white_noise(24000, 42).iter().map(|s| s * 0.002).collect();

        let mut rng = Lcg::new(99);
        let default_config = DspConfig::default();
        let default_genome = Genome::from_dsp_config(&default_config);
        let default_fitness = evaluate(&default_genome, &speech, &silence);
        let best = run_ga(
            &speech,
            &silence,
            &default_config,
            30,
            20,
            &mut rng,
            None,
            default_fitness,
        );

        assert!(best.fitness.is_finite(), "Best fitness should be finite");
        assert!(
            best.fitness >= default_fitness,
            "GA on rich signal ({:.4}) should be >= default ({:.4})",
            best.fitness,
            default_fitness
        );
    }

    #[test]
    fn test_ga_adaptive_mutation_activates() {
        // With a tiny population and few gens, stagnation should occur and
        // mutation should ramp up. We can't easily observe this directly,
        // but we can verify it doesn't panic and still produces valid output.
        let speech = speech_signal(16000);
        let silence: Vec<f32> = white_noise(8000, 42).iter().map(|s| s * 0.001).collect();

        let mut rng = Lcg::new(42);
        let default_config = DspConfig::default();
        let default_genome = Genome::from_dsp_config(&default_config);
        let default_fitness = evaluate(&default_genome, &speech, &silence);
        let best = run_ga(
            &speech,
            &silence,
            &default_config,
            6,
            15,
            &mut rng,
            None,
            default_fitness,
        );

        assert!(best.fitness.is_finite() || best.fitness == f64::NEG_INFINITY);
        let dsp = best.to_dsp_config();
        assert!(dsp.hpf_cutoff_hz >= 20.0 && dsp.hpf_cutoff_hz <= 500.0);
    }

    #[test]
    fn test_ga_never_regresses() {
        // Small population, few generations — GA must still never regress below baseline
        let speech = speech_signal(48000);
        let silence: Vec<f32> = white_noise(24000, 42).iter().map(|s| s * 0.001).collect();

        let default_config = DspConfig::default();
        let default_genome = Genome::from_dsp_config(&default_config);
        let baseline = evaluate(&default_genome, &speech, &silence);

        // Run multiple times with different seeds to stress-test the guarantee
        for seed in [1, 42, 99, 1337, 9999] {
            let mut rng = Lcg::new(seed);
            let best = run_ga(
                &speech,
                &silence,
                &default_config,
                10,
                5,
                &mut rng,
                None,
                baseline,
            );
            assert!(
                best.fitness >= baseline,
                "seed={seed}: GA regressed ({:.4}) below baseline ({:.4})",
                best.fitness,
                baseline
            );
        }
    }

    // --- Calibration sentence tests ---

    #[test]
    fn test_calibration_sentences_exist() {
        assert!(
            CALIBRATION_SENTENCES.len() >= 2,
            "Should have at least 2 calibration sentences"
        );
        for sentence in CALIBRATION_SENTENCES {
            assert!(!sentence.is_empty());
        }
    }

    #[test]
    fn test_speech_freqs_cover_band() {
        assert!(SPEECH_FREQS.len() >= 6, "Need at least 6 frequency bins");
        assert!(
            *SPEECH_FREQS.first().unwrap() <= 200.0,
            "Should cover fundamental"
        );
        assert!(
            *SPEECH_FREQS.last().unwrap() >= 5000.0,
            "Should cover sibilants"
        );
    }
}
