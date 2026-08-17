// Copyright: lituus-io, all rights reserved.
// Author: terekete <spicyzhug@gmail.com>
//
//! Shared bench telemetry helpers: a deterministic PRNG, RSS sampling, the
//! `AUTORESEARCH` line emitter, and a least-squares slope for the memory-expansion
//! (atomizer) regression heuristic. Included from each bench via
//! `#[path = "support/mod.rs"] mod support;`.

/// Resident set size in KiB. Linux reads `/proc/self/status`; macOS shells out to
/// `ps` (which reports KiB); other platforms return 0.
#[must_use]
pub fn rss_kib() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    return rest
                        .split_whitespace()
                        .next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                }
            }
        }
        0
    }
    #[cfg(target_os = "macos")]
    {
        let pid = std::process::id().to_string();
        std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

/// Deterministic xorshift PRNG (no `Math.random` — reproducible across runs).
pub struct Rng(pub u64);

impl Rng {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A float in `[-1, 1)`.
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// Emit one `AUTORESEARCH` telemetry line in the house format.
pub fn autoresearch(bench: &str, name: &str, scale: usize, metric: &str, value: f64, rss_before: u64) {
    let rss = rss_kib();
    println!(
        "AUTORESEARCH {bench} bench={name} scale={scale} {metric}={value:.2} process_rss_kib={rss} rss_delta_kib={}",
        rss.saturating_sub(rss_before)
    );
}

/// Least-squares slope of `(x, y)` points (0 for fewer than two points).
#[must_use]
pub fn least_squares_slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let num: f64 = points.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
    let den: f64 = points.iter().map(|p| (p.0 - mean_x).powi(2)).sum();
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}
