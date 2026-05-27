//! Lloyd-Max scalar quantizer for the Beta distribution.
//!
//! After orthogonal rotation, each coordinate of a unit vector on S^{d-1}
//! follows Beta((d-1)/2, (d-1)/2) on [-1, 1]. This module computes optimal
//! quantization boundaries and centroids for that distribution.

use statrs::distribution::{Beta, Continuous, ContinuousCDF};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

const CODEBOOK_MAGIC: &[u8; 4] = b"TVCB";
const CODEBOOK_VERSION: u8 = 1;

type CodebookKey = (usize, usize);
type CodebookParts = (Vec<f32>, Vec<f32>);

static CODEBOOK_CACHE: OnceLock<Mutex<HashMap<CodebookKey, CodebookParts>>> = OnceLock::new();

/// Returns (boundaries, centroids) for the given bit width and dimension.
///
/// Crate-internal: trusts `bits ∈ {2,3,4}` and `dim >= 2`. Callers reach it
/// only after those bounds are enforced (`TurboQuantIndex::new` /
/// `from_parts`). Exposing it publicly would let a caller pass `bits` in the
/// ~32..63 range and drive an unbounded `1 << bits` allocation (DoS), or
/// `dim` 0/1 and hit a `Beta::new` panic — see the validated
/// [`from_parts`](crate::TurboQuantIndex::from_parts) boundary instead.
pub(crate) fn codebook(bits: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    lloyd_max(bits, dim, 200, 1e-12)
}

/// Returns a codebook from the process cache, the default local cache, or
/// deterministic computation.
///
/// The on-disk cache is keyed by `(bits, dim)` and only accelerates startup.
/// If the cache is unavailable or contains invalid data, turbovec recomputes
/// the deterministic codebook and tries to repair the cache.
pub fn cached_codebook(bits: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let key = (bits, dim);
    if let Some(parts) = get_cached_codebook(key) {
        return parts;
    }

    let parts = match default_codebook_cache_dir() {
        Some(dir) => {
            load_or_create_codebook(&dir, bits, dim).unwrap_or_else(|_| codebook(bits, dim))
        }
        None => codebook(bits, dim),
    };
    insert_cached_codebook(key, &parts);
    parts
}

/// Returns a codebook from `cache_dir`, creating or repairing the cache file
/// when needed.
pub fn load_or_create_codebook(
    cache_dir: impl AsRef<Path>,
    bits: usize,
    dim: usize,
) -> io::Result<(Vec<f32>, Vec<f32>)> {
    let path = codebook_cache_path(cache_dir, bits, dim)?;
    match load_codebook(&path, bits, dim) {
        Ok(parts) => Ok(parts),
        Err(_) => {
            let parts = codebook(bits, dim);
            save_codebook(&path, bits, dim, &parts.0, &parts.1)?;
            Ok(parts)
        }
    }
}

/// Returns the default local codebook cache directory.
///
/// `TURBOVEC_CODEBOOK_CACHE_DIR` takes precedence. Without it, macOS uses
/// `~/Library/Caches/turbovec/codebooks`; other platforms use
/// `$XDG_CACHE_HOME/turbovec/codebooks` or `~/.cache/turbovec/codebooks`.
pub fn default_codebook_cache_dir() -> Option<PathBuf> {
    if let Some(dir) = non_empty_env_path("TURBOVEC_CODEBOOK_CACHE_DIR") {
        return Some(dir);
    }

    #[cfg(target_os = "macos")]
    {
        return non_empty_env_path("HOME").map(|home| {
            home.join("Library")
                .join("Caches")
                .join("turbovec")
                .join("codebooks")
        });
    }

    #[cfg(not(target_os = "macos"))]
    {
        if let Some(cache_home) = non_empty_env_path("XDG_CACHE_HOME") {
            return Some(cache_home.join("turbovec").join("codebooks"));
        }
        non_empty_env_path("HOME")
            .map(|home| home.join(".cache").join("turbovec").join("codebooks"))
    }
}

/// Returns the cache file path for a `(bits, dim)` codebook under `cache_dir`.
pub fn codebook_cache_path(
    cache_dir: impl AsRef<Path>,
    bits: usize,
    dim: usize,
) -> io::Result<PathBuf> {
    validate_codebook_params(bits, dim)?;
    Ok(cache_dir
        .as_ref()
        .join(format!("codebook-b{bits}-d{dim}-v{CODEBOOK_VERSION}.tvcb")))
}

/// Saves a validated codebook file.
pub fn save_codebook(
    path: impl AsRef<Path>,
    bits: usize,
    dim: usize,
    boundaries: &[f32],
    centroids: &[f32],
) -> io::Result<()> {
    validate_codebook_parts(bits, dim, boundaries, centroids)?;

    let path = path.as_ref();
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = temp_path_for(path);
    let write_result = write_codebook_file(&tmp_path, bits, dim, boundaries, centroids);
    if let Err(err) = write_result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    if let Err(err) = replace_file(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    Ok(())
}

/// Loads a codebook file and verifies it matches the requested `(bits, dim)`.
pub fn load_codebook(
    path: impl AsRef<Path>,
    bits: usize,
    dim: usize,
) -> io::Result<(Vec<f32>, Vec<f32>)> {
    validate_codebook_params(bits, dim)?;

    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != CODEBOOK_MAGIC {
        return Err(invalid_data("not a turbovec codebook file: wrong magic"));
    }

    let mut version = [0u8; 1];
    reader.read_exact(&mut version)?;
    if version[0] != CODEBOOK_VERSION {
        return Err(invalid_data(format!(
            "unsupported codebook format version: {}",
            version[0]
        )));
    }

    let file_bits = read_u8(&mut reader)? as usize;
    let file_dim = read_u32(&mut reader)? as usize;
    if file_bits != bits || file_dim != dim {
        return Err(invalid_data(format!(
            "codebook key mismatch: file has bits={file_bits}, dim={file_dim}; \
             requested bits={bits}, dim={dim}"
        )));
    }

    let n_boundaries = read_u32(&mut reader)? as usize;
    let n_centroids = read_u32(&mut reader)? as usize;
    let (expected_boundaries, expected_centroids) = expected_lengths(bits)?;
    if n_boundaries != expected_boundaries || n_centroids != expected_centroids {
        return Err(invalid_data(format!(
            "invalid codebook lengths: boundaries={n_boundaries}, centroids={n_centroids}; \
             expected boundaries={expected_boundaries}, centroids={expected_centroids}"
        )));
    }

    let boundaries = read_f32_array(&mut reader, n_boundaries)?;
    let centroids = read_f32_array(&mut reader, n_centroids)?;
    validate_codebook_parts(bits, dim, &boundaries, &centroids)?;
    Ok((boundaries, centroids))
}

fn get_cached_codebook(key: CodebookKey) -> Option<CodebookParts> {
    let cache = CODEBOOK_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    cache.lock().ok().and_then(|guard| guard.get(&key).cloned())
}

fn insert_cached_codebook(key: CodebookKey, parts: &CodebookParts) {
    let cache = CODEBOOK_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut guard) = cache.lock() {
        guard.insert(key, parts.clone());
    }
}

fn write_codebook_file(
    path: &Path,
    bits: usize,
    dim: usize,
    boundaries: &[f32],
    centroids: &[f32],
) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(CODEBOOK_MAGIC)?;
    writer.write_all(&[CODEBOOK_VERSION])?;
    writer.write_all(&[bits as u8])?;
    writer.write_all(&(dim as u32).to_le_bytes())?;
    writer.write_all(&(boundaries.len() as u32).to_le_bytes())?;
    writer.write_all(&(centroids.len() as u32).to_le_bytes())?;
    for &value in boundaries {
        writer.write_all(&value.to_le_bytes())?;
    }
    for &value in centroids {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()
}

fn validate_codebook_parts(
    bits: usize,
    dim: usize,
    boundaries: &[f32],
    centroids: &[f32],
) -> io::Result<()> {
    validate_codebook_params(bits, dim)?;
    let (expected_boundaries, expected_centroids) = expected_lengths(bits)?;
    if boundaries.len() != expected_boundaries || centroids.len() != expected_centroids {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "codebook lengths must be boundaries={expected_boundaries}, \
                 centroids={expected_centroids}; got boundaries={}, centroids={}",
                boundaries.len(),
                centroids.len(),
            ),
        ));
    }
    Ok(())
}

fn validate_codebook_params(bits: usize, dim: usize) -> io::Result<()> {
    expected_lengths(bits)?;
    if dim < 2 || dim > u32::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("dim must be in 2..={}, got {dim}", u32::MAX),
        ));
    }
    Ok(())
}

fn expected_lengths(bits: usize) -> io::Result<(usize, usize)> {
    if !(2..=4).contains(&bits) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("bits must be 2, 3, or 4, got {bits}"),
        ));
    }
    let centroids = 1usize << bits;
    Ok((centroids - 1, centroids))
}

fn read_u8<R: Read>(reader: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32_array<R: Read>(reader: &mut R, n: usize) -> io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    reader.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn temp_path_for(path: &Path) -> PathBuf {
    let mut tmp_path = path.to_path_buf();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    tmp_path.set_extension(format!("tmp-{}-{nanos}", std::process::id()));
    tmp_path
}

fn replace_file(from: &Path, to: &Path) -> io::Result<()> {
    match fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
            fs::remove_file(to)?;
            fs::rename(from, to)
        }
        Err(err) => Err(err),
    }
}

fn non_empty_env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_path_depends_on_bits_and_dim() {
        let dir = temp_dir("path");
        let b2_d128 = codebook_cache_path(&dir, 2, 128).unwrap();
        let b4_d128 = codebook_cache_path(&dir, 4, 128).unwrap();
        let b4_d256 = codebook_cache_path(&dir, 4, 256).unwrap();

        assert_ne!(b2_d128, b4_d128);
        assert_ne!(b4_d128, b4_d256);
        assert_eq!(
            b4_d256.file_name().unwrap().to_str().unwrap(),
            "codebook-b4-d256-v1.tvcb"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn save_and_load_round_trip_for_key() {
        let dir = temp_dir("roundtrip");
        let path = codebook_cache_path(&dir, 4, 128).unwrap();
        let expected = codebook(4, 128);

        save_codebook(&path, 4, 128, &expected.0, &expected.1).unwrap();
        let loaded = load_codebook(&path, 4, 128).unwrap();

        assert_eq!(loaded, expected);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn load_rejects_bits_or_dim_mismatch() {
        let dir = temp_dir("mismatch");
        let path = codebook_cache_path(&dir, 4, 128).unwrap();
        let expected = codebook(4, 128);
        save_codebook(&path, 4, 128, &expected.0, &expected.1).unwrap();

        let bits_err = load_codebook(&path, 2, 128).unwrap_err();
        assert_eq!(bits_err.kind(), io::ErrorKind::InvalidData);
        assert!(bits_err.to_string().contains("codebook key mismatch"));

        let dim_err = load_codebook(&path, 4, 256).unwrap_err();
        assert_eq!(dim_err.kind(), io::ErrorKind::InvalidData);
        assert!(dim_err.to_string().contains("codebook key mismatch"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn load_or_create_repairs_corrupt_file() {
        let dir = temp_dir("repair");
        let path = codebook_cache_path(&dir, 4, 128).unwrap();
        fs::create_dir_all(&dir).unwrap();
        fs::write(&path, b"bad cache").unwrap();

        let expected = codebook(4, 128);
        let loaded = load_or_create_codebook(&dir, 4, 128).unwrap();
        let repaired = load_codebook(&path, 4, 128).unwrap();

        assert_eq!(loaded, expected);
        assert_eq!(repaired, expected);

        let _ = fs::remove_dir_all(dir);
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "turbovec-codebook-test-{label}-{}-{nanos}",
            std::process::id()
        ))
    }
}

fn lloyd_max(bits: usize, dim: usize, max_iter: usize, tol: f64) -> (Vec<f32>, Vec<f32>) {
    let a = (dim as f64 - 1.0) / 2.0;
    // Beta(a, a) on [0, 1], shifted to [-1, 1] via loc=-1, scale=2
    let beta = Beta::new(a, a).unwrap();

    let n_levels = 1usize << bits;

    // Initialize centroids within +/- 3 std devs
    let std_dev = (2.0 * a / ((2.0 * a + 1.0) * 4.0 * a)).sqrt(); // std of Beta on [-1,1]
    let spread = 3.0 * std_dev;
    let mut centroids: Vec<f64> = (0..n_levels)
        .map(|i| -spread + 2.0 * spread * i as f64 / (n_levels as f64 - 1.0))
        .collect();

    for _ in 0..max_iter {
        // Boundaries = midpoints between consecutive centroids
        let boundaries: Vec<f64> = (0..n_levels - 1)
            .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
            .collect();

        let mut edges = Vec::with_capacity(n_levels + 1);
        edges.push(-1.0);
        edges.extend_from_slice(&boundaries);
        edges.push(1.0);

        let mut new_centroids = vec![0.0f64; n_levels];

        for i in 0..n_levels {
            let lo = edges[i];
            let hi = edges[i + 1];

            // CDF on [-1, 1]: transform to [0, 1] for Beta
            let cdf_lo = beta.cdf((lo + 1.0) / 2.0);
            let cdf_hi = beta.cdf((hi + 1.0) / 2.0);
            let prob = cdf_hi - cdf_lo;

            if prob < 1e-15 {
                new_centroids[i] = centroids[i];
            } else {
                // Conditional mean = integral(x * pdf(x), lo, hi) / prob
                // where pdf is on [-1, 1]: pdf_shifted(x) = beta.pdf((x+1)/2) / 2
                let mean = adaptive_simpson(
                    |x| {
                        let t = (x + 1.0) / 2.0;
                        x * beta.pdf(t) / 2.0
                    },
                    lo,
                    hi,
                    1e-14,
                    50,
                );
                new_centroids[i] = mean / prob;
            }
        }

        let max_change = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        centroids = new_centroids;

        if max_change < tol {
            break;
        }
    }

    let boundaries: Vec<f32> = (0..n_levels - 1)
        .map(|i| ((centroids[i] + centroids[i + 1]) / 2.0) as f32)
        .collect();
    let centroids_f32: Vec<f32> = centroids.iter().map(|&c| c as f32).collect();

    (boundaries, centroids_f32)
}

/// Adaptive Simpson's rule for numerical integration.
fn adaptive_simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol: f64, max_depth: usize) -> f64 {
    let mid = (a + b) / 2.0;
    let fa = f(a);
    let fb = f(b);
    let fm = f(mid);
    let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    adaptive_simpson_rec(&f, a, b, fa, fb, fm, whole, tol, max_depth)
}

fn adaptive_simpson_rec<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    whole: f64,
    tol: f64,
    depth: usize,
) -> f64 {
    let mid = (a + b) / 2.0;
    let m1 = (a + mid) / 2.0;
    let m2 = (mid + b) / 2.0;
    let fm1 = f(m1);
    let fm2 = f(m2);
    let left = (mid - a) / 6.0 * (fa + 4.0 * fm1 + fm);
    let right = (b - mid) / 6.0 * (fm + 4.0 * fm2 + fb);
    let refined = left + right;

    if depth == 0 || (refined - whole).abs() < 15.0 * tol {
        refined + (refined - whole) / 15.0
    } else {
        adaptive_simpson_rec(f, a, mid, fa, fm, fm1, left, tol / 2.0, depth - 1)
            + adaptive_simpson_rec(f, mid, b, fm, fb, fm2, right, tol / 2.0, depth - 1)
    }
}
