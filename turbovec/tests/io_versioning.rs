//! Format-versioning tests for `.tv` and `.tvim`.
//!
//! Verifies:
//! 1. Round-trip via the public write/load functions works on the current
//!    format for both file types.
//! 2. A hand-constructed version-1 `.tv` file (bare bit_width-first
//!    header, no magic) is rejected with the upgrade-hint error.
//! 3. A hand-constructed version-1 `.tvim` file (TVIM magic with
//!    version byte 1) is rejected with the upgrade-hint error.

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use turbovec::io::{load, load_id_map, write, write_id_map};

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-{}-{}", nonce, name));
    p
}

fn runtime_cache_path(path: &std::path::Path) -> PathBuf {
    let mut file_name = path.file_name().unwrap().to_os_string();
    file_name.push(".");
    file_name.push(runtime_cache_backend_suffix());
    file_name.push(".cache");
    path.with_file_name(file_name)
}

#[test]
fn tv_round_trip_current_format() {
    let path = temp_path("v2.tv");
    let bit_width = 4;
    let dim = 32;
    let n_vectors = 3;
    let packed = vec![0xABu8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![1.5f32, 2.5, 3.5];

    // Round-trip with empty TQ+ calibration (identity); behaviour identical
    // to a v2 file otherwise. Separate test below covers populated calibration.
    write(&path, bit_width, dim, n_vectors, &packed, &scales, &[], &[]).unwrap();
    let cache_path = runtime_cache_path(&path);
    assert!(cache_path.exists());
    let (bw, d, n, p, s, shift, scale_tq) = load(&path).unwrap();

    assert_eq!(bw, bit_width);
    assert_eq!(d, dim);
    assert_eq!(n, n_vectors);
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert!(shift.is_empty());
    assert!(scale_tq.is_empty());
    fs::remove_file(&cache_path).ok();
    fs::remove_file(&path).ok();
}

#[test]
fn tv_round_trip_with_tqplus_calibration() {
    let path = temp_path("v3-tqplus.tv");
    let bit_width = 4;
    let dim = 32;
    let n_vectors = 3;
    let packed = vec![0xABu8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![1.5f32, 2.5, 3.5];
    let shift: Vec<f32> = (0..dim).map(|d| d as f32 * 0.01).collect();
    let scale_tq: Vec<f32> = (0..dim).map(|d| 1.0 + d as f32 * 0.02).collect();

    write(
        &path, bit_width, dim, n_vectors, &packed, &scales, &shift, &scale_tq,
    )
    .unwrap();
    let (bw, d, n, p, s, loaded_shift, loaded_scale) = load(&path).unwrap();

    assert_eq!(bw, bit_width);
    assert_eq!(d, dim);
    assert_eq!(n, n_vectors);
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert_eq!(loaded_shift, shift);
    assert_eq!(loaded_scale, scale_tq);
    fs::remove_file(runtime_cache_path(&path)).ok();
    fs::remove_file(&path).ok();
}

#[test]
fn tv_v1_file_is_rejected_with_upgrade_hint() {
    // Hand-construct a turbovec ≤ 0.4.3 `.tv` file: bare header
    // (bit_width=4, dim=32, n_vectors=2), packed codes, two f32 norms.
    let path = temp_path("v1.tv");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(&[4u8]).unwrap(); // bit_width
        f.write_all(&(32u32).to_le_bytes()).unwrap(); // dim
        f.write_all(&(2u32).to_le_bytes()).unwrap(); // n_vectors
        f.write_all(&vec![0u8; (32 / 8) * 4 * 2]).unwrap(); // packed codes
        f.write_all(&(1.0f32).to_le_bytes()).unwrap(); // norm 0
        f.write_all(&(2.0f32).to_le_bytes()).unwrap(); // norm 1
    }

    let err = load(&path).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("turbovec ≤ 0.4.3") && msg.contains("Rebuild"),
        "expected upgrade hint, got: {}",
        msg
    );
    fs::remove_file(&path).ok();
}

#[test]
fn tvim_round_trip_current_format() {
    let path = temp_path("v2.tvim");
    let bit_width = 2;
    let dim = 16;
    let n_vectors = 4;
    let packed = vec![0x55u8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![0.5f32, 1.0, 1.5, 2.0];
    let ids = vec![100u64, 200, 300, 400];

    write_id_map(
        &path,
        bit_width,
        dim,
        n_vectors,
        &packed,
        &scales,
        &[],
        &[],
        &ids,
    )
    .unwrap();
    let cache_path = runtime_cache_path(&path);
    assert!(cache_path.exists());
    let (bw, d, n, p, s, shift, scale_tq, slot_to_id) = load_id_map(&path).unwrap();

    assert_eq!(bw, bit_width);
    assert_eq!(d, dim);
    assert_eq!(n, n_vectors);
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert!(shift.is_empty());
    assert!(scale_tq.is_empty());
    assert_eq!(slot_to_id, ids);
    fs::remove_file(&cache_path).ok();
    fs::remove_file(&path).ok();
}

#[test]
fn tvim_load_creates_missing_runtime_cache_for_existing_file() {
    let path = temp_path("existing.tvim");
    let bit_width = 2;
    let dim = 16;
    let n_vectors = 4;
    let packed = vec![0x55u8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![0.5f32, 1.0, 1.5, 2.0];
    let ids = vec![100u64, 200, 300, 400];

    write_id_map(
        &path,
        bit_width,
        dim,
        n_vectors,
        &packed,
        &scales,
        &[],
        &[],
        &ids,
    )
    .unwrap();
    let cache_path = runtime_cache_path(&path);
    fs::remove_file(&cache_path).unwrap();

    let (bw, d, n, p, s, shift, scale_tq, slot_to_id) = load_id_map(&path).unwrap();

    assert!(cache_path.exists());
    assert_eq!(bw, bit_width);
    assert_eq!(d, dim);
    assert_eq!(n, n_vectors);
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert!(shift.is_empty());
    assert!(scale_tq.is_empty());
    assert_eq!(slot_to_id, ids);
    fs::remove_file(&cache_path).ok();
    fs::remove_file(&path).ok();
}

#[test]
fn tvim_load_repairs_corrupt_runtime_cache_for_existing_file() {
    let path = temp_path("corrupt-cache.tvim");
    let bit_width = 2;
    let dim = 16;
    let n_vectors = 4;
    let packed = vec![0x55u8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![0.5f32, 1.0, 1.5, 2.0];
    let ids = vec![100u64, 200, 300, 400];

    write_id_map(
        &path,
        bit_width,
        dim,
        n_vectors,
        &packed,
        &scales,
        &[],
        &[],
        &ids,
    )
    .unwrap();
    let cache_path = runtime_cache_path(&path);
    fs::write(&cache_path, b"bad cache").unwrap();

    let (bw, d, n, p, s, shift, scale_tq, slot_to_id) = load_id_map(&path).unwrap();

    assert!(cache_path.metadata().unwrap().len() > b"bad cache".len() as u64);
    assert_eq!(bw, bit_width);
    assert_eq!(d, dim);
    assert_eq!(n, n_vectors);
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert!(shift.is_empty());
    assert!(scale_tq.is_empty());
    assert_eq!(slot_to_id, ids);
    fs::remove_file(&cache_path).ok();
    fs::remove_file(&path).ok();
}

#[test]
fn runtime_cache_path_is_backend_specific() {
    let path = temp_path("backend.tvim");
    let cache_path = runtime_cache_path(&path);
    let cache_name = cache_path.file_name().unwrap().to_string_lossy();

    assert!(cache_name.contains(runtime_cache_backend_suffix()));
}

#[test]
fn tvim_load_repairs_wrong_backend_runtime_cache_for_existing_file() {
    let path = temp_path("wrong-backend-cache.tvim");
    let bit_width = 2;
    let dim = 16;
    let n_vectors = 4;
    let packed = vec![0x55u8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![0.5f32, 1.0, 1.5, 2.0];
    let ids = vec![100u64, 200, 300, 400];

    write_id_map(
        &path,
        bit_width,
        dim,
        n_vectors,
        &packed,
        &scales,
        &[],
        &[],
        &ids,
    )
    .unwrap();
    let cache_path = runtime_cache_path(&path);
    let mut cache_bytes = fs::read(&cache_path).unwrap();
    cache_bytes[5] = 0;
    fs::write(&cache_path, cache_bytes).unwrap();

    let (bw, d, n, p, s, shift, scale_tq, slot_to_id) = load_id_map(&path).unwrap();
    let repaired_cache = fs::read(&cache_path).unwrap();

    assert_eq!(repaired_cache[5], runtime_cache_backend_id());
    assert_eq!(bw, bit_width);
    assert_eq!(d, dim);
    assert_eq!(n, n_vectors);
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert!(shift.is_empty());
    assert!(scale_tq.is_empty());
    assert_eq!(slot_to_id, ids);
    fs::remove_file(&cache_path).ok();
    fs::remove_file(&path).ok();
}

#[test]
fn tvim_v1_file_is_rejected_with_upgrade_hint() {
    // Hand-construct a turbovec ≤ 0.4.3 `.tvim` file: TVIM magic, version
    // byte = 1, then the same v1 core layout.
    let path = temp_path("v1.tvim");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"TVIM").unwrap();
        f.write_all(&[1u8]).unwrap(); // version
        f.write_all(&[4u8]).unwrap(); // bit_width
        f.write_all(&(32u32).to_le_bytes()).unwrap(); // dim
        f.write_all(&(1u32).to_le_bytes()).unwrap(); // n_vectors
        f.write_all(&vec![0u8; (32 / 8) * 4]).unwrap(); // packed codes
        f.write_all(&(1.0f32).to_le_bytes()).unwrap(); // norm
        f.write_all(&(42u64).to_le_bytes()).unwrap(); // id
    }

    let err = load_id_map(&path).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("turbovec ≤ 0.4.3") && msg.contains("Rebuild"),
        "expected upgrade hint, got: {}",
        msg
    );
    fs::remove_file(&path).ok();
}

#[test]
fn tv_truncated_payload_errors_cleanly() {
    // Write a valid v3 .tv file, then truncate it mid-payload. `load`
    // must surface a clean io::Error (UnexpectedEof) rather than panic
    // or return malformed state.
    let path = temp_path("truncated.tv");
    let bit_width = 4;
    let dim = 32;
    let n_vectors = 5;
    let packed = vec![0xCDu8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![1.0f32; n_vectors];
    write(&path, bit_width, dim, n_vectors, &packed, &scales, &[], &[]).unwrap();

    // Truncate the file to half its size.
    let len = std::fs::metadata(&path).unwrap().len();
    let f = File::options().write(true).open(&path).unwrap();
    f.set_len(len / 2).unwrap();
    drop(f);

    let err = load(&path).unwrap_err();
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::UnexpectedEof,
        "expected UnexpectedEof on truncated file, got: {err}",
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_unsupported_version_errors_with_useful_message() {
    // Hand-construct a .tv file with a recognised magic but a version
    // byte we don't support. Loader must surface a clean InvalidData
    // error rather than try to parse with the wrong layout.
    let path = temp_path("future_version.tv");
    let mut f = File::create(&path).unwrap();
    f.write_all(b"TVPI").unwrap();
    f.write_all(&[99u8]).unwrap(); // version=99 — not 2, not 3
                                   // Pad with arbitrary bytes so the read doesn't fail before the
                                   // version check.
    f.write_all(&[0u8; 64]).unwrap();
    drop(f);

    let err = load(&path).unwrap_err();
    let msg = err.to_string();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        msg.contains("unsupported"),
        "expected 'unsupported' in error message, got: {msg}",
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_v3_invalid_n_calib_errors_cleanly() {
    // Hand-construct a v3 .tv file whose n_calib is neither 0 nor dim.
    // Loader must reject with InvalidData per the contract in io.rs.
    let path = temp_path("bad_n_calib.tv");
    let bit_width = 4u8;
    let dim = 32u32;
    let n_vectors = 1u32;

    let mut f = File::create(&path).unwrap();
    f.write_all(b"TVPI").unwrap();
    f.write_all(&[3u8]).unwrap(); // version=3
    f.write_all(&[bit_width]).unwrap();
    f.write_all(&dim.to_le_bytes()).unwrap();
    f.write_all(&n_vectors.to_le_bytes()).unwrap();
    // Packed codes: (dim/8) * bit_width * n_vectors = 4 * 4 * 1 = 16 bytes.
    f.write_all(&[0xAAu8; 16]).unwrap();
    // Scale: 1 f32.
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    // n_calib = 7 — neither 0 nor dim (32). Invalid.
    f.write_all(&7u32.to_le_bytes()).unwrap();
    // Pad with junk so the read doesn't fail before the n_calib check.
    f.write_all(&[0u8; 64]).unwrap();
    drop(f);

    let err = load(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("n_calib"),
        "expected 'n_calib' in error message, got: {err}",
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_garbage_file_rejected_without_upgrade_hint() {
    let path = temp_path("garbage.tv");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"NOPE").unwrap();
        f.write_all(&[0u8; 32]).unwrap();
    }
    let err = load(&path).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("wrong magic"),
        "expected wrong-magic error, got: {}",
        msg
    );
    assert!(
        !msg.contains("turbovec ≤ 0.4.3"),
        "should not suggest upgrade for garbage: {}",
        msg
    );
    fs::remove_file(&path).ok();
}

#[cfg(target_arch = "x86_64")]
fn runtime_cache_backend_suffix() -> &'static str {
    "x86_64-faiss-v1"
}

#[cfg(target_arch = "x86_64")]
fn runtime_cache_backend_id() -> u8 {
    1
}

#[cfg(target_arch = "aarch64")]
fn runtime_cache_backend_suffix() -> &'static str {
    "aarch64-neon-v1"
}

#[cfg(target_arch = "aarch64")]
fn runtime_cache_backend_id() -> u8 {
    2
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn runtime_cache_backend_suffix() -> &'static str {
    "scalar-v1"
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn runtime_cache_backend_id() -> u8 {
    3
}
