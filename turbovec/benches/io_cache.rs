extern crate blas_src;

use std::fs;
use std::path::{Path, PathBuf};

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use turbovec::IdMapIndex;

const DIM: usize = 256;
const BIT_WIDTH: usize = 2;
const N_VECTORS: usize = 4096;
const APPEND_ID: u64 = 10_000_000;
const DELETE_IDS: &[u64] = &[3, 41, 97, 511, 1025, 2049, 3073];

fn bench_io_cache(c: &mut Criterion) {
    let fixture = Fixture::new();
    let append_vector = gaussian_normalized(1, DIM, 0xBEE5_1001);

    let mut group = c.benchmark_group("tvim_io_cache");
    group.sample_size(20);

    group.bench_function("load_with_existing_cache", |b| {
        b.iter(|| {
            let index = IdMapIndex::load(black_box(fixture.path())).unwrap();
            black_box(index.len());
        });
    });

    group.bench_function("write_unchanged", |b| {
        b.iter_batched(
            || {
                let path = fixture.copy_case("write_unchanged");
                let index = IdMapIndex::load(&path).unwrap();
                (path, index)
            },
            |(path, index)| {
                index.write(black_box(&path)).unwrap();
                black_box(path.metadata().unwrap().len());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("write_new_index_creates_cache", |b| {
        b.iter_batched(
            || {
                let path = fixture.new_case_path("write_new_index_creates_cache");
                let index = IdMapIndex::load(fixture.path()).unwrap();
                (path, index)
            },
            |(path, index)| {
                index.write(black_box(&path)).unwrap();
                black_box(path.metadata().unwrap().len());
                black_box(runtime_cache_path(&path).metadata().unwrap().len());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("append_one_then_write", |b| {
        b.iter_batched(
            || {
                let path = fixture.copy_case("append_one_then_write");
                let mut index = IdMapIndex::load(&path).unwrap();
                index
                    .add_with_ids_2d(&append_vector, DIM, &[APPEND_ID])
                    .unwrap();
                (path, index)
            },
            |(path, index)| {
                index.write(black_box(&path)).unwrap();
                black_box(path.metadata().unwrap().len());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("delete_one_then_write", |b| {
        b.iter_batched(
            || {
                let path = fixture.copy_case("delete_one_then_write");
                let mut index = IdMapIndex::load(&path).unwrap();
                assert!(index.remove(41));
                (path, index)
            },
            |(path, index)| {
                index.write(black_box(&path)).unwrap();
                black_box(path.metadata().unwrap().len());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("delete_many_blocks_then_four_writes", |b| {
        b.iter_batched(
            || {
                let path = fixture.copy_case("delete_many_blocks_then_four_writes");
                let mut index = IdMapIndex::load(&path).unwrap();
                for &id in DELETE_IDS {
                    assert!(index.remove(id));
                }
                (path, index)
            },
            |(path, index)| {
                for _ in 0..4 {
                    index.write(black_box(&path)).unwrap();
                }
                black_box(path.metadata().unwrap().len());
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

struct Fixture {
    dir: PathBuf,
    path: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir =
            std::env::temp_dir().join(format!("turbovec_io_cache_bench_{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.tvim");

        let mut index = IdMapIndex::new(DIM, BIT_WIDTH).unwrap();
        let ids: Vec<u64> = (0..N_VECTORS as u64).collect();
        let vectors = gaussian_normalized(N_VECTORS, DIM, 0xBEE5_0001);
        index.add_with_ids_2d(&vectors, DIM, &ids).unwrap();
        index.write(&path).unwrap();

        Self { dir, path }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn copy_case(&self, name: &str) -> PathBuf {
        let path = self.new_case_path(name);
        fs::copy(&self.path, &path).unwrap();
        fs::copy(runtime_cache_path(&self.path), runtime_cache_path(&path)).unwrap();
        path
    }

    fn new_case_path(&self, name: &str) -> PathBuf {
        let stem = format!("{name}_{}", unique_suffix());
        self.dir.join(format!("{stem}.tvim"))
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.dir).ok();
    }
}

fn runtime_cache_path(path: &Path) -> PathBuf {
    let mut file_name = path.file_name().unwrap().to_os_string();
    file_name.push(".");
    file_name.push(runtime_cache_backend_suffix());
    file_name.push(".cache");
    path.with_file_name(file_name)
}

fn unique_suffix() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed).to_string()
}

fn gaussian_normalized(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut uniform = || {
        let raw = (next() >> 40) as u32 | 1;
        raw as f32 / (1u32 << 24) as f32
    };
    let two_pi = 2.0_f32 * std::f32::consts::PI;
    let mut data = vec![0.0f32; n * dim];
    let mut i = 0;
    while i < data.len() {
        let u1 = uniform().max(1e-7);
        let u2 = uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = two_pi * u2;
        data[i] = r * theta.cos();
        if i + 1 < data.len() {
            data[i + 1] = r * theta.sin();
        }
        i += 2;
    }
    for row_i in 0..n {
        let row = &mut data[row_i * dim..(row_i + 1) * dim];
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for x in row.iter_mut() {
                *x *= inv;
            }
        }
    }
    data
}

#[cfg(target_arch = "x86_64")]
fn runtime_cache_backend_suffix() -> &'static str {
    "x86_64-faiss-v1"
}

#[cfg(target_arch = "aarch64")]
fn runtime_cache_backend_suffix() -> &'static str {
    "aarch64-neon-v1"
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn runtime_cache_backend_suffix() -> &'static str {
    "scalar-v1"
}

criterion_group!(benches, bench_io_cache);
criterion_main!(benches);
