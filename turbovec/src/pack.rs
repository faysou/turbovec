//! Bit-plane to SIMD-blocked layout repacking.
//!
//! Converts bit-plane packed codes into a layout optimised for SIMD scoring:
//! - x86: FAISS-style perm0-interleaved for AVX2 cross-lane compatibility
//! - ARM: Sequential layout for NEON

use crate::BLOCK;

/// Repack bit-plane codes into SIMD-blocked layout.
/// Returns (blocked_codes, n_blocks).
///
/// Crate-internal: trusts `2 <= bits <= 4`, `dim` a multiple of 8, and
/// `packed_codes.len() == n_vectors * (dim/8) * bits`. A raw caller passing
/// `bits == 0` divides by zero and a short `packed_codes` reads out of
/// bounds. Construct through
/// [`from_parts`](crate::TurboQuantIndex::from_parts) instead, which
/// validates these before the blocked layout is ever built.
pub(crate) fn repack(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
) -> (Vec<u8>, usize) {
    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    let blocked = repack_block_range(packed_codes, n_vectors, bits, dim, 0, n_blocks);
    (blocked, n_blocks)
}

/// Repack bit-plane codes for a contiguous block range.
///
/// `start_block` and `end_block` use 32-vector block indices. The returned
/// bytes are laid out as if sliced from the full blocked layout at
/// `start_block`.
pub(crate) fn repack_block_range(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
    start_block: usize,
    end_block: usize,
) -> Vec<u8> {
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    let start_block = start_block.min(n_blocks);
    let end_block = end_block.min(n_blocks);
    if start_block >= end_block {
        return Vec::new();
    }

    let blocked_size = (end_block - start_block) * n_byte_groups * BLOCK;
    pack_block_range(
        packed_codes,
        n_vectors,
        bits,
        dim,
        n_byte_groups,
        start_block,
        end_block,
        blocked_size,
    )
}

#[cfg(target_arch = "x86_64")]
fn pack_block_range(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
    n_byte_groups: usize,
    start_block: usize,
    end_block: usize,
    blocked_size: usize,
) -> Vec<u8> {
    let perm0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];
    let mut blocked = vec![0u8; blocked_size];
    for block_idx in start_block..end_block {
        let base_vec = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = ((block_idx - start_block) * n_byte_groups + g) * BLOCK;
            for j in 0..16 {
                let va = base_vec + perm0[j];
                let vb = base_vec + perm0[j] + 16;
                let ba = packed_group_byte(packed_codes, n_vectors, bits, dim, va, g);
                let bb = packed_group_byte(packed_codes, n_vectors, bits, dim, vb, g);
                blocked[out_offset + j] = (ba >> 4) | ((bb >> 4) << 4);
                blocked[out_offset + 16 + j] = (ba & 0x0F) | ((bb & 0x0F) << 4);
            }
        }
    }
    blocked
}

#[cfg(not(target_arch = "x86_64"))]
fn pack_block_range(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
    n_byte_groups: usize,
    start_block: usize,
    end_block: usize,
    blocked_size: usize,
) -> Vec<u8> {
    let mut blocked = vec![0u8; blocked_size];
    for block_idx in start_block..end_block {
        let base_vec = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = ((block_idx - start_block) * n_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vi = base_vec + lane;
                blocked[out_offset + lane] =
                    packed_group_byte(packed_codes, n_vectors, bits, dim, vi, g);
            }
        }
    }
    blocked
}

fn packed_group_byte(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
    vec_idx: usize,
    group: usize,
) -> u8 {
    if vec_idx >= n_vectors {
        return 0;
    }

    let bytes_per_plane = dim / 8;
    let codes_per_byte = 8 / bits;
    let bytes_per_row = bits * bytes_per_plane;
    let dim_start = group * codes_per_byte;
    let mut byte_val = 0u8;

    for c in 0..codes_per_byte {
        let j = dim_start + c;
        let byte_in_plane = j / 8;
        let bit_in_byte = 7 - (j % 8);
        let mask = 1u8 << bit_in_byte;

        let mut code = 0u8;
        for p in 0..bits {
            let plane_byte =
                packed_codes[vec_idx * bytes_per_row + p * bytes_per_plane + byte_in_plane];
            if plane_byte & mask != 0 {
                code |= 1 << p;
            }
        }

        let shift = if bits == 3 {
            (codes_per_byte - 1 - c) * 4
        } else {
            (codes_per_byte - 1 - c) * bits
        };
        byte_val |= code << shift;
    }
    byte_val
}

/// Inverse of the `perm0` permutation used by the x86 blocked layout:
/// `INV_PERM0[lane] == j` such that `perm0[j] == lane`, for `lane` in 0..16.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) const INV_PERM0: [usize; 16] = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15];

/// Reconstruct the sequential code byte for vector `lane` of a block group
/// from the x86 `perm0`-interleaved hi/lo-nibble layout.
#[inline]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn deinterleave_x86_code_byte(blocked: &[u8], group_off: usize, lane: usize) -> u8 {
    let j = INV_PERM0[lane & 15];
    let hi_plane = blocked[group_off + j];
    let lo_plane = blocked[group_off + 16 + j];
    let (hi, lo) = if lane < 16 {
        (hi_plane & 0x0F, lo_plane & 0x0F)
    } else {
        (hi_plane >> 4, lo_plane >> 4)
    };
    (hi << 4) | lo
}

#[cfg(test)]
mod tests {
    use super::{deinterleave_x86_code_byte, BLOCK};

    #[test]
    fn deinterleave_x86_recovers_sequential_code_bytes() {
        let n_byte_groups = 5usize;
        let mut codes_flat = vec![vec![0u8; n_byte_groups]; BLOCK];
        let mut s = 0x1234_5678u32;
        for row in &mut codes_flat {
            for code in row {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *code = (s >> 24) as u8;
            }
        }

        let perm0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];
        let mut blocked = vec![0u8; n_byte_groups * BLOCK];
        for g in 0..n_byte_groups {
            let out_offset = g * BLOCK;
            for j in 0..16 {
                let ba = codes_flat[perm0[j]][g];
                let bb = codes_flat[perm0[j] + 16][g];
                blocked[out_offset + j] = (ba >> 4) | ((bb >> 4) << 4);
                blocked[out_offset + 16 + j] = (ba & 0x0F) | ((bb & 0x0F) << 4);
            }
        }

        for g in 0..n_byte_groups {
            for lane in 0..BLOCK {
                assert_eq!(
                    deinterleave_x86_code_byte(&blocked, g * BLOCK, lane),
                    codes_flat[lane][g],
                    "mismatch at lane {lane}, group {g}",
                );
            }
        }
    }
}
