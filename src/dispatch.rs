/// WebGPU spec guarantee for the largest dispatch dimension.
pub const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Reshape a 1-D dispatch of `total` workgroups into a 2-D `(x, y)`
/// dispatch where both dimensions are at most [`MAX_WORKGROUPS_PER_DIM`].
pub fn split_1d_into_2d(total: u32) -> (u32, u32) {
    if total <= MAX_WORKGROUPS_PER_DIM {
        return (total, 1);
    }
    let y = total.div_ceil(MAX_WORKGROUPS_PER_DIM);
    let x = total.div_ceil(y);
    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_fits_in_x() {
        assert_eq!(split_1d_into_2d(0), (0, 1));
        assert_eq!(split_1d_into_2d(1), (1, 1));
        assert_eq!(split_1d_into_2d(65535), (65535, 1));
    }

    #[test]
    fn overflow_rounds_up_to_y() {
        let (x, y) = split_1d_into_2d(65536);
        assert!(x <= MAX_WORKGROUPS_PER_DIM);
        assert_eq!(y, 2);
        assert!(x * y >= 65536);

        let total = 890_880;
        let (x, y) = split_1d_into_2d(total);
        assert!(x <= MAX_WORKGROUPS_PER_DIM);
        assert!(y <= MAX_WORKGROUPS_PER_DIM);
        assert!(x * y >= total);
    }

    /// The split should overshoot `total` by at most `x` (a single trailing
    /// row of workgroups). Without this property, large dispatches waste
    /// tens of thousands of workgroups on the early-return path.
    #[test]
    fn overshoot_is_at_most_one_row() {
        for &total in &[65_536u32, 100_000, 890_880, 4_000_000, u32::MAX / 4] {
            let (x, y) = split_1d_into_2d(total);
            let dispatched = (x as u64) * (y as u64);
            assert!(dispatched >= total as u64);
            assert!(
                dispatched - total as u64 <= x as u64,
                "total={total} (x,y)=({x},{y}) overshoot={}",
                dispatched - total as u64
            );
        }
    }
}
