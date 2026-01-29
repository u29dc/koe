use crate::error::ProcessError;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

const INPUT_RATE: f64 = 48_000.0;
const OUTPUT_RATE: f64 = 16_000.0;
const RATIO: f64 = OUTPUT_RATE / INPUT_RATE; // 1/3
const CHUNK_SIZE: usize = 480; // 10 ms at 48 kHz

/// Resamples 48 kHz mono audio to 16 kHz mono using a windowed sinc filter.
pub struct ResampleConverter {
    inner: SincFixedIn<f32>,
}

impl ResampleConverter {
    pub fn new() -> Result<Self, ProcessError> {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            oversampling_factor: 256,
            interpolation: SincInterpolationType::Linear,
            window: WindowFunction::BlackmanHarris2,
        };

        let resampler = SincFixedIn::<f32>::new(RATIO, 1.0, params, CHUNK_SIZE, 1)
            .map_err(|e| ProcessError::ResamplerInit(e.to_string()))?;

        Ok(Self { inner: resampler })
    }

    /// Resample a buffer of 48 kHz mono f32 samples to 16 kHz.
    /// Input length must be a multiple of `CHUNK_SIZE` (480 samples).
    /// Remaining samples that don't fill a complete chunk are dropped;
    /// callers should buffer and prepend them to the next call.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, ProcessError> {
        let mut output = Vec::with_capacity((input.len() as f64 * RATIO) as usize + 64);

        let full_chunks = input.len() / CHUNK_SIZE;
        for i in 0..full_chunks {
            let start = i * CHUNK_SIZE;
            let end = start + CHUNK_SIZE;
            let chunk = &input[start..end];

            let result = self
                .inner
                .process(&[chunk], None)
                .map_err(|e| ProcessError::ResampleFailed(e.to_string()))?;

            if let Some(channel) = result.first() {
                output.extend_from_slice(channel);
            }
        }

        Ok(output)
    }

    /// Returns the chunk size the resampler expects.
    pub fn chunk_size(&self) -> usize {
        CHUNK_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_length_approximately_one_third() {
        let mut resampler = ResampleConverter::new().unwrap();
        let input_len = CHUNK_SIZE * 100; // 100 chunks
        let input: Vec<f32> = vec![0.0; input_len];
        let output = resampler.process(&input).unwrap();

        let expected = (input_len as f64 * RATIO) as usize;
        let tolerance = expected / 10; // 10% tolerance for filter delay
        assert!(
            output.len().abs_diff(expected) < tolerance,
            "output {} not close to expected {}",
            output.len(),
            expected
        );
    }

    #[test]
    fn processes_partial_input() {
        let mut resampler = ResampleConverter::new().unwrap();
        // Input not a multiple of CHUNK_SIZE - remainder is dropped
        let input: Vec<f32> = vec![0.0; CHUNK_SIZE * 5 + 100];
        let output = resampler.process(&input).unwrap();

        // Should process 5 full chunks worth
        let expected = (CHUNK_SIZE as f64 * 5.0 * RATIO) as usize;
        let tolerance = expected / 5;
        assert!(
            output.len().abs_diff(expected) < tolerance,
            "output {} not close to expected {}",
            output.len(),
            expected
        );
    }

    #[test]
    fn empty_input_produces_empty_output() {
        let mut resampler = ResampleConverter::new().unwrap();
        let output = resampler.process(&[]).unwrap();
        assert!(output.is_empty());
    }
}
