pub mod groq;
pub mod whisper;

use crate::{AudioChunk, TranscribeError, TranscriptSegment};

/// Speech-to-text provider abstraction.
pub trait TranscribeProvider: Send {
    fn name(&self) -> &'static str;
    fn transcribe(&mut self, chunk: &AudioChunk)
    -> Result<Vec<TranscriptSegment>, TranscribeError>;
}

/// Create a transcribe provider by name.
///
/// - `"whisper"` requires `model` pointing to a GGML model file path.
/// - `"groq"` requires an API key; `model` selects the Groq model name
///   (defaults to `whisper-large-v3-turbo`).
pub fn create_transcribe_provider(
    provider: &str,
    model: Option<&str>,
    api_key: Option<&str>,
) -> Result<Box<dyn TranscribeProvider>, TranscribeError> {
    match provider {
        "whisper" => {
            let path = model.ok_or_else(|| {
                TranscribeError::ModelLoad(
                    "model path required for whisper provider (--transcribe-model /path/to/ggml-*.bin)"
                        .into(),
                )
            })?;
            Ok(Box::new(whisper::WhisperProvider::new(path)?))
        }
        "groq" => Ok(Box::new(groq::GroqProvider::new(model, api_key)?)),
        other => Err(TranscribeError::ModelLoad(format!(
            "unknown transcribe provider: {other}"
        ))),
    }
}

/// Encode f32 PCM samples as a WAV file (RIFF/WAVE, IEEE float32, mono).
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 32;
    let block_align = num_channels * (bits_per_sample / 8);
    let byte_rate = sample_rate * u32::from(block_align);
    let data_size = (samples.len() * 4) as u32;
    // RIFF header = 12 bytes, fmt chunk = 26 bytes (8 header + 18 body), fact = 12 bytes, data chunk = 8 + data
    // For IEEE float we need: fmt size = 18 (format code 3 + cbSize=0), plus a fact chunk.
    // Simpler approach: use extended fmt (size 18) + fact chunk.
    let fmt_chunk_size: u32 = 18;
    let fact_chunk_size: u32 = 4;
    // Total file size = 4 (WAVE) + (8 + fmt_chunk_size) + (8 + fact_chunk_size) + (8 + data_size)
    let file_size = 4 + (8 + fmt_chunk_size) + (8 + fact_chunk_size) + (8 + data_size);

    let mut buf = Vec::with_capacity(12 + file_size as usize);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt sub-chunk (IEEE float = format code 3)
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&fmt_chunk_size.to_le_bytes());
    buf.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // cbSize = 0

    // fact sub-chunk (required for non-PCM)
    buf.extend_from_slice(b"fact");
    buf.extend_from_slice(&fact_chunk_size.to_le_bytes());
    buf.extend_from_slice(&(samples.len() as u32).to_le_bytes());

    // data sub-chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_encoder_produces_valid_header() {
        let samples = vec![0.0f32; 160]; // 10ms at 16kHz
        let wav = encode_wav(&samples, 16000);

        // RIFF magic
        assert_eq!(&wav[0..4], b"RIFF");
        // WAVE magic
        assert_eq!(&wav[8..12], b"WAVE");
        // fmt chunk
        assert_eq!(&wav[12..16], b"fmt ");
        // format = IEEE float (3)
        let format = u16::from_le_bytes([wav[20], wav[21]]);
        assert_eq!(format, 3);
        // channels = 1
        let channels = u16::from_le_bytes([wav[22], wav[23]]);
        assert_eq!(channels, 1);
        // sample rate = 16000
        let sr = u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]);
        assert_eq!(sr, 16000);
        // data chunk exists
        let data_offset = 12 + 26 + 12; // RIFF header + fmt chunk + fact chunk
        assert_eq!(&wav[data_offset..data_offset + 4], b"data");
        // data size = 160 * 4 = 640
        let data_size = u32::from_le_bytes([
            wav[data_offset + 4],
            wav[data_offset + 5],
            wav[data_offset + 6],
            wav[data_offset + 7],
        ]);
        assert_eq!(data_size, 640);
        // total file size
        let file_size = u32::from_le_bytes([wav[4], wav[5], wav[6], wav[7]]);
        assert_eq!(file_size as usize + 8, wav.len());
    }

    #[test]
    fn wav_encoder_round_trip_samples() {
        let samples = vec![1.0f32, -1.0, 0.5, -0.5];
        let wav = encode_wav(&samples, 16000);
        let data_offset = 12 + 26 + 12 + 8; // after data chunk header
        for (i, &expected) in samples.iter().enumerate() {
            let offset = data_offset + i * 4;
            let value = f32::from_le_bytes([
                wav[offset],
                wav[offset + 1],
                wav[offset + 2],
                wav[offset + 3],
            ]);
            assert_eq!(value, expected);
        }
    }
}
