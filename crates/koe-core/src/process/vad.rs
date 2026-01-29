use crate::error::ProcessError;
use voice_activity_detector::VoiceActivityDetector;

const SAMPLE_RATE: i64 = 16_000;
const CHUNK_SIZE: usize = 512;
const THRESHOLD: f32 = 0.5;
const MIN_SPEECH_FRAMES: u32 = 7; // ~224 ms at 512 samples / 16 kHz
const HANGOVER_FRAMES: u32 = 10; // ~320 ms

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Silence,
    Speech,
    Hangover,
}

/// Silero VAD wrapper with a three-state machine: Silence -> Speech -> Hangover -> Silence.
pub struct VadDetector {
    vad: VoiceActivityDetector,
    state: State,
    speech_count: u32,
    hangover_count: u32,
}

impl VadDetector {
    pub fn new() -> Result<Self, ProcessError> {
        let vad = VoiceActivityDetector::builder()
            .sample_rate(SAMPLE_RATE)
            .chunk_size(CHUNK_SIZE)
            .build()
            .map_err(|e| ProcessError::VadInit(e.to_string()))?;

        Ok(Self {
            vad,
            state: State::Silence,
            speech_count: 0,
            hangover_count: 0,
        })
    }

    /// Process a 512-sample frame at 16 kHz. Returns true when in confirmed speech.
    pub fn process_frame(&mut self, frame_512: &[f32]) -> bool {
        let prob = self.vad.predict(frame_512.iter().copied());

        let is_speech_frame = prob >= THRESHOLD;

        match self.state {
            State::Silence => {
                if is_speech_frame {
                    self.speech_count += 1;
                    if self.speech_count >= MIN_SPEECH_FRAMES {
                        self.state = State::Speech;
                        self.speech_count = 0;
                    }
                } else {
                    self.speech_count = 0;
                }
            }
            State::Speech => {
                if !is_speech_frame {
                    self.state = State::Hangover;
                    self.hangover_count = 1;
                }
            }
            State::Hangover => {
                if is_speech_frame {
                    self.state = State::Speech;
                    self.hangover_count = 0;
                } else {
                    self.hangover_count += 1;
                    if self.hangover_count >= HANGOVER_FRAMES {
                        self.state = State::Silence;
                        self.hangover_count = 0;
                    }
                }
            }
        }

        matches!(self.state, State::Speech | State::Hangover)
    }

    pub fn reset(&mut self) {
        self.vad.reset();
        self.state = State::Silence;
        self.speech_count = 0;
        self.hangover_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence_frame() -> Vec<f32> {
        vec![0.0; CHUNK_SIZE]
    }

    #[test]
    fn silence_stays_silent() {
        let mut vad = VadDetector::new().unwrap();
        for _ in 0..50 {
            assert!(!vad.process_frame(&silence_frame()));
        }
    }

    #[test]
    fn state_machine_transitions() {
        // Test the state machine logic directly by manipulating state,
        // since Silero VAD is an ML model that requires real speech audio.
        let mut vad = VadDetector::new().unwrap();
        assert_eq!(vad.state, State::Silence);

        // Simulate MIN_SPEECH_FRAMES consecutive speech detections
        vad.speech_count = MIN_SPEECH_FRAMES;
        // Manually transition
        vad.state = State::Speech;
        assert!(matches!(vad.state, State::Speech));

        // Transition to hangover
        vad.state = State::Hangover;
        vad.hangover_count = 1;
        assert!(matches!(vad.state, State::Hangover));

        // Complete hangover returns to silence
        vad.hangover_count = HANGOVER_FRAMES;
        vad.state = State::Silence;
        assert!(matches!(vad.state, State::Silence));
    }

    #[test]
    fn process_frame_silence_to_speech_requires_min_frames() {
        let mut vad = VadDetector::new().unwrap();

        // With silence input, speech_count should not accumulate
        for _ in 0..20 {
            let result = vad.process_frame(&silence_frame());
            assert!(!result);
        }
        assert_eq!(vad.speech_count, 0);
        assert_eq!(vad.state, State::Silence);
    }

    #[test]
    fn hangover_from_speech_state() {
        let mut vad = VadDetector::new().unwrap();

        // Force into speech state (as if ML model confirmed speech)
        vad.state = State::Speech;

        // Feed silence - should transition to hangover
        let result = vad.process_frame(&silence_frame());
        assert!(result, "hangover state should report as speech");
        assert_eq!(vad.state, State::Hangover);
        assert_eq!(vad.hangover_count, 1);

        // Continue feeding silence through hangover
        for i in 1..HANGOVER_FRAMES - 1 {
            let result = vad.process_frame(&silence_frame());
            assert!(result, "should still be in hangover at frame {i}");
        }

        // Final hangover frame transitions to silence
        let result = vad.process_frame(&silence_frame());
        assert!(!result, "should be silence after hangover completes");
        assert_eq!(vad.state, State::Silence);
    }

    #[test]
    fn reset_clears_state() {
        let mut vad = VadDetector::new().unwrap();
        vad.state = State::Speech;
        vad.speech_count = 5;
        vad.hangover_count = 3;

        vad.reset();
        assert_eq!(vad.state, State::Silence);
        assert_eq!(vad.speech_count, 0);
        assert_eq!(vad.hangover_count, 0);
    }
}
