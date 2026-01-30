use crate::types::AudioChunk;
use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SendOutcome {
    Sent,
    DroppedOldest,
    Disconnected,
}

struct QueueState {
    closed: bool,
    items: VecDeque<AudioChunk>,
}

struct ChunkQueue {
    capacity: usize,
    state: Mutex<QueueState>,
    available: Condvar,
}

pub(crate) struct ChunkSender {
    inner: Arc<ChunkQueue>,
}

pub struct ChunkReceiver {
    inner: Arc<ChunkQueue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkRecvError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkRecvTimeoutError {
    Timeout,
    Disconnected,
}

impl fmt::Display for ChunkRecvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("chunk channel closed")
    }
}

impl fmt::Display for ChunkRecvTimeoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChunkRecvTimeoutError::Timeout => f.write_str("chunk receive timed out"),
            ChunkRecvTimeoutError::Disconnected => f.write_str("chunk channel closed"),
        }
    }
}

impl std::error::Error for ChunkRecvError {}
impl std::error::Error for ChunkRecvTimeoutError {}

pub(crate) fn chunk_channel(capacity: usize) -> (ChunkSender, ChunkReceiver) {
    debug_assert!(capacity > 0, "chunk channel capacity must be non-zero");
    let queue = Arc::new(ChunkQueue {
        capacity: capacity.max(1),
        state: Mutex::new(QueueState {
            closed: false,
            items: VecDeque::with_capacity(capacity),
        }),
        available: Condvar::new(),
    });

    (
        ChunkSender {
            inner: Arc::clone(&queue),
        },
        ChunkReceiver { inner: queue },
    )
}

impl ChunkSender {
    pub(crate) fn send_drop_oldest(&self, chunk: AudioChunk) -> SendOutcome {
        let mut state = self.inner.state.lock().unwrap();
        if state.closed {
            return SendOutcome::Disconnected;
        }

        let mut dropped = false;
        if state.items.len() == self.inner.capacity {
            state.items.pop_front();
            dropped = true;
        }

        state.items.push_back(chunk);
        self.inner.available.notify_one();

        if dropped {
            SendOutcome::DroppedOldest
        } else {
            SendOutcome::Sent
        }
    }
}

impl Drop for ChunkSender {
    fn drop(&mut self) {
        let mut state = self.inner.state.lock().unwrap();
        if !state.closed {
            state.closed = true;
            self.inner.available.notify_all();
        }
    }
}

impl ChunkReceiver {
    pub fn recv(&self) -> Result<AudioChunk, ChunkRecvError> {
        let mut state = self.inner.state.lock().unwrap();
        loop {
            if let Some(item) = state.items.pop_front() {
                return Ok(item);
            }

            if state.closed {
                return Err(ChunkRecvError);
            }

            state = self.inner.available.wait(state).unwrap();
        }
    }

    pub fn recv_timeout(&self, timeout: Duration) -> Result<AudioChunk, ChunkRecvTimeoutError> {
        let mut state = self.inner.state.lock().unwrap();
        let start = Instant::now();
        let mut remaining = timeout;

        loop {
            if let Some(item) = state.items.pop_front() {
                return Ok(item);
            }

            if state.closed {
                return Err(ChunkRecvTimeoutError::Disconnected);
            }

            let (next_state, result) = self.inner.available.wait_timeout(state, remaining).unwrap();
            state = next_state;

            if result.timed_out() {
                return Err(ChunkRecvTimeoutError::Timeout);
            }

            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return Err(ChunkRecvTimeoutError::Timeout);
            }
            remaining = timeout - elapsed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ChunkRecvTimeoutError, SendOutcome, chunk_channel};
    use crate::types::{AudioChunk, AudioSource};
    use std::time::Duration;

    fn make_chunk(id: i128) -> AudioChunk {
        AudioChunk {
            source: AudioSource::System,
            start_pts_ns: id,
            sample_rate_hz: 16_000,
            pcm_mono_f32: vec![id as f32],
        }
    }

    #[test]
    fn drop_oldest_when_full() {
        let (tx, rx) = chunk_channel(2);

        assert_eq!(tx.send_drop_oldest(make_chunk(1)), SendOutcome::Sent);
        assert_eq!(tx.send_drop_oldest(make_chunk(2)), SendOutcome::Sent);
        assert_eq!(
            tx.send_drop_oldest(make_chunk(3)),
            SendOutcome::DroppedOldest
        );

        assert_eq!(rx.recv().unwrap().start_pts_ns, 2);
        assert_eq!(rx.recv().unwrap().start_pts_ns, 3);
    }

    #[test]
    fn recv_errors_after_sender_drop() {
        let (tx, rx) = chunk_channel(1);
        drop(tx);
        assert!(rx.recv().is_err());
    }

    #[test]
    fn recv_timeout_times_out() {
        let (_tx, rx) = chunk_channel(1);
        assert!(matches!(
            rx.recv_timeout(Duration::from_millis(10)),
            Err(ChunkRecvTimeoutError::Timeout)
        ));
    }
}
