use std::time::Duration;
use ureq::{Agent, Error as UreqError};

const TIMEOUT_GLOBAL: Duration = Duration::from_secs(90);
const TIMEOUT_PER_CALL: Duration = Duration::from_secs(60);
const TIMEOUT_RESOLVE: Duration = Duration::from_secs(5);
const TIMEOUT_CONNECT: Duration = Duration::from_secs(5);
const TIMEOUT_SEND_REQUEST: Duration = Duration::from_secs(5);
const TIMEOUT_SEND_BODY: Duration = Duration::from_secs(15);
const TIMEOUT_RECV_RESPONSE: Duration = Duration::from_secs(10);
const TIMEOUT_RECV_BODY: Duration = Duration::from_secs(60);

const RETRY_BASE_MS: u64 = 200;

pub fn default_agent() -> Agent {
    let config = Agent::config_builder()
        .timeout_global(Some(TIMEOUT_GLOBAL))
        .timeout_per_call(Some(TIMEOUT_PER_CALL))
        .timeout_resolve(Some(TIMEOUT_RESOLVE))
        .timeout_connect(Some(TIMEOUT_CONNECT))
        .timeout_send_request(Some(TIMEOUT_SEND_REQUEST))
        .timeout_send_body(Some(TIMEOUT_SEND_BODY))
        .timeout_recv_response(Some(TIMEOUT_RECV_RESPONSE))
        .timeout_recv_body(Some(TIMEOUT_RECV_BODY))
        .build();
    config.into()
}

pub fn should_retry(err: &UreqError) -> bool {
    match err {
        UreqError::StatusCode(code) => *code == 429 || (500..=599).contains(code),
        UreqError::Timeout(_)
        | UreqError::Io(_)
        | UreqError::HostNotFound
        | UreqError::ConnectionFailed
        | UreqError::TooManyRedirects
        | UreqError::RedirectFailed => true,
        _ => false,
    }
}

pub fn retry_delay(attempt: usize) -> Duration {
    let shift = attempt.min(6) as u32;
    let delay = RETRY_BASE_MS.saturating_mul(1_u64 << shift);
    Duration::from_millis(delay)
}
