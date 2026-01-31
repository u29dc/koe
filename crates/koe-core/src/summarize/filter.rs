use std::collections::HashSet;

pub fn build_participant_tokens(participants: &[String]) -> HashSet<String> {
    let mut tokens = HashSet::new();
    for participant in participants {
        for token in normalize_text(participant).split_whitespace() {
            if !token.is_empty() {
                tokens.insert(token.to_string());
            }
        }
    }
    tokens
}

pub fn should_keep_segment(text: &str, participant_tokens: &HashSet<String>) -> bool {
    let normalized = normalize_text(text);
    if normalized.is_empty() || is_ack_phrase(&normalized) {
        return false;
    }

    let word_count = normalized.split_whitespace().count();
    if word_count >= 3 {
        return true;
    }

    if contains_digit(text) || contains_temporal_keyword(&normalized) {
        return true;
    }

    if word_count == 1 {
        if let Some(word) = normalized.split_whitespace().next()
            && (participant_tokens.contains(word) || is_acronym(text))
        {
            return true;
        }
        return false;
    }

    true
}

pub fn normalize_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = true;
    for ch in text.chars() {
        let mapped = if ch.is_ascii_alphanumeric() {
            ch.to_ascii_lowercase()
        } else {
            ' '
        };

        if mapped == ' ' {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(mapped);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

fn is_ack_phrase(normalized: &str) -> bool {
    const ACK_PHRASES: [&str; 16] = [
        "ok",
        "okay",
        "thanks",
        "thank you",
        "yeah",
        "yep",
        "uh",
        "um",
        "hmm",
        "right",
        "got it",
        "sounds good",
        "all right",
        "alright",
        "cool",
        "great",
    ];

    ACK_PHRASES.contains(&normalized)
}

fn contains_temporal_keyword(normalized: &str) -> bool {
    const TEMPORAL: [&str; 36] = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "today",
        "tomorrow",
        "yesterday",
        "tonight",
        "week",
        "weeks",
        "month",
        "months",
        "quarter",
        "quarters",
        "q1",
        "q2",
        "q3",
        "q4",
        "eod",
        "eow",
        "eom",
    ];

    normalized
        .split_whitespace()
        .any(|token| TEMPORAL.iter().any(|value| value == &token))
}

fn contains_digit(text: &str) -> bool {
    text.chars().any(|ch| ch.is_ascii_digit())
}

fn is_acronym(text: &str) -> bool {
    let letters: Vec<char> = text.chars().filter(|ch| ch.is_ascii_alphabetic()).collect();
    letters.len() >= 2 && letters.iter().all(|ch| ch.is_ascii_uppercase())
}
