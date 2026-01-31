use crate::config::{Config, ConfigError, ConfigPaths};
use clap::Args;
use koe_core::capture::list_audio_inputs as list_input_devices;
use std::process::Command;

#[derive(Args, Debug, Clone)]
pub struct ConfigArgs {
    /// Print config with secrets redacted
    #[arg(long)]
    pub print: bool,

    /// List available audio input devices
    #[arg(long)]
    pub list_inputs: bool,

    /// Edit config in $EDITOR
    #[arg(long)]
    pub edit: bool,

    /// Set a config value (dotted key=value)
    #[arg(long, value_name = "key=value")]
    pub set: Vec<String>,
}

pub fn run(args: &ConfigArgs, paths: &ConfigPaths) -> Result<(), ConfigError> {
    if args.list_inputs {
        list_audio_inputs();
        return Ok(());
    }

    if args.edit && (!args.set.is_empty() || args.print) {
        return Err(ConfigError::Validation(
            "--edit cannot be combined with --set or --print".into(),
        ));
    }

    let mut config = Config::load_or_create(paths)?;

    if args.edit {
        edit_config(paths)?;
        config = Config::load(paths)?;
        config.validate()?;
        return Ok(());
    }

    if !args.set.is_empty() {
        for assignment in &args.set {
            apply_set(&mut config, assignment)?;
        }
        config.validate()?;
        Config::write(paths, &config)?;
    }

    if args.print || (args.set.is_empty() && !args.edit) {
        let redacted = config.redacted();
        let output = toml::to_string_pretty(&redacted)?;
        println!("{output}");
    }

    Ok(())
}

fn edit_config(paths: &ConfigPaths) -> Result<(), ConfigError> {
    let editor = std::env::var("EDITOR")
        .map_err(|_| ConfigError::Validation("$EDITOR not set; use --set or set EDITOR".into()))?;
    let parts = split_editor_command(&editor)?;
    let (program, args) = parts
        .split_first()
        .ok_or_else(|| ConfigError::Validation("$EDITOR is empty".into()))?;
    let status = Command::new(program)
        .args(args)
        .arg(&paths.config_path)
        .status()
        .map_err(ConfigError::Io)?;
    if !status.success() {
        return Err(ConfigError::Validation(
            "editor exited with a non-zero status".into(),
        ));
    }
    Ok(())
}

fn split_editor_command(editor: &str) -> Result<Vec<String>, ConfigError> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = editor.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '\'' if !in_double => {
                in_single = !in_single;
            }
            '"' if !in_single => {
                in_double = !in_double;
            }
            '\\' if !in_single => {
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            }
            ch if ch.is_whitespace() && !in_single && !in_double => {
                if !current.is_empty() {
                    parts.push(current.clone());
                    current.clear();
                }
            }
            _ => current.push(ch),
        }
    }

    if in_single || in_double {
        return Err(ConfigError::Validation(
            "$EDITOR has unmatched quotes".into(),
        ));
    }
    if !current.is_empty() {
        parts.push(current);
    }

    if parts.is_empty() {
        return Err(ConfigError::Validation("$EDITOR is empty".into()));
    }

    Ok(parts)
}

fn list_audio_inputs() {
    let devices = list_input_devices();
    if devices.is_empty() {
        println!("no audio input devices found");
        return;
    }

    println!("audio input devices:");
    for device in devices {
        if device.is_default {
            println!("- {} (default)\n  id: {}", device.name, device.id);
        } else {
            println!("- {}\n  id: {}", device.name, device.id);
        }
    }
    println!("set with: koe config --set audio.microphone_device_id=DEVICE_ID");
}

fn apply_set(config: &mut Config, assignment: &str) -> Result<(), ConfigError> {
    let (key, value) = assignment
        .split_once('=')
        .ok_or_else(|| ConfigError::Validation("expected key=value for --set".into()))?;
    let value = value.trim();
    match key {
        "audio.sample_rate" => {
            let parsed = parse_u32(value, key)?;
            if parsed == 0 {
                return Err(ConfigError::Validation(
                    "audio.sample_rate must be greater than 0".into(),
                ));
            }
            config.audio.sample_rate = parsed;
        }
        "audio.channels" => {
            let parsed = parse_u16(value, key)?;
            if parsed == 0 {
                return Err(ConfigError::Validation(
                    "audio.channels must be greater than 0".into(),
                ));
            }
            config.audio.channels = parsed;
        }
        "audio.sources" => {
            let sources = parse_sources(value)?;
            config.audio.sources = sources;
        }
        "audio.microphone_device_id" => {
            config.audio.microphone_device_id = value.to_string();
        }
        "audio.mixdown.agc.enabled" => {
            config.audio.mixdown.agc.enabled = parse_bool(value, key)?;
        }
        "audio.mixdown.agc.target_rms_dbfs" => {
            config.audio.mixdown.agc.target_rms_dbfs = parse_f32(value, key)?;
        }
        "audio.mixdown.agc.max_gain_db" => {
            config.audio.mixdown.agc.max_gain_db = parse_f32(value, key)?;
        }
        "audio.mixdown.agc.min_gain_db" => {
            config.audio.mixdown.agc.min_gain_db = parse_f32(value, key)?;
        }
        "audio.mixdown.agc.attack_ms" => {
            config.audio.mixdown.agc.attack_ms = parse_u32(value, key)?;
        }
        "audio.mixdown.agc.release_ms" => {
            config.audio.mixdown.agc.release_ms = parse_u32(value, key)?;
        }
        "audio.mixdown.agc.limiter_ceiling_dbfs" => {
            config.audio.mixdown.agc.limiter_ceiling_dbfs = parse_f32(value, key)?;
        }
        "audio.mixdown.denoise.enabled" => {
            config.audio.mixdown.denoise.enabled = parse_bool(value, key)?;
        }
        "audio.mixdown.denoise.threshold_dbfs" => {
            config.audio.mixdown.denoise.threshold_dbfs = parse_f32(value, key)?;
        }
        "audio.mixdown.denoise.reduction_db" => {
            config.audio.mixdown.denoise.reduction_db = parse_f32(value, key)?;
        }
        "audio.mixdown.denoise.attack_ms" => {
            config.audio.mixdown.denoise.attack_ms = parse_u32(value, key)?;
        }
        "audio.mixdown.denoise.release_ms" => {
            config.audio.mixdown.denoise.release_ms = parse_u32(value, key)?;
        }
        "audio.mixdown.high_pass.enabled" => {
            config.audio.mixdown.high_pass.enabled = parse_bool(value, key)?;
        }
        "audio.mixdown.high_pass.cutoff_hz" => {
            config.audio.mixdown.high_pass.cutoff_hz = parse_f32(value, key)?;
        }
        "transcribe.active" => {
            config.transcribe.active = value.to_string();
        }
        "transcribe.local.provider" => {
            config.transcribe.local.provider = value.to_string();
        }
        "transcribe.local.model" => {
            config.transcribe.local.model = value.to_string();
        }
        "transcribe.local.api_key" => {
            config.transcribe.local.api_key = value.to_string();
        }
        "transcribe.cloud.provider" => {
            config.transcribe.cloud.provider = value.to_string();
        }
        "transcribe.cloud.model" => {
            config.transcribe.cloud.model = value.to_string();
        }
        "transcribe.cloud.api_key" => {
            config.transcribe.cloud.api_key = value.to_string();
        }
        "transcribe.provider" => {
            set_active_provider(
                "transcribe.provider",
                &config.transcribe.active,
                value,
                &mut config.transcribe.local.provider,
                &mut config.transcribe.cloud.provider,
            )?;
        }
        "transcribe.model" => {
            set_active_value(
                "transcribe.model",
                &config.transcribe.active,
                value,
                &mut config.transcribe.local.model,
                &mut config.transcribe.cloud.model,
            )?;
        }
        "transcribe.api_key" => {
            set_active_value(
                "transcribe.api_key",
                &config.transcribe.active,
                value,
                &mut config.transcribe.local.api_key,
                &mut config.transcribe.cloud.api_key,
            )?;
        }
        "summarize.active" => {
            config.summarize.active = value.to_string();
        }
        "summarize.local.provider" => {
            config.summarize.local.provider = value.to_string();
        }
        "summarize.local.model" => {
            config.summarize.local.model = value.to_string();
        }
        "summarize.local.api_key" => {
            config.summarize.local.api_key = value.to_string();
        }
        "summarize.cloud.provider" => {
            config.summarize.cloud.provider = value.to_string();
        }
        "summarize.cloud.model" => {
            config.summarize.cloud.model = value.to_string();
        }
        "summarize.cloud.api_key" => {
            config.summarize.cloud.api_key = value.to_string();
        }
        "summarize.provider" => {
            set_active_provider(
                "summarize.provider",
                &config.summarize.active,
                value,
                &mut config.summarize.local.provider,
                &mut config.summarize.cloud.provider,
            )?;
        }
        "summarize.model" => {
            set_active_value(
                "summarize.model",
                &config.summarize.active,
                value,
                &mut config.summarize.local.model,
                &mut config.summarize.cloud.model,
            )?;
        }
        "summarize.api_key" => {
            set_active_value(
                "summarize.api_key",
                &config.summarize.active,
                value,
                &mut config.summarize.local.api_key,
                &mut config.summarize.cloud.api_key,
            )?;
        }
        "summarize.prompt_profile" => {
            config.summarize.prompt_profile = value.to_string();
        }
        "session.context" => {
            config.session.context = value.to_string();
        }
        "session.participants" => {
            config.session.participants = parse_participants(value)?;
        }
        "session.export_dir" => {
            config.session.export_dir = value.to_string();
        }
        "ui.show_transcript" => {
            config.ui.show_transcript = parse_bool(value, key)?;
        }
        "ui.notes_only_default" => {
            config.ui.notes_only_default = parse_bool(value, key)?;
        }
        "ui.color_theme" => {
            config.ui.color_theme = value.to_string();
        }
        _ => {
            return Err(ConfigError::Validation(format!(
                "unknown config key: {key}"
            )));
        }
    }
    Ok(())
}

fn set_active_value(
    label: &str,
    active: &str,
    value: &str,
    local: &mut String,
    cloud: &mut String,
) -> Result<(), ConfigError> {
    match active {
        "local" => {
            *local = value.to_string();
            Ok(())
        }
        "cloud" => {
            *cloud = value.to_string();
            Ok(())
        }
        _ => Err(ConfigError::Validation(format!(
            "{label} cannot be set because active profile is invalid"
        ))),
    }
}

fn set_active_provider(
    label: &str,
    active: &str,
    value: &str,
    local: &mut String,
    cloud: &mut String,
) -> Result<(), ConfigError> {
    set_active_value(label, active, value, local, cloud)
}

fn parse_bool(value: &str, key: &str) -> Result<bool, ConfigError> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(ConfigError::Validation(format!(
            "{key} expects true or false"
        ))),
    }
}

fn parse_u32(value: &str, key: &str) -> Result<u32, ConfigError> {
    value
        .parse()
        .map_err(|_| ConfigError::Validation(format!("{key} expects an unsigned integer")))
}

fn parse_u16(value: &str, key: &str) -> Result<u16, ConfigError> {
    value
        .parse()
        .map_err(|_| ConfigError::Validation(format!("{key} expects an unsigned integer")))
}

fn parse_f32(value: &str, key: &str) -> Result<f32, ConfigError> {
    value
        .parse()
        .map_err(|_| ConfigError::Validation(format!("{key} expects a number")))
}

fn parse_sources(value: &str) -> Result<Vec<String>, ConfigError> {
    let sources: Vec<String> = value
        .split(',')
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect();
    if sources.is_empty() {
        return Err(ConfigError::Validation(
            "audio.sources must include at least one value".into(),
        ));
    }
    for source in &sources {
        match source.as_str() {
            "system" | "microphone" | "mixed" => {}
            other => {
                return Err(ConfigError::Validation(format!(
                    "audio.sources includes invalid value {other}"
                )));
            }
        }
    }
    Ok(sources)
}

fn parse_participants(value: &str) -> Result<Vec<String>, ConfigError> {
    if value.trim().is_empty() {
        return Ok(Vec::new());
    }
    let participants: Vec<String> = value
        .split(',')
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect();
    if participants.is_empty() {
        return Err(ConfigError::Validation(
            "session.participants must include at least one name".into(),
        ));
    }
    Ok(participants)
}

#[cfg(test)]
mod tests {
    use super::split_editor_command;

    #[test]
    fn split_editor_command_handles_args() {
        let parts = split_editor_command("code --wait").unwrap();
        assert_eq!(parts, vec!["code", "--wait"]);
    }

    #[test]
    fn split_editor_command_handles_quotes() {
        let parts = split_editor_command("\"/Applications/VS Code\" --wait").unwrap();
        assert_eq!(parts, vec!["/Applications/VS Code", "--wait"]);
    }

    #[test]
    fn split_editor_command_rejects_unmatched_quotes() {
        let err = split_editor_command("\"unterminated").unwrap_err();
        assert!(err.to_string().contains("unmatched quotes"));
    }
}
