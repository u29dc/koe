use crate::config::{Config, ConfigError, ConfigPaths};
use clap::Args;
use std::process::Command;

#[derive(Args, Debug, Clone)]
pub struct ConfigArgs {
    /// Print config with secrets redacted
    #[arg(long)]
    pub print: bool,

    /// Edit config in $EDITOR
    #[arg(long)]
    pub edit: bool,

    /// Set a config value (dotted key=value)
    #[arg(long, value_name = "key=value")]
    pub set: Vec<String>,
}

pub fn run(args: &ConfigArgs, paths: &ConfigPaths) -> Result<(), ConfigError> {
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
    let status = Command::new(editor)
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
        "asr.provider" => {
            config.asr.provider = value.to_string();
        }
        "asr.model" => {
            config.asr.model = value.to_string();
        }
        "asr.api_key" => {
            config.asr.api_key = value.to_string();
        }
        "summarizer.provider" => {
            config.summarizer.provider = value.to_string();
        }
        "summarizer.model" => {
            config.summarizer.model = value.to_string();
        }
        "summarizer.api_key" => {
            config.summarizer.api_key = value.to_string();
        }
        "summarizer.prompt_profile" => {
            config.summarizer.prompt_profile = value.to_string();
        }
        "session.context" => {
            config.session.context = value.to_string();
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
