use crate::config::{Config, ConfigError, ConfigPaths, ProviderConfig};
use clap::Args;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const DEFAULT_WHISPER_MODEL: &str = "base.en";
const DEFAULT_GROQ_MODEL: &str = "whisper-large-v3-turbo";
const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

struct ModelOption {
    name: &'static str,
    size: &'static str,
}

const WHISPER_MODELS: &[ModelOption] = &[
    ModelOption {
        name: "base.en",
        size: "~142 MB",
    },
    ModelOption {
        name: "small",
        size: "~466 MB",
    },
    ModelOption {
        name: "medium",
        size: "~1.5 GB",
    },
    ModelOption {
        name: "large-v3-turbo",
        size: "~1.5 GB",
    },
];

#[derive(Args, Debug, Clone)]
pub struct InitArgs {
    /// Overwrite existing config values
    #[arg(long)]
    pub force: bool,
}

#[derive(Debug, Error)]
pub enum InitError {
    #[error("config error: {0}")]
    Config(#[from] ConfigError),
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("init failed: {0}")]
    Message(String),
}

pub fn run(args: &InitArgs, paths: &ConfigPaths) -> Result<(), InitError> {
    print_permissions();

    let mut config = Config::load_or_create(paths)?;

    let mut changed = Vec::new();
    let mut kept = Vec::new();

    let current_transcribe_active = if args.force {
        ""
    } else {
        config.transcribe.active.as_str()
    };
    let transcribe_active = prompt_provider(
        "Transcribe mode",
        &["local", "cloud"],
        current_transcribe_active,
    )?;
    track_update(
        &mut config.transcribe.active,
        transcribe_active,
        "transcribe.active",
        &mut changed,
        &mut kept,
        args.force,
    );

    let configure_all_transcribe = args.force;
    let active_transcribe = config.transcribe.active == "local";
    if active_transcribe || configure_all_transcribe {
        configure_transcribe_profile(
            "transcribe.local",
            &mut config.transcribe.local,
            paths,
            args,
            &mut changed,
            &mut kept,
        )?;
    }
    if !active_transcribe || configure_all_transcribe {
        configure_transcribe_profile(
            "transcribe.cloud",
            &mut config.transcribe.cloud,
            paths,
            args,
            &mut changed,
            &mut kept,
        )?;
    }

    let current_summarize_active = if args.force {
        ""
    } else {
        config.summarize.active.as_str()
    };
    let summarize_active = prompt_provider(
        "Summarize mode",
        &["local", "cloud"],
        current_summarize_active,
    )?;
    track_update(
        &mut config.summarize.active,
        summarize_active,
        "summarize.active",
        &mut changed,
        &mut kept,
        args.force,
    );

    let configure_all_summarize = args.force;
    let active_summarize = config.summarize.active == "local";
    if active_summarize || configure_all_summarize {
        configure_summarize_profile(
            "summarize.local",
            &mut config.summarize.local,
            args,
            &mut changed,
            &mut kept,
        )?;
    }
    if !active_summarize || configure_all_summarize {
        configure_summarize_profile(
            "summarize.cloud",
            &mut config.summarize.cloud,
            args,
            &mut changed,
            &mut kept,
        )?;
    }

    config.validate()?;
    Config::write(paths, &config)?;

    print_summary(&changed, &kept);
    println!("next: koe");

    Ok(())
}

pub fn download_model(model: &str, models_dir: &Path, force: bool) -> Result<PathBuf, InitError> {
    fs::create_dir_all(models_dir)?;
    let model_file = model_filename(model);
    let dest = models_dir.join(model_file);

    if dest.exists() && !force {
        println!("model already present at {}", dest.display());
        return Ok(dest);
    }

    let url = format!(
        "{MODEL_BASE_URL}/{}",
        dest.file_name().unwrap().to_string_lossy()
    );
    println!("downloading model from {url}");
    download_to_path(&url, &dest)?;
    println!("model saved to {}", dest.display());
    Ok(dest)
}

fn print_permissions() {
    println!("permissions required:");
    println!("System Settings → Privacy & Security → Screen Recording: allow koe");
    println!("System Settings → Privacy & Security → Microphone: allow koe");
    println!("restart koe after granting permissions");
    println!("checklist: [ ] screen recording  [ ] microphone\n");
}

fn print_summary(changed: &[String], kept: &[String]) {
    if !changed.is_empty() {
        println!("updated:");
        for item in changed {
            println!("- {item}");
        }
    }
    if !kept.is_empty() {
        println!("kept:");
        for item in kept {
            println!("- {item}");
        }
    }
}

fn prompt_provider(prompt: &str, options: &[&str], current: &str) -> Result<String, InitError> {
    loop {
        println!("{prompt}:");
        for (idx, option) in options.iter().enumerate() {
            if *option == current {
                println!("  {}) {} (current)", idx + 1, option);
            } else {
                println!("  {}) {}", idx + 1, option);
            }
        }
        let default_index = options
            .iter()
            .position(|option| *option == current)
            .unwrap_or(0);
        let selection = prompt_line(&format!("select [default {}]: ", default_index + 1))?;
        let trimmed = selection.trim();
        if trimmed.is_empty() {
            return Ok(options[default_index].to_string());
        }
        if let Ok(choice) = trimmed.parse::<usize>() {
            if choice >= 1 && choice <= options.len() {
                return Ok(options[choice - 1].to_string());
            }
        }
        println!("invalid selection, try again");
    }
}

fn prompt_model_choice(current: &str) -> Result<String, InitError> {
    println!("whisper model (sizes are approximate):");
    for (idx, option) in WHISPER_MODELS.iter().enumerate() {
        let label = format!("{} ({})", option.name, option.size);
        if option.name == current {
            println!("  {}) {} (current)", idx + 1, label);
        } else {
            println!("  {}) {}", idx + 1, label);
        }
    }
    let default_index = WHISPER_MODELS
        .iter()
        .position(|option| option.name == current)
        .unwrap_or(0);

    loop {
        let selection = prompt_line(&format!("select [default {}]: ", default_index + 1))?;
        let trimmed = selection.trim();
        if trimmed.is_empty() {
            return Ok(WHISPER_MODELS[default_index].name.to_string());
        }
        if let Ok(choice) = trimmed.parse::<usize>() {
            if choice >= 1 && choice <= WHISPER_MODELS.len() {
                return Ok(WHISPER_MODELS[choice - 1].name.to_string());
            }
        }
        println!("invalid selection, try again");
    }
}

fn prompt_with_default(prompt: &str, current: &str, fallback: &str) -> Result<String, InitError> {
    let default = if current.trim().is_empty() {
        fallback
    } else {
        current
    };
    let input = prompt_line(&format!("{prompt} [default {default}]: "))?;
    let trimmed = input.trim();
    if trimmed.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

fn prompt_secret(prompt: &str, current: &str, force: bool) -> Result<String, InitError> {
    loop {
        let hint = if !current.trim().is_empty() && !force {
            "leave blank to keep current"
        } else {
            "required"
        };
        let input = prompt_line(&format!("{prompt} ({hint}): "))?;
        let trimmed = input.trim();
        if trimmed.is_empty() {
            if !current.trim().is_empty() && !force {
                return Ok(current.to_string());
            }
            println!("value required");
            continue;
        }
        return Ok(trimmed.to_string());
    }
}

fn prompt_line(prompt: &str) -> Result<String, InitError> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut input = String::new();
    let bytes = io::stdin().read_line(&mut input)?;
    if bytes == 0 {
        return Err(InitError::Message("no input received".into()));
    }
    Ok(input)
}

fn track_update(
    current: &mut String,
    next: String,
    label: &str,
    changed: &mut Vec<String>,
    kept: &mut Vec<String>,
    force: bool,
) {
    if !force && *current == next {
        kept.push(label.to_string());
        return;
    }
    if *current != next {
        *current = next;
        changed.push(label.to_string());
    } else {
        kept.push(label.to_string());
    }
}

fn configure_transcribe_profile(
    label: &str,
    profile: &mut ProviderConfig,
    paths: &ConfigPaths,
    args: &InitArgs,
    changed: &mut Vec<String>,
    kept: &mut Vec<String>,
) -> Result<(), InitError> {
    let current_provider = if args.force {
        ""
    } else {
        profile.provider.as_str()
    };
    let provider = prompt_provider(
        &format!("{label} provider"),
        &["whisper", "groq"],
        current_provider,
    )?;
    track_update(
        &mut profile.provider,
        provider,
        &format!("{label}.provider"),
        changed,
        kept,
        args.force,
    );

    if profile.provider == "whisper" {
        let current_model = if args.force {
            None
        } else {
            current_whisper_model_name(profile.model.as_str())
        };
        let model_choice =
            prompt_model_choice(current_model.as_deref().unwrap_or(DEFAULT_WHISPER_MODEL))?;
        let model_path = download_model(&model_choice, &paths.models_dir, args.force)?;
        track_update(
            &mut profile.model,
            model_path.to_string_lossy().to_string(),
            &format!("{label}.model"),
            changed,
            kept,
            args.force,
        );
    } else {
        let current_groq_model = if args.force {
            ""
        } else {
            profile.model.as_str()
        };
        let groq_model = prompt_with_default("Groq model", current_groq_model, DEFAULT_GROQ_MODEL)?;
        track_update(
            &mut profile.model,
            groq_model,
            &format!("{label}.model"),
            changed,
            kept,
            args.force,
        );
        let groq_key = prompt_secret("Groq API key", &profile.api_key, args.force)?;
        track_update(
            &mut profile.api_key,
            groq_key,
            &format!("{label}.api_key"),
            changed,
            kept,
            args.force,
        );
    }
    Ok(())
}

fn configure_summarize_profile(
    label: &str,
    profile: &mut ProviderConfig,
    args: &InitArgs,
    changed: &mut Vec<String>,
    kept: &mut Vec<String>,
) -> Result<(), InitError> {
    let current_provider = if args.force {
        ""
    } else {
        profile.provider.as_str()
    };
    let provider = prompt_provider(
        &format!("{label} provider"),
        &["ollama", "openrouter"],
        current_provider,
    )?;
    track_update(
        &mut profile.provider,
        provider,
        &format!("{label}.provider"),
        changed,
        kept,
        args.force,
    );

    if profile.provider == "ollama" {
        let current_model = if args.force {
            ""
        } else {
            profile.model.as_str()
        };
        let model = prompt_with_default("Ollama model tag", current_model, "qwen3:30b-a3b")?;
        track_update(
            &mut profile.model,
            model,
            &format!("{label}.model"),
            changed,
            kept,
            args.force,
        );
    } else {
        let current_model = if args.force {
            ""
        } else {
            profile.model.as_str()
        };
        let model =
            prompt_with_default("OpenRouter model", current_model, "google/gemini-2.5-flash")?;
        track_update(
            &mut profile.model,
            model,
            &format!("{label}.model"),
            changed,
            kept,
            args.force,
        );
        let key = prompt_secret("OpenRouter API key", &profile.api_key, args.force)?;
        track_update(
            &mut profile.api_key,
            key,
            &format!("{label}.api_key"),
            changed,
            kept,
            args.force,
        );
    }
    Ok(())
}

fn current_whisper_model_name(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let filename = Path::new(trimmed)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())?;
    let without_prefix = filename.strip_prefix("ggml-").unwrap_or(&filename);
    let without_suffix = without_prefix
        .strip_suffix(".bin")
        .unwrap_or(without_prefix);
    Some(without_suffix.to_string())
}

fn model_filename(model: &str) -> String {
    if model.ends_with(".bin") {
        model.to_string()
    } else {
        format!("ggml-{model}.bin")
    }
}

fn download_to_path(url: &str, dest: &Path) -> Result<(), InitError> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| InitError::Message(format!("model download failed: {e}")))?;

    let tmp_path = dest.with_extension("download");
    let mut reader = response.into_body().into_reader();
    let mut file = File::create(&tmp_path)?;
    io::copy(&mut reader, &mut file)?;
    file.sync_all()?;
    fs::rename(tmp_path, dest)?;
    Ok(())
}
