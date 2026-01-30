use clap::Args;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

const DEFAULT_MODEL: &str = "base.en";
const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

#[derive(Args, Debug, Clone)]
pub struct InitArgs {
    /// Whisper model name (e.g. base.en, small, large-v3-turbo)
    #[arg(long, default_value = DEFAULT_MODEL)]
    pub model: String,

    /// Override models directory (default: ~/.koe/models)
    #[arg(long)]
    pub dir: Option<PathBuf>,

    /// Force re-download even if the model already exists
    #[arg(long)]
    pub force: bool,

    /// Write GROQ_API_KEY to .env (defaults to GROQ_API_KEY from current env)
    #[arg(long)]
    pub groq_key: Option<String>,
}

pub fn run(args: &InitArgs) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let models_dir = resolve_models_dir(args.dir.as_ref());
    fs::create_dir_all(&models_dir)?;

    let model_file = model_filename(&args.model);
    let dest = models_dir.join(model_file);

    if dest.exists() && !args.force {
        println!("model already present at {}", dest.display());
        write_env_file(&dest, args)?;
        return Ok(dest);
    }

    let url = format!(
        "{MODEL_BASE_URL}/{}",
        dest.file_name().unwrap().to_string_lossy()
    );
    println!("downloading model from {url}");
    download_to_path(&url, &dest)?;
    println!("model saved to {}", dest.display());

    write_env_file(&dest, args)?;
    Ok(dest)
}

fn resolve_models_dir(dir: Option<&PathBuf>) -> PathBuf {
    if let Some(override_dir) = dir {
        return override_dir.to_path_buf();
    }
    if let Ok(env_dir) = std::env::var("KOE_MODELS_DIR") {
        return PathBuf::from(env_dir);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".koe").join("models");
    }
    PathBuf::from("models")
}

fn model_filename(model: &str) -> String {
    if model.ends_with(".bin") {
        model.to_string()
    } else {
        format!("ggml-{model}.bin")
    }
}

fn download_to_path(url: &str, dest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let response = ureq::get(url).call().map_err(|e| {
        let msg = format!("model download failed: {e}");
        io::Error::other(msg)
    })?;

    let tmp_path = dest.with_extension("download");
    let mut reader = response.into_body().into_reader();
    let mut file = File::create(&tmp_path)?;
    io::copy(&mut reader, &mut file)?;
    file.sync_all()?;
    fs::rename(tmp_path, dest)?;
    Ok(())
}

fn write_env_file(model_path: &Path, args: &InitArgs) -> Result<(), io::Error> {
    let mut lines = match fs::read_to_string(".env") {
        Ok(contents) => contents.lines().map(|l| l.to_string()).collect(),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(err) => return Err(err),
    };

    let env_groq_key = std::env::var("GROQ_API_KEY").ok();
    let groq_key = match &args.groq_key {
        Some(key) => Some(key.as_str()),
        None => env_groq_key.as_deref(),
    };

    upsert_env_var(
        &mut lines,
        "KOE_WHISPER_MODEL",
        model_path.to_string_lossy().as_ref(),
    );

    if let Some(key) = groq_key {
        upsert_env_var(&mut lines, "GROQ_API_KEY", key);
    }

    let content = if lines.is_empty() {
        String::new()
    } else {
        let mut joined = lines.join("\n");
        joined.push('\n');
        joined
    };
    fs::write(".env", content)?;
    Ok(())
}

fn upsert_env_var(lines: &mut Vec<String>, key: &str, value: &str) {
    let prefix = format!("{key}=");
    for line in lines.iter_mut() {
        if line.starts_with(&prefix) {
            *line = format!("{key}={value}");
            return;
        }
    }
    lines.push(format!("{key}={value}"));
}
