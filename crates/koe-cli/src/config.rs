use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;

const CONFIG_VERSION: u32 = 1;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("home directory not found; set HOME")]
    HomeMissing,
    #[error("config io error: {0}")]
    Io(#[from] io::Error),
    #[error("config parse error: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("config serialize error: {0}")]
    Serialize(#[from] toml::ser::Error),
    #[error("config validation error: {0}")]
    Validation(String),
}

#[derive(Debug, Clone)]
pub struct ConfigPaths {
    pub base_dir: PathBuf,
    pub config_path: PathBuf,
    pub models_dir: PathBuf,
    pub sessions_dir: PathBuf,
}

impl ConfigPaths {
    pub fn from_home() -> Result<Self, ConfigError> {
        let home = std::env::var("HOME").map_err(|_| ConfigError::HomeMissing)?;
        Ok(Self::from_base(PathBuf::from(home).join(".koe")))
    }

    pub fn from_base(base_dir: PathBuf) -> Self {
        let config_path = base_dir.join("config.toml");
        let models_dir = base_dir.join("models");
        let sessions_dir = base_dir.join("sessions");
        Self {
            base_dir,
            config_path,
            models_dir,
            sessions_dir,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub version: u32,
    pub audio: AudioConfig,
    pub asr: AsrConfig,
    pub summarizer: SummarizerConfig,
    pub session: SessionConfig,
    pub ui: UiConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            audio: AudioConfig::default(),
            asr: AsrConfig::default(),
            summarizer: SummarizerConfig::default(),
            session: SessionConfig::default(),
            ui: UiConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub sources: Vec<String>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 1,
            sources: vec!["system".to_string(), "microphone".to_string()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AsrConfig {
    pub provider: String,
    pub model: String,
    pub api_key: String,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            provider: "whisper".to_string(),
            model: "base.en".to_string(),
            api_key: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SummarizerConfig {
    pub provider: String,
    pub model: String,
    pub api_key: String,
    pub prompt_profile: String,
}

impl Default for SummarizerConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            model: "qwen3:30b-a3b".to_string(),
            api_key: String::new(),
            prompt_profile: "minimal".to_string(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionConfig {
    pub context: String,
    pub participants: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    pub show_transcript: bool,
    pub notes_only_default: bool,
    pub color_theme: String,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            show_transcript: false,
            notes_only_default: true,
            color_theme: "minimal".to_string(),
        }
    }
}

impl Config {
    pub fn load_or_create(paths: &ConfigPaths) -> Result<Self, ConfigError> {
        ensure_dirs(paths)?;
        if paths.config_path.exists() {
            let config = Self::load(paths)?;
            return Ok(config);
        }

        let config = Self::default();
        Self::write(paths, &config)?;
        Ok(config)
    }

    pub fn load(paths: &ConfigPaths) -> Result<Self, ConfigError> {
        ensure_dirs(paths)?;
        let content = fs::read_to_string(&paths.config_path)?;
        let raw: toml::Value = toml::from_str(&content)?;
        let file_version = raw
            .get("version")
            .and_then(|value| value.as_integer())
            .unwrap_or(0) as u32;

        let mut config: Config = toml::from_str(&content)?;
        let mut migrated = false;

        if file_version < CONFIG_VERSION {
            config.version = CONFIG_VERSION;
            migrated = true;
        } else if file_version > CONFIG_VERSION {
            eprintln!(
                "config version {file_version} is newer than supported {CONFIG_VERSION}; proceeding"
            );
        }

        warn_if_loose_permissions(&paths.config_path)?;

        if migrated {
            Self::write(paths, &config)?;
        }

        Ok(config)
    }

    pub fn write(paths: &ConfigPaths, config: &Config) -> Result<(), ConfigError> {
        ensure_dirs(paths)?;
        let content = toml::to_string_pretty(config)?;
        write_atomic(&paths.config_path, content.as_bytes())?;
        Ok(())
    }

    pub fn redacted(&self) -> Self {
        let mut redacted = self.clone();
        if !redacted.asr.api_key.trim().is_empty() {
            redacted.asr.api_key = "<redacted>".to_string();
        }
        if !redacted.summarizer.api_key.trim().is_empty() {
            redacted.summarizer.api_key = "<redacted>".to_string();
        }
        redacted
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        match self.asr.provider.as_str() {
            "whisper" | "groq" => {}
            other => {
                return Err(ConfigError::Validation(format!(
                    "asr.provider must be whisper or groq (got {other})"
                )));
            }
        }

        match self.summarizer.provider.as_str() {
            "ollama" | "openrouter" => {}
            other => {
                return Err(ConfigError::Validation(format!(
                    "summarizer.provider must be ollama or openrouter (got {other})"
                )));
            }
        }

        if self.audio.sample_rate == 0 {
            return Err(ConfigError::Validation(
                "audio.sample_rate must be greater than 0".into(),
            ));
        }
        if self.audio.channels == 0 {
            return Err(ConfigError::Validation(
                "audio.channels must be greater than 0".into(),
            ));
        }
        if self.audio.sources.is_empty() {
            return Err(ConfigError::Validation(
                "audio.sources must include at least one source".into(),
            ));
        }
        for source in &self.audio.sources {
            match source.as_str() {
                "system" | "microphone" | "mixed" => {}
                other => {
                    return Err(ConfigError::Validation(format!(
                        "audio.sources includes invalid value {other}"
                    )));
                }
            }
        }

        if self.asr.model.trim().is_empty() {
            return Err(ConfigError::Validation(
                "asr.model must not be empty".into(),
            ));
        }
        if self.asr.provider == "whisper"
            && looks_like_path(self.asr.model.as_str())
            && !Path::new(&self.asr.model).exists()
        {
            return Err(ConfigError::Validation(format!(
                "asr.model path not found: {}",
                self.asr.model
            )));
        }
        if self.asr.provider == "groq" && self.asr.api_key.trim().is_empty() {
            return Err(ConfigError::Validation(
                "asr.api_key required when asr.provider=groq".into(),
            ));
        }

        if self.summarizer.model.trim().is_empty() {
            return Err(ConfigError::Validation(
                "summarizer.model must not be empty".into(),
            ));
        }
        if self.summarizer.prompt_profile.trim().is_empty() {
            return Err(ConfigError::Validation(
                "summarizer.prompt_profile must not be empty".into(),
            ));
        }
        if self.summarizer.provider == "openrouter" && self.summarizer.api_key.trim().is_empty() {
            return Err(ConfigError::Validation(
                "summarizer.api_key required when summarizer.provider=openrouter".into(),
            ));
        }
        if self.ui.color_theme.trim().is_empty() {
            return Err(ConfigError::Validation(
                "ui.color_theme must not be empty".into(),
            ));
        }
        for participant in &self.session.participants {
            if participant.trim().is_empty() {
                return Err(ConfigError::Validation(
                    "session.participants entries must not be empty".into(),
                ));
            }
        }

        Ok(())
    }
}

fn ensure_dirs(paths: &ConfigPaths) -> Result<(), ConfigError> {
    fs::create_dir_all(&paths.base_dir)?;
    fs::create_dir_all(&paths.models_dir)?;
    fs::create_dir_all(&paths.sessions_dir)?;
    Ok(())
}

fn write_atomic(path: &Path, contents: &[u8]) -> Result<(), ConfigError> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("config path missing parent directory"))?;
    let tmp_path = parent.join("config.toml.tmp");
    fs::write(&tmp_path, contents)?;
    set_strict_permissions(&tmp_path)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

fn set_strict_permissions(path: &Path) -> Result<(), ConfigError> {
    #[cfg(unix)]
    {
        let perm = fs::Permissions::from_mode(0o600);
        fs::set_permissions(path, perm)?;
    }
    Ok(())
}

fn warn_if_loose_permissions(path: &Path) -> Result<(), ConfigError> {
    #[cfg(unix)]
    {
        let metadata = fs::metadata(path)?;
        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            eprintln!(
                "config file {} is group/world readable; set permissions to 0600",
                path.display()
            );
        }
    }
    Ok(())
}

fn looks_like_path(value: &str) -> bool {
    value.ends_with(".bin") || value.contains('/') || value.contains(std::path::MAIN_SEPARATOR)
}

#[cfg(test)]
mod tests {
    use super::{CONFIG_VERSION, Config, ConfigPaths};
    use std::fs;

    #[test]
    fn load_or_create_writes_defaults_and_dirs() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("koe");
        let paths = ConfigPaths::from_base(base);
        let config = Config::load_or_create(&paths).unwrap();

        assert!(paths.config_path.exists());
        assert!(paths.models_dir.is_dir());
        assert!(paths.sessions_dir.is_dir());
        assert_eq!(config.version, CONFIG_VERSION);
        assert_eq!(config.asr.provider, "whisper");
        assert_eq!(config.summarizer.provider, "ollama");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&paths.config_path)
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    fn load_migrates_missing_fields() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("koe");
        let paths = ConfigPaths::from_base(base);
        fs::create_dir_all(&paths.base_dir).unwrap();
        let content = r#"version = 0

[asr]
provider = "whisper"
model = "base.en"
api_key = ""
"#;
        fs::write(&paths.config_path, content).unwrap();

        let config = Config::load(&paths).unwrap();
        assert_eq!(config.version, CONFIG_VERSION);
        assert_eq!(config.ui.color_theme, "minimal");

        let updated = fs::read_to_string(&paths.config_path).unwrap();
        assert!(updated.contains("version = 1"));
        assert!(updated.contains("[summarizer]"));
    }

    #[test]
    fn redacted_hides_api_keys() {
        let mut config = Config::default();
        config.asr.api_key = "secret".to_string();
        config.summarizer.api_key = "secret2".to_string();
        let redacted = config.redacted();
        assert_eq!(redacted.asr.api_key, "<redacted>");
        assert_eq!(redacted.summarizer.api_key, "<redacted>");
    }

    #[test]
    fn validate_rejects_bad_provider() {
        let mut config = Config::default();
        config.asr.provider = "bad".to_string();
        assert!(config.validate().is_err());
    }
}
