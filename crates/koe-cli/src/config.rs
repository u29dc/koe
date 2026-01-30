use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;

const CONFIG_VERSION: u32 = 2;

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
    pub microphone_device_id: String,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 1,
            sources: vec!["system".to_string(), "microphone".to_string()],
            microphone_device_id: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AsrConfig {
    pub active: String,
    pub local: ProviderConfig,
    pub cloud: ProviderConfig,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            active: "local".to_string(),
            local: ProviderConfig {
                provider: "whisper".to_string(),
                model: "base.en".to_string(),
                api_key: String::new(),
            },
            cloud: ProviderConfig {
                provider: "groq".to_string(),
                model: "whisper-large-v3-turbo".to_string(),
                api_key: String::new(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SummarizerConfig {
    pub active: String,
    pub local: ProviderConfig,
    pub cloud: ProviderConfig,
    pub prompt_profile: String,
}

impl Default for SummarizerConfig {
    fn default() -> Self {
        Self {
            active: "local".to_string(),
            local: ProviderConfig {
                provider: "ollama".to_string(),
                model: "qwen3:30b-a3b".to_string(),
                api_key: String::new(),
            },
            cloud: ProviderConfig {
                provider: "openrouter".to_string(),
                model: "google/gemini-2.5-flash".to_string(),
                api_key: String::new(),
            },
            prompt_profile: "minimal".to_string(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderConfig {
    pub provider: String,
    pub model: String,
    pub api_key: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionConfig {
    pub context: String,
    pub participants: Vec<String>,
    pub export_dir: String,
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
            migrate_legacy_config(&raw, &mut config);
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
        redact_provider(&mut redacted.asr.local);
        redact_provider(&mut redacted.asr.cloud);
        redact_provider(&mut redacted.summarizer.local);
        redact_provider(&mut redacted.summarizer.cloud);
        redacted
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_active("asr.active", self.asr.active.as_str())?;
        validate_active("summarizer.active", self.summarizer.active.as_str())?;
        validate_asr_profile("asr.local", &self.asr.local, self.asr.active == "local")?;
        validate_asr_profile("asr.cloud", &self.asr.cloud, self.asr.active == "cloud")?;
        validate_summarizer_profile(
            "summarizer.local",
            &self.summarizer.local,
            self.summarizer.active == "local",
        )?;
        validate_summarizer_profile(
            "summarizer.cloud",
            &self.summarizer.cloud,
            self.summarizer.active == "cloud",
        )?;

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

        if self.summarizer.prompt_profile.trim().is_empty() {
            return Err(ConfigError::Validation(
                "summarizer.prompt_profile must not be empty".into(),
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

fn validate_active(field: &str, value: &str) -> Result<(), ConfigError> {
    match value {
        "local" | "cloud" => Ok(()),
        other => Err(ConfigError::Validation(format!(
            "{field} must be local or cloud (got {other})"
        ))),
    }
}

fn validate_asr_profile(
    label: &str,
    profile: &ProviderConfig,
    is_active: bool,
) -> Result<(), ConfigError> {
    match profile.provider.as_str() {
        "whisper" | "groq" => {}
        other => {
            return Err(ConfigError::Validation(format!(
                "{label}.provider must be whisper or groq (got {other})"
            )));
        }
    }

    if profile.model.trim().is_empty() {
        return Err(ConfigError::Validation(format!(
            "{label}.model must not be empty"
        )));
    }
    if profile.provider == "whisper"
        && looks_like_path(profile.model.as_str())
        && !Path::new(&profile.model).exists()
    {
        return Err(ConfigError::Validation(format!(
            "{label}.model path not found: {}",
            profile.model
        )));
    }
    if is_active && profile.provider == "groq" && profile.api_key.trim().is_empty() {
        return Err(ConfigError::Validation(format!(
            "{label}.api_key required when {label}.provider=groq"
        )));
    }
    Ok(())
}

fn validate_summarizer_profile(
    label: &str,
    profile: &ProviderConfig,
    is_active: bool,
) -> Result<(), ConfigError> {
    match profile.provider.as_str() {
        "ollama" | "openrouter" => {}
        other => {
            return Err(ConfigError::Validation(format!(
                "{label}.provider must be ollama or openrouter (got {other})"
            )));
        }
    }

    if profile.model.trim().is_empty() {
        return Err(ConfigError::Validation(format!(
            "{label}.model must not be empty"
        )));
    }
    if is_active && profile.provider == "openrouter" && profile.api_key.trim().is_empty() {
        return Err(ConfigError::Validation(format!(
            "{label}.api_key required when {label}.provider=openrouter"
        )));
    }
    Ok(())
}

fn redact_provider(profile: &mut ProviderConfig) {
    if !profile.api_key.trim().is_empty() {
        profile.api_key = "<redacted>".to_string();
    }
}

fn migrate_legacy_config(raw: &toml::Value, config: &mut Config) {
    let asr_table = raw.get("asr").and_then(|value| value.as_table());
    let summarizer_table = raw.get("summarizer").and_then(|value| value.as_table());

    if let Some(asr) = asr_table {
        let provider = asr
            .get("provider")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        let model = asr
            .get("model")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        let api_key = asr
            .get("api_key")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();

        if provider == "groq" {
            config.asr.cloud.provider = provider;
            if !model.is_empty() {
                config.asr.cloud.model = model;
            }
            config.asr.cloud.api_key = api_key;
            config.asr.active = "cloud".to_string();
        } else if !provider.is_empty() {
            config.asr.local.provider = provider;
            if !model.is_empty() {
                config.asr.local.model = model;
            }
            config.asr.local.api_key = api_key;
            config.asr.active = "local".to_string();
        }
    }

    if let Some(summarizer) = summarizer_table {
        let provider = summarizer
            .get("provider")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        let model = summarizer
            .get("model")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        let api_key = summarizer
            .get("api_key")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        let prompt_profile = summarizer
            .get("prompt_profile")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();

        if !prompt_profile.is_empty() {
            config.summarizer.prompt_profile = prompt_profile;
        }

        if provider == "openrouter" {
            config.summarizer.cloud.provider = provider;
            if !model.is_empty() {
                config.summarizer.cloud.model = model;
            }
            config.summarizer.cloud.api_key = api_key;
            config.summarizer.active = "cloud".to_string();
        } else if !provider.is_empty() {
            config.summarizer.local.provider = provider;
            if !model.is_empty() {
                config.summarizer.local.model = model;
            }
            config.summarizer.local.api_key = api_key;
            config.summarizer.active = "local".to_string();
        }
    }
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
        assert_eq!(config.asr.local.provider, "whisper");
        assert_eq!(config.summarizer.local.provider, "ollama");

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
        let content = r#"version = 1

[asr]
provider = "whisper"
model = "base.en"
api_key = ""
"#;
        fs::write(&paths.config_path, content).unwrap();

        let config = Config::load(&paths).unwrap();
        assert_eq!(config.version, CONFIG_VERSION);
        assert_eq!(config.ui.color_theme, "minimal");
        assert_eq!(config.asr.local.provider, "whisper");

        let updated = fs::read_to_string(&paths.config_path).unwrap();
        assert!(updated.contains("version = 2"));
        assert!(updated.contains("[summarizer.local]"));
    }

    #[test]
    fn redacted_hides_api_keys() {
        let mut config = Config::default();
        config.asr.local.api_key = "secret".to_string();
        config.summarizer.cloud.api_key = "secret2".to_string();
        let redacted = config.redacted();
        assert_eq!(redacted.asr.local.api_key, "<redacted>");
        assert_eq!(redacted.summarizer.cloud.api_key, "<redacted>");
    }

    #[test]
    fn validate_rejects_bad_provider() {
        let mut config = Config::default();
        config.asr.local.provider = "bad".to_string();
        assert!(config.validate().is_err());
    }
}
