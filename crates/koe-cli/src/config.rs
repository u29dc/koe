use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;

const CONFIG_VERSION: u32 = 4;

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
    pub transcribe: TranscribeConfig,
    pub summarize: SummarizeConfig,
    pub session: SessionConfig,
    pub ui: UiConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: CONFIG_VERSION,
            audio: AudioConfig::default(),
            transcribe: TranscribeConfig::default(),
            summarize: SummarizeConfig::default(),
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
    pub mixdown: MixdownConfig,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 1,
            sources: vec!["system".to_string(), "microphone".to_string()],
            microphone_device_id: String::new(),
            mixdown: MixdownConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MixdownConfig {
    pub agc: AgcConfig,
    pub denoise: DenoiseConfig,
    pub high_pass: HighPassConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AgcConfig {
    pub enabled: bool,
    pub target_rms_dbfs: f32,
    pub max_gain_db: f32,
    pub min_gain_db: f32,
    pub attack_ms: u32,
    pub release_ms: u32,
    pub limiter_ceiling_dbfs: f32,
}

impl Default for AgcConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_rms_dbfs: -20.0,
            max_gain_db: 12.0,
            min_gain_db: -12.0,
            attack_ms: 10,
            release_ms: 250,
            limiter_ceiling_dbfs: -1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DenoiseConfig {
    pub enabled: bool,
    pub threshold_dbfs: f32,
    pub reduction_db: f32,
    pub attack_ms: u32,
    pub release_ms: u32,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold_dbfs: -45.0,
            reduction_db: 10.0,
            attack_ms: 10,
            release_ms: 200,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HighPassConfig {
    pub enabled: bool,
    pub cutoff_hz: f32,
}

impl Default for HighPassConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cutoff_hz: 100.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TranscribeConfig {
    pub active: String,
    pub local: ProviderConfig,
    pub cloud: ProviderConfig,
}

impl Default for TranscribeConfig {
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
pub struct SummarizeConfig {
    pub active: String,
    pub local: ProviderConfig,
    pub cloud: ProviderConfig,
    pub prompt_profile: String,
}

impl Default for SummarizeConfig {
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
            show_transcript: true,
            notes_only_default: false,
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
        redact_provider(&mut redacted.transcribe.local);
        redact_provider(&mut redacted.transcribe.cloud);
        redact_provider(&mut redacted.summarize.local);
        redact_provider(&mut redacted.summarize.cloud);
        redacted
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_active("transcribe.active", self.transcribe.active.as_str())?;
        validate_active("summarize.active", self.summarize.active.as_str())?;
        validate_transcribe_profile(
            "transcribe.local",
            &self.transcribe.local,
            self.transcribe.active == "local",
        )?;
        validate_transcribe_profile(
            "transcribe.cloud",
            &self.transcribe.cloud,
            self.transcribe.active == "cloud",
        )?;
        validate_summarize_profile(
            "summarize.local",
            &self.summarize.local,
            self.summarize.active == "local",
        )?;
        validate_summarize_profile(
            "summarize.cloud",
            &self.summarize.cloud,
            self.summarize.active == "cloud",
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
        let agc = &self.audio.mixdown.agc;
        if agc.target_rms_dbfs > 0.0 {
            return Err(ConfigError::Validation(
                "audio.mixdown.agc.target_rms_dbfs must be at or below 0".into(),
            ));
        }
        if agc.max_gain_db < agc.min_gain_db {
            return Err(ConfigError::Validation(
                "audio.mixdown.agc.max_gain_db must be >= min_gain_db".into(),
            ));
        }
        if agc.attack_ms == 0 {
            return Err(ConfigError::Validation(
                "audio.mixdown.agc.attack_ms must be greater than 0".into(),
            ));
        }
        if agc.release_ms == 0 {
            return Err(ConfigError::Validation(
                "audio.mixdown.agc.release_ms must be greater than 0".into(),
            ));
        }
        if agc.limiter_ceiling_dbfs > 0.0 {
            return Err(ConfigError::Validation(
                "audio.mixdown.agc.limiter_ceiling_dbfs must be at or below 0".into(),
            ));
        }
        let denoise = &self.audio.mixdown.denoise;
        if denoise.threshold_dbfs > 0.0 {
            return Err(ConfigError::Validation(
                "audio.mixdown.denoise.threshold_dbfs must be at or below 0".into(),
            ));
        }
        if denoise.reduction_db < 0.0 {
            return Err(ConfigError::Validation(
                "audio.mixdown.denoise.reduction_db must be non-negative".into(),
            ));
        }
        if denoise.attack_ms == 0 {
            return Err(ConfigError::Validation(
                "audio.mixdown.denoise.attack_ms must be greater than 0".into(),
            ));
        }
        if denoise.release_ms == 0 {
            return Err(ConfigError::Validation(
                "audio.mixdown.denoise.release_ms must be greater than 0".into(),
            ));
        }
        let high_pass = &self.audio.mixdown.high_pass;
        if high_pass.cutoff_hz <= 0.0 {
            return Err(ConfigError::Validation(
                "audio.mixdown.high_pass.cutoff_hz must be greater than 0".into(),
            ));
        }
        let nyquist = self.audio.sample_rate as f32 / 2.0;
        if high_pass.cutoff_hz >= nyquist {
            return Err(ConfigError::Validation(
                "audio.mixdown.high_pass.cutoff_hz must be below the Nyquist frequency".into(),
            ));
        }

        if self.summarize.prompt_profile.trim().is_empty() {
            return Err(ConfigError::Validation(
                "summarize.prompt_profile must not be empty".into(),
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

fn validate_transcribe_profile(
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

fn validate_summarize_profile(
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
        assert_eq!(config.transcribe.local.provider, "whisper");
        assert_eq!(config.summarize.local.provider, "ollama");

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
    fn load_updates_version_and_defaults() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("koe");
        let paths = ConfigPaths::from_base(base);
        fs::create_dir_all(&paths.base_dir).unwrap();
        let content = r#"version = 2

[transcribe]
active = "local"

[transcribe.local]
provider = "whisper"
model = "base.en"
api_key = ""
"#;
        fs::write(&paths.config_path, content).unwrap();

        let config = Config::load(&paths).unwrap();
        assert_eq!(config.version, CONFIG_VERSION);
        assert_eq!(config.ui.color_theme, "minimal");
        assert_eq!(config.transcribe.local.provider, "whisper");

        let updated = fs::read_to_string(&paths.config_path).unwrap();
        assert!(updated.contains("version = 4"));
        assert!(updated.contains("[summarize.local]"));
    }

    #[test]
    fn redacted_hides_api_keys() {
        let mut config = Config::default();
        config.transcribe.local.api_key = "secret".to_string();
        config.summarize.cloud.api_key = "secret2".to_string();
        let redacted = config.redacted();
        assert_eq!(redacted.transcribe.local.api_key, "<redacted>");
        assert_eq!(redacted.summarize.cloud.api_key, "<redacted>");
    }

    #[test]
    fn validate_rejects_bad_provider() {
        let mut config = Config::default();
        config.transcribe.local.provider = "bad".to_string();
        assert!(config.validate().is_err());
    }
}
