/// Error returned when reading or parsing a HuggingFace `config.json`
/// from disk. Wraps the underlying I/O or JSON deserialization error so
/// callers can match on the failure category.
#[derive(Debug)]
pub enum ConfigLoadError {
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl std::fmt::Display for ConfigLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "failed to read config.json: {e}"),
            Self::Json(e) => write!(f, "failed to parse config.json: {e}"),
        }
    }
}

impl std::error::Error for ConfigLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for ConfigLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for ConfigLoadError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
