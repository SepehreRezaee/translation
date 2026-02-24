from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    model_path: str = "./models/sharifsetup-translate"
    model_display_name: str = "Sharifsetup-Translator"
    verbose_logs: bool = False
    model_device: str = "auto"
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    trust_remote_code: bool = True
    use_fast_processor: bool = False
    fix_mistral_regex: bool = True

    default_temperature: float = 0.0
    default_top_p: float = 0.95
    default_max_new_tokens: int = 512
    default_repetition_penalty: float = 1.0

    api_title: str = "Sharifsetup-Translator API"
    api_version: str = "1.0.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
