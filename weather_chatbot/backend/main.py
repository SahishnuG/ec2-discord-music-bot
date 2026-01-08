import httpx
from langfuse import Langfuse
import json
from dotenv import set_key
from config.settings import Settings
from src.api import serve

class WeatherAgent:
    def __init__(self):
        self.settings = Settings()
        self.env = self.settings.env_path
        #langfuse
        insecure_client = httpx.Client(verify=False) #bypass Zscaler
        self.langfuse = Langfuse(host=self.settings.langfuse_host,
                            public_key=self.settings.langfuse_public_key,
                            secret_key=self.settings.langfuse_secret_key,
                            httpx_client=insecure_client
                            )

    def get_system_content(self, prompt_name: str, prompt_type: str = "chat", label: str | None = None) -> str:
        """Fetch the system message content robustly for chat or text prompts."""
        p = self.langfuse.get_prompt(name=prompt_name, type=prompt_type, label=label)
        raw = p.prompt
        if isinstance(raw, list):
            # Prefer 'system' role; fallback to first item content
            sys_msg = next((m.get("content") for m in raw if m.get("role") == "system"), None)
            if sys_msg is None and raw and isinstance(raw[0], dict):
                sys_msg = raw[0].get("content")
        elif isinstance(raw, str):
            sys_msg = raw
        else:
            raise TypeError(f"Unsupported prompt payload type: {type(raw)}")

        if not isinstance(sys_msg, str) or not sys_msg.strip():
            raise ValueError(f"System content missing/invalid for prompt '{prompt_name}'")
        return sys_msg
    def edit_env(self):
        # Fetch fresh values
        router_sys     = self.get_system_content("router",     prompt_type="chat")      # change to "text" if stored as text in langfuse
        summariser_sys = self.get_system_content("summariser", prompt_type="chat")      # change to "text" if stored as text in langfuse
        print(f"router: {router_sys}\nsummariser: {summariser_sys}")

        # Upsert (replace existing lines or create new ones) to .env. JSON-escape to preserve newlines/quotes.
        set_key(self.env, "ROUTER_SYSTEM_PROMPT",    json.dumps(router_sys))
        set_key(self.env, "SUMMARISER_SYSTEM_PROMPT", json.dumps(summariser_sys))

if __name__ == "__main__":
    WeatherAgent().edit_env()
    serve()