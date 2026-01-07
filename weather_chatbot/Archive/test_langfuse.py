
from langfuse import Langfuse
from config.settings import Settings
import httpx
from datetime import datetime, timezone

settings = Settings()
# For smoke tests only; better to supply a Zscaler CA bundle for verify=...
insecure_client = httpx.Client(verify=False)

lf = Langfuse(
    host=settings.langfuse_host,
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    httpx_client=insecure_client,
)

# --- Trace ---
trace = lf.trace(name="smoke-test", input={"ping": "pong"})
try:
    # --- Span ---
    span = lf.span(name="step-1", trace_id=trace.id)
    try:
        span.update(status_message="doing work")
    finally:
        # Preferred if your version supports it:
        if hasattr(span, "end"):
            span.end()
        else:
            # Fallback for very old builds: set end_time via update payload
            span.update(end_time=datetime.now(timezone.utc).isoformat())

    # --- Generation (child of span) ---
    gen = lf.generation(
        name="llm-call",
        trace_id=trace.id,
        parent_observation_id=span.id,
        model_name="router-model",
    )
    try:
        gen.update(input={"prompt": "Hello"})
        gen.update(output={"text": "World"})
    finally:
        if hasattr(gen, "end"):
            gen.end()
        else:
            gen.update(end_time=datetime.now(timezone.utc).isoformat())

    # Final trace output
    trace.update(output={"ok": True})
finally:
    if hasattr(trace, "end"):
        trace.end()
    else:
        trace.update(end_time=datetime.now(timezone.utc).isoformat())

lf.flush()  # important for short-lived apps
print("DONE. Trace ID:", trace.id)
