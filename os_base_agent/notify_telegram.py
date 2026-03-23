from __future__ import annotations
import time

import requests


MAX_TG_TEXT = 3900


def send(bot_token: str, chat_id: str, text: str, retries: int = 3, timeout_sec: int = 20) -> None:
    payload_text = text if len(text) <= MAX_TG_TEXT else text[: MAX_TG_TEXT - 20] + "\n...[truncated]"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json={"chat_id": chat_id, "text": payload_text}, timeout=timeout_sec)
            r.raise_for_status()
            return
        except requests.RequestException as e:
            last_err = e
            if attempt == retries:
                break
            time.sleep(1.0 * attempt)
    if last_err is not None:
        raise last_err
