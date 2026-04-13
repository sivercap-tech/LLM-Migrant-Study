#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import requests

DEFAULT_INPUT = "output_2023_2026/youtube_videos.csv"
DEFAULT_OUTPUT = "output_2023_2026/youtube_videos_classified.csv"
DEFAULT_PROVIDER = "openai"
DEFAULT_MODE = "sync"
DEFAULT_BATCH_INPUT = "output_2023_2026/batch_requests.jsonl"
DEFAULT_BATCH_METADATA = "output_2023_2026/batch_requests.meta.json"
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3
TRANSIENT_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}

SYSTEM_PROMPT = """
You are a strict classification function inside a data pipeline.

Task:
Classify a YouTube video for inclusion in a research dataset about:
"discussion of migrants in Russia".

Return ONLY valid JSON.

--------------------------------
DECISION LABELS
--------------------------------
- include
- exclude
- borderline

--------------------------------
INCLUDE
--------------------------------
Return "include" if ALL are true:
1. Russia is the main geographic context.
2. The video is about migrants, migration, or foreign residents/workers IN Russia,
   OR migration/migrants in Russia are a clearly present and meaningful topic,
   even if not the only or dominant topic.
3. The migration link is explicit in the title or description.

--------------------------------
Examples of explicit migration anchors:
- migrants / migration / immigration
- foreign workers in Russia
- arriving, settling, living, or working in Russia as non-citizens
- visas, patents, registration, citizenship, deportation
- police raids, legal status, adaptation, integration
- public debate specifically about migrants in Russia

--------------------------------
BORDERLINE
--------------------------------
Return "borderline" if:
- Russia is the main context, but the migration link is indirect, weak, or ambiguous; OR
- the video mentions foreigners / ethnic tension / foreign suspects / nationalist groups,
  but metadata does not clearly confirm that migrants in Russia are an actual discussion topic.

--------------------------------
EXCLUDE
--------------------------------
Return "exclude" if ANY are true:
- Russia is not the main context.
- The topic is emigration from Russia or Russians abroad.
- The topic is generic ethnicity/culture/nationalism without a clear migration-in-Russia link.
- The video is about tourists, expats, foreigners, or cultural differences without a clear migration anchor.
- Foreign nationals are mentioned only as crime suspects or public figures, without a clear migration discussion.
- Migration is only a passing mention with no meaningful discussion.
- The main title text is predominantly in Kyrgyz, Tajik, or Uzbek.

--------------------------------
IMPORTANT DISAMBIGUATION RULES
--------------------------------
- "Foreigner" does NOT automatically mean "migrant".
- Mentioning Uzbekistan, Tajikistan, Kyrgyzstan, or ethnic conflict does NOT automatically mean the video is about migrants in Russia.
- Use the description, not only the title. Titles may be misleading.
- If uncertain between include and exclude, use borderline only when the migration link is plausible but not explicit.

--------------------------------
PROXY TERMS AND EUPHEMISMS
--------------------------------
В российском контексте видео могут говорить о мигрантах косвенно, через прокси-термины или эвфемизмы.

Такие слова и выражения, как:
- иностранцы
- иностранные специалисты
- иноязычные / люди, не говорящие по-русски
- приезжие
- граждане стран Центральной Азии / граждане СНГ
- нелегалы / нелегальные работники
- Русская община
- этнизированные упоминания узбеков, таджиков, кыргызов в России

могут считаться миграционной темой ТОЛЬКО если контекст явно указывает на:
- проживание или работу в России
- трудовую миграцию
- правовой статус, регистрацию, депортацию, гражданство, рейды
- адаптацию, интеграцию, межэтническое напряжение вокруг присутствия приезжих в России
- общественную или политическую дискуссию, которая по сути касается мигрантов в России

Упоминание "Русской общины" часто связано с конфликтами, рейдами, давлением на приезжих,
иностранцев или мигрантов в России, но само по себе не является автоматическим признаком
миграционной темы без дополнительного контекста в title/description.

НЕ считай такие термины миграционной темой, если контекст в основном про:
- туристов
- иностранных студентов за рубежом
- иностранных специалистов, не связанных с миграцией
- иностранных подозреваемых без более широкой дискуссии о миграции

--------------------------------
OUTPUT FORMAT
--------------------------------
{
  "decision": "include|exclude|borderline",
  "passes_filter": true|false,
  "migration_anchor": "explicit|implicit|none",
  "confidence": 0.0,
  "reason": "one short sentence"
}

--------------------------------
FIELD RULES
--------------------------------
- passes_filter = true ONLY if decision = "include", otherwise false
- confidence must reflect certainty
- reason must be short and factual

Return ONLY JSON.
"""

ALLOWED_DECISIONS = {"include", "borderline", "exclude"}
ALLOWED_MIGRATION_ANCHORS = {"explicit", "implicit", "none"}
OUTPUT_COLUMNS = [
    "provider",
    "model",
    "decision",
    "passes_filter",
    "migration_anchor",
    "confidence",
    "reason",
]


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    env_var: str
    default_model: str
    default_base_url: str


PROVIDER_CONFIGS = {
    "openai": ProviderConfig(
        name="openai",
        env_var="OPENAI_API_KEY",
        default_model="gpt-4.1-mini",
        default_base_url="https://api.openai.com/v1",
    ),
    "gemini": ProviderConfig(
        name="gemini",
        env_var="GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
        default_base_url="https://generativelanguage.googleapis.com/v1beta",
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        env_var="ANTHROPIC_API_KEY",
        default_model="claude-3-5-haiku-latest",
        default_base_url="https://api.anthropic.com/v1",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Классифицирует YouTube-видео через LLM для датасета о мигрантах в России."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV с видео для классификации.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV с добавленными полями классификации.")
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDER_CONFIGS.keys()),
        default=DEFAULT_PROVIDER,
        help="LLM provider: openai, gemini или anthropic.",
    )
    parser.add_argument(
        "--mode",
        choices=["sync", "batch-prepare", "batch-submit", "batch-status", "batch-download"],
        default=DEFAULT_MODE,
        help="Режим работы.",
    )
    parser.add_argument("--api-key", default=None, help="API key. По умолчанию из env переменной провайдера.")
    parser.add_argument("--base-url", default=None, help="Переопределить базовый URL API.")
    parser.add_argument("--model", default=None, help="Имя модели для классификации.")
    parser.add_argument("--max-rows", type=int, default=None, help="Ограничить количество строк.")
    parser.add_argument("--start-row", type=int, default=0, help="С какой строки начать, без заголовка.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Пауза между sync запросами.")
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Дописывать результаты в существующий output CSV вместо перезаписи.",
    )
    parser.add_argument(
        "--skip-existing-output",
        action="store_true",
        help="При sync-режиме пропускать video_id, уже присутствующие в output CSV.",
    )
    parser.add_argument("--batch-input", default=DEFAULT_BATCH_INPUT, help="JSONL для batch prepare/submit.")
    parser.add_argument("--batch-metadata", default=DEFAULT_BATCH_METADATA, help="JSON с маппингом custom_id -> строка.")
    parser.add_argument("--batch-id", default=None, help="ID или resource name batch job.")
    parser.add_argument("--file-id", default=None, help="ID/имя output file, если нужно скачать напрямую.")
    return parser.parse_args()


def resolve_provider_settings(args: argparse.Namespace) -> tuple[str, str, str, str]:
    config = PROVIDER_CONFIGS[args.provider]
    return (
        config.name,
        args.api_key or os.getenv(config.env_var) or "",
        args.model or config.default_model,
        args.base_url or config.default_base_url,
    )


def require_api_key(provider: str, api_key: str) -> None:
    if api_key:
        return
    raise SystemExit(f"Missing API key. Set {PROVIDER_CONFIGS[provider].env_var} or pass --api-key.")


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_existing_video_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8", newline="") as file:
        return {
            row.get("video_id", "").strip()
            for row in csv.DictReader(file)
            if row.get("video_id", "").strip()
        }


def slice_rows(rows: List[Dict[str, str]], start_row: int, max_rows: int | None) -> List[Dict[str, str]]:
    if start_row < 0:
        raise ValueError("--start-row must be >= 0")
    sliced = rows[start_row:]
    if max_rows is not None:
        if max_rows < 0:
            raise ValueError("--max-rows must be >= 0")
        sliced = sliced[:max_rows]
    return sliced


def build_user_prompt(row: Dict[str, str]) -> str:
    payload = {
        "video_id": row.get("video_id", ""),
        "title": row.get("title", ""),
        "description": row.get("description", ""),
        "channel_title": row.get("channel_title", ""),
        "published_at": row.get("published_at", ""),
        "queries": row.get("queries", ""),
        "url": row.get("url", ""),
    }
    return "Classify this video.\n\nVIDEO_METADATA:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def extract_openai_message_content(payload: Dict[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("API response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("API response content is empty")
    return content.strip()


def extract_anthropic_message_content(payload: Dict[str, object]) -> str:
    content_blocks = payload.get("content")
    if not isinstance(content_blocks, list) or not content_blocks:
        raise ValueError("Anthropic response missing content")
    text_parts: List[str] = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
    content = "\n".join(text_parts).strip()
    if not content:
        raise ValueError("Anthropic response content is empty")
    return content


def extract_gemini_message_content(payload: Dict[str, object]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini response missing candidates")
    first_candidate = candidates[0]
    content = first_candidate.get("content") if isinstance(first_candidate, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not parts:
        raise ValueError("Gemini response missing parts")
    text_parts: List[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
    result = "\n".join(text_parts).strip()
    if not result:
        raise ValueError("Gemini response content is empty")
    return result


def normalize_result(raw: Dict[str, object]) -> Dict[str, object]:
    required_fields = [field for field in OUTPUT_COLUMNS if field not in {"provider", "model"}]
    missing = [field for field in required_fields if field not in raw]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    decision = raw["decision"]
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"Invalid decision: {decision}")

    passes_filter = raw["passes_filter"]
    if not isinstance(passes_filter, bool):
        raise ValueError("passes_filter must be boolean")
    if passes_filter != (decision == "include"):
        raise ValueError("passes_filter must be true only for include")

    migration_anchor = raw["migration_anchor"]
    if migration_anchor not in ALLOWED_MIGRATION_ANCHORS:
        raise ValueError(f"Invalid migration_anchor: {migration_anchor}")

    confidence = raw["confidence"]
    if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
        raise ValueError("confidence must be between 0.0 and 1.0")

    reason = raw["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be non-empty string")
    if len(reason.split()) > 20:
        raise ValueError("reason must be at most 20 words")

    return {
        "decision": decision,
        "passes_filter": passes_filter,
        "migration_anchor": migration_anchor,
        "confidence": round(float(confidence), 4),
        "reason": reason.strip(),
    }


def parse_json_object(text: str) -> Dict[str, object]:
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Model response is not a JSON object")
    return normalize_result(result)


def post_json(session: requests.Session, url: str, headers: Dict[str, str], body: Dict[str, object]) -> Dict[str, object]:
    response = session.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
    raise_for_status_with_body(response)
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("API response is not a JSON object")
    return payload


def raise_for_status_with_body(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip()
        if body:
            raise RuntimeError(
                f"HTTP {response.status_code} for {response.request.method} {response.url}\n{body}"
            ) from exc
        raise


def is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, requests.RequestException):
        response = getattr(exc, "response", None)
        if response is None:
            return True
        return response.status_code in TRANSIENT_HTTP_STATUS_CODES
    if isinstance(exc, RuntimeError):
        text = str(exc)
        for status_code in TRANSIENT_HTTP_STATUS_CODES:
            if f"HTTP {status_code} " in text:
                return True
    return False


def retry_delay_seconds(attempt: int) -> float:
    return min(5 * (2 ** (attempt - 1)), 60)


def classify_openai_row(*, row: Dict[str, str], api_key: str, base_url: str, model: str, session: requests.Session) -> Dict[str, object]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ],
    }
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            payload = post_json(session, url, headers, body)
            return parse_json_object(extract_openai_message_content(payload))
        except (requests.RequestException, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES and is_transient_error(exc):
                time.sleep(retry_delay_seconds(attempt))
                continue
            break
    raise RuntimeError(f"Failed to classify video_id={row.get('video_id', '')}: {last_error}") from last_error


def classify_gemini_row(*, row: Dict[str, str], api_key: str, base_url: str, model: str, session: requests.Session) -> Dict[str, object]:
    url = base_url.rstrip("/") + f"/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        "contents": [{"role": "user", "parts": [{"text": build_user_prompt(row)}]}],
    }
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            payload = post_json(session, url, headers, body)
            return parse_json_object(extract_gemini_message_content(payload))
        except (requests.RequestException, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES and is_transient_error(exc):
                time.sleep(retry_delay_seconds(attempt))
                continue
            break
    raise RuntimeError(f"Failed to classify video_id={row.get('video_id', '')}: {last_error}") from last_error


def classify_anthropic_row(*, row: Dict[str, str], api_key: str, base_url: str, model: str, session: requests.Session) -> Dict[str, object]:
    url = base_url.rstrip("/") + "/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 300,
        "temperature": 0,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": build_user_prompt(row)}],
    }
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            payload = post_json(session, url, headers, body)
            return parse_json_object(extract_anthropic_message_content(payload))
        except (requests.RequestException, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES and is_transient_error(exc):
                time.sleep(retry_delay_seconds(attempt))
                continue
            break
    raise RuntimeError(f"Failed to classify video_id={row.get('video_id', '')}: {last_error}") from last_error


def classify_row(*, provider: str, row: Dict[str, str], api_key: str, base_url: str, model: str, session: requests.Session) -> Dict[str, object]:
    if provider == "openai":
        return classify_openai_row(row=row, api_key=api_key, base_url=base_url, model=model, session=session)
    if provider == "gemini":
        return classify_gemini_row(row=row, api_key=api_key, base_url=base_url, model=model, session=session)
    if provider == "anthropic":
        return classify_anthropic_row(row=row, api_key=api_key, base_url=base_url, model=model, session=session)
    raise ValueError(f"Unsupported provider: {provider}")


def write_rows(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_rows(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def load_batch_metadata(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Batch metadata file is invalid")
    return payload


def build_batch_request_line(provider: str, custom_id: str, model: str, row: Dict[str, str]) -> Dict[str, object]:
    if provider == "openai":
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(row)},
                ],
            },
        }
    if provider == "gemini":
        return {
            "key": custom_id,
            "request": {
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
                "contents": [{"role": "user", "parts": [{"text": build_user_prompt(row)}]}],
            },
        }
    if provider == "anthropic":
        return {
            "custom_id": custom_id,
            "params": {
                "model": model,
                "max_tokens": 300,
                "temperature": 0,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": build_user_prompt(row)}],
            },
        }
    raise ValueError(f"Unsupported provider: {provider}")


def prepare_batch(
    *,
    provider: str,
    model: str,
    input_path: Path,
    batch_input_path: Path,
    batch_metadata_path: Path,
    start_row: int,
    max_rows: int | None,
) -> int:
    rows = read_rows(input_path)
    selected_rows = slice_rows(rows, start_row, max_rows)
    batch_input_path.parent.mkdir(parents=True, exist_ok=True)
    batch_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_rows: List[Dict[str, object]] = []
    with batch_input_path.open("w", encoding="utf-8") as batch_file:
        for offset, row in enumerate(selected_rows, start=start_row):
            custom_id = f"row-{offset}-video-{row.get('video_id', '')}"
            batch_file.write(
                json.dumps(build_batch_request_line(provider, custom_id, model, row), ensure_ascii=False) + "\n"
            )
            metadata_rows.append(
                {
                    "custom_id": custom_id,
                    "row_index": offset,
                    "provider": provider,
                    "model": model,
                    "row": row,
                }
            )

    batch_metadata_path.write_text(
        json.dumps(
            {
                "provider": provider,
                "model": model,
                "input_csv": str(input_path),
                "batch_input": str(batch_input_path),
                "start_row": start_row,
                "max_rows": max_rows,
                "request_count": len(metadata_rows),
                "rows": metadata_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved batch input to {batch_input_path}")
    print(f"Saved batch metadata to {batch_metadata_path}")
    print(f"Prepared {len(metadata_rows)} {provider} requests for model {model}")
    return 0


def openai_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def upload_gemini_file(*, api_key: str, path: Path, display_name: str) -> Dict[str, object]:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/jsonl"
    num_bytes = path.stat().st_size
    start_response = requests.post(
        "https://generativelanguage.googleapis.com/upload/v1beta/files",
        headers={
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(num_bytes),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": display_name}},
        timeout=REQUEST_TIMEOUT,
    )
    start_response.raise_for_status()
    upload_url = start_response.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        raise RuntimeError("Gemini upload did not return X-Goog-Upload-URL")

    with path.open("rb") as file:
        upload_response = requests.post(
            upload_url,
            headers={
                "Content-Length": str(num_bytes),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            data=file.read(),
            timeout=REQUEST_TIMEOUT,
        )
    upload_response.raise_for_status()
    payload = upload_response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Gemini file upload response is invalid")
    return payload


def submit_batch(*, provider: str, api_key: str, base_url: str, model: str, batch_input_path: Path, batch_metadata_path: Path) -> int:
    if provider == "openai":
        session = requests.Session()
        upload_url = base_url.rstrip("/") + "/files"
        batch_url = base_url.rstrip("/") + "/batches"
        with batch_input_path.open("rb") as batch_file:
            response = session.post(
                upload_url,
                headers=openai_headers(api_key),
                data={"purpose": "batch"},
                files={"file": (batch_input_path.name, batch_file, "application/jsonl")},
                timeout=REQUEST_TIMEOUT,
            )
        raise_for_status_with_body(response)
        file_payload = response.json()
        input_file_id = file_payload.get("id")
        if not isinstance(input_file_id, str) or not input_file_id:
            raise RuntimeError("OpenAI file upload did not return file id")
        response = session.post(
            batch_url,
            headers={**openai_headers(api_key), "Content-Type": "application/json"},
            json={"input_file_id": input_file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"},
            timeout=REQUEST_TIMEOUT,
        )
        raise_for_status_with_body(response)
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return 0

    if provider == "gemini":
        file_payload = upload_gemini_file(api_key=api_key, path=batch_input_path, display_name=batch_input_path.stem)
        file_info = file_payload.get("file")
        if not isinstance(file_info, dict):
            raise RuntimeError("Gemini file upload did not return file object")
        file_name = file_info.get("name")
        if not isinstance(file_name, str) or not file_name:
            raise RuntimeError("Gemini file upload did not return file name")
        response = requests.post(
            base_url.rstrip("/") + f"/models/{model}:batchGenerateContent",
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={"batch": {"display_name": batch_input_path.stem, "input_config": {"file_name": file_name}}},
            timeout=REQUEST_TIMEOUT,
        )
        raise_for_status_with_body(response)
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return 0

    if provider == "anthropic":
        requests_payload: List[Dict[str, object]] = []
        with batch_input_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise RuntimeError("Anthropic batch input contains invalid JSON object")
                    requests_payload.append(payload)
        response = requests.post(
            base_url.rstrip("/") + "/messages/batches",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={"requests": requests_payload},
            timeout=REQUEST_TIMEOUT,
        )
        raise_for_status_with_body(response)
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f"Unsupported provider: {provider}")


def retrieve_batch(*, provider: str, api_key: str, base_url: str, batch_id: str) -> Dict[str, object]:
    if provider == "openai":
        response = requests.get(
            base_url.rstrip("/") + f"/batches/{batch_id}",
            headers=openai_headers(api_key),
            timeout=REQUEST_TIMEOUT,
        )
        raise_for_status_with_body(response)
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("OpenAI batch status response is invalid")
        return payload

    if provider == "gemini":
        normalized = batch_id if batch_id.startswith("batches/") else f"batches/{batch_id}"
        response = requests.get(
            base_url.rstrip("/") + f"/{normalized}",
            headers={"x-goog-api-key": api_key},
            timeout=REQUEST_TIMEOUT,
        )
        raise_for_status_with_body(response)
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Gemini batch status response is invalid")
        return payload

    if provider == "anthropic":
        response = requests.get(
            base_url.rstrip("/") + f"/messages/batches/{batch_id}",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=REQUEST_TIMEOUT,
        )
        raise_for_status_with_body(response)
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Anthropic batch status response is invalid")
        return payload

    raise ValueError(f"Unsupported provider: {provider}")


def download_openai_file(*, api_key: str, base_url: str, file_id: str) -> str:
    response = requests.get(
        base_url.rstrip("/") + f"/files/{file_id}/content",
        headers=openai_headers(api_key),
        timeout=REQUEST_TIMEOUT,
    )
    raise_for_status_with_body(response)
    return response.text


def download_gemini_file(*, api_key: str, base_url: str, file_name: str) -> str:
    normalized = file_name if file_name.startswith("files/") else f"files/{file_name}"
    response = requests.get(
        base_url.rstrip("/") + f"/{normalized}",
        headers={"x-goog-api-key": api_key},
        timeout=REQUEST_TIMEOUT,
    )
    raise_for_status_with_body(response)
    metadata = response.json()
    if not isinstance(metadata, dict):
        raise RuntimeError("Gemini file metadata response is invalid")
    download_uri = metadata.get("downloadUri") or metadata.get("download_uri")
    if not isinstance(download_uri, str) or not download_uri:
        raise RuntimeError("Gemini file metadata does not contain downloadUri")
    file_response = requests.get(download_uri, headers={"x-goog-api-key": api_key}, timeout=REQUEST_TIMEOUT)
    raise_for_status_with_body(file_response)
    return file_response.text


def row_index_from_metadata(metadata: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {
        item["custom_id"]: item
        for item in metadata.get("rows", [])
        if isinstance(item, dict) and isinstance(item.get("custom_id"), str)
    }


def finalize_output(*, output_path: Path, metadata: Dict[str, object], enriched_rows: List[Dict[str, object]]) -> int:
    first_row = metadata.get("rows", [{}])[0]
    base_row = first_row.get("row") if isinstance(first_row, dict) else None
    fieldnames = list(base_row.keys()) + OUTPUT_COLUMNS if isinstance(base_row, dict) else OUTPUT_COLUMNS
    write_rows(output_path, enriched_rows, fieldnames)
    print(f"Saved {len(enriched_rows)} classified rows to {output_path}")
    return 0


def download_batch_results(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    batch_id: str | None,
    file_id: str | None,
    batch_metadata_path: Path,
    output_path: Path,
) -> int:
    metadata = load_batch_metadata(batch_metadata_path)
    row_index = row_index_from_metadata(metadata)
    if not row_index:
        raise RuntimeError("Batch metadata does not contain rows")

    if provider == "openai":
        resolved_file_id = file_id
        if not resolved_file_id:
            if not batch_id:
                raise RuntimeError("Provide --batch-id or --file-id for batch-download")
            batch_payload = retrieve_batch(provider=provider, api_key=api_key, base_url=base_url, batch_id=batch_id)
            resolved_file_id = batch_payload.get("output_file_id")
            if not isinstance(resolved_file_id, str) or not resolved_file_id:
                raise RuntimeError(f"Batch {batch_id} has no output_file_id yet. Status: {batch_payload.get('status')}")
        content = download_openai_file(api_key=api_key, base_url=base_url, file_id=resolved_file_id)
        enriched_rows: List[Dict[str, object]] = []
        for line in content.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            custom_id = payload.get("custom_id") if isinstance(payload, dict) else None
            source = row_index.get(custom_id or "")
            if not source:
                raise RuntimeError(f"Unknown custom_id in batch output: {custom_id}")
            response_payload = payload.get("response")
            body = response_payload.get("body") if isinstance(response_payload, dict) else None
            if not isinstance(body, dict):
                raise RuntimeError(f"Missing response body for custom_id={custom_id}")
            classification = parse_json_object(extract_openai_message_content(body))
            row = source["row"]
            enriched_row = dict(row)
            enriched_row["provider"] = source.get("provider", provider)
            enriched_row["model"] = source.get("model", metadata.get("model", ""))
            enriched_row.update(classification)
            enriched_rows.append(enriched_row)
        return finalize_output(output_path=output_path, metadata=metadata, enriched_rows=enriched_rows)

    if provider == "gemini":
        resolved_file_name = file_id
        if not resolved_file_name:
            if not batch_id:
                raise RuntimeError("Provide --batch-id or --file-id for batch-download")
            batch_payload = retrieve_batch(provider=provider, api_key=api_key, base_url=base_url, batch_id=batch_id)
            dest = batch_payload.get("dest")
            resolved_file_name = dest.get("fileName") if isinstance(dest, dict) else None
            state = batch_payload.get("state")
            if not isinstance(resolved_file_name, str) or not resolved_file_name:
                raise RuntimeError(f"Gemini batch {batch_id} has no result file yet. State: {state}")
        content = download_gemini_file(api_key=api_key, base_url=base_url, file_name=resolved_file_name)
        enriched_rows = []
        for line in content.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError("Gemini batch output line is invalid")
            custom_id = payload.get("key")
            source = row_index.get(custom_id or "")
            if not source:
                raise RuntimeError(f"Unknown key in Gemini batch output: {custom_id}")
            response_payload = payload.get("response")
            if not isinstance(response_payload, dict):
                raise RuntimeError(f"Missing response for key={custom_id}")
            classification = parse_json_object(extract_gemini_message_content(response_payload))
            row = source["row"]
            enriched_row = dict(row)
            enriched_row["provider"] = source.get("provider", provider)
            enriched_row["model"] = source.get("model", metadata.get("model", ""))
            enriched_row.update(classification)
            enriched_rows.append(enriched_row)
        return finalize_output(output_path=output_path, metadata=metadata, enriched_rows=enriched_rows)

    if provider == "anthropic":
        if not batch_id:
            raise RuntimeError("Provide --batch-id for Anthropic batch-download")
        response = requests.get(
            base_url.rstrip("/") + f"/messages/batches/{batch_id}/results",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=REQUEST_TIMEOUT,
        )
        raise_for_status_with_body(response)
        enriched_rows = []
        for line in response.text.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError("Anthropic batch output line is invalid")
            custom_id = payload.get("custom_id")
            source = row_index.get(custom_id or "")
            if not source:
                raise RuntimeError(f"Unknown custom_id in Anthropic batch output: {custom_id}")
            result_payload = payload.get("result")
            if not isinstance(result_payload, dict) or result_payload.get("type") != "succeeded":
                raise RuntimeError(f"Anthropic batch request failed for custom_id={custom_id}: {payload}")
            message = result_payload.get("message")
            if not isinstance(message, dict):
                raise RuntimeError(f"Anthropic batch result missing message for custom_id={custom_id}")
            classification = parse_json_object(extract_anthropic_message_content(message))
            row = source["row"]
            enriched_row = dict(row)
            enriched_row["provider"] = source.get("provider", provider)
            enriched_row["model"] = source.get("model", metadata.get("model", ""))
            enriched_row.update(classification)
            enriched_rows.append(enriched_row)
        return finalize_output(output_path=output_path, metadata=metadata, enriched_rows=enriched_rows)

    raise ValueError(f"Unsupported provider: {provider}")


def run_sync(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    input_path: Path,
    output_path: Path,
    start_row: int,
    max_rows: int | None,
    sleep_seconds: float,
    append_output: bool,
    skip_existing_output: bool,
) -> int:
    rows = read_rows(input_path)
    selected_rows = slice_rows(rows, start_row, max_rows)
    enriched_rows: List[Dict[str, object]] = []
    session = requests.Session()
    fieldnames = list(rows[0].keys()) + OUTPUT_COLUMNS if rows else OUTPUT_COLUMNS
    existing_video_ids = read_existing_video_ids(output_path) if skip_existing_output else set()
    skipped_existing = 0

    if append_output and output_path.exists() and output_path.stat().st_size > 0:
        pass

    for index, row in enumerate(selected_rows, start=start_row + 1):
        video_id = row.get("video_id", "").strip()
        if video_id and video_id in existing_video_ids:
            skipped_existing += 1
            print(f"[{index}/{start_row + len(selected_rows)}] {video_id}: skipped existing output")
            continue
        classification = classify_row(
            provider=provider,
            row=row,
            api_key=api_key,
            base_url=base_url,
            model=model,
            session=session,
        )
        enriched_row = dict(row)
        enriched_row["provider"] = provider
        enriched_row["model"] = model
        enriched_row.update(classification)
        enriched_rows.append(enriched_row)
        if append_output:
            append_rows(output_path, [enriched_row], fieldnames)
        if video_id:
            existing_video_ids.add(video_id)
        print(f"[{index}/{start_row + len(selected_rows)}] {row.get('video_id', '')}: {provider}/{model} -> {classification['decision']}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not append_output:
        write_rows(output_path, enriched_rows, fieldnames)
    print(f"Saved {len(enriched_rows)} classified rows to {output_path}")
    if skip_existing_output:
        print(f"Skipped {skipped_existing} rows already present in output")
    return 0


def main() -> int:
    args = parse_args()
    provider, api_key, model, base_url = resolve_provider_settings(args)

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    batch_input_path = Path(args.batch_input).resolve()
    batch_metadata_path = Path(args.batch_metadata).resolve()

    if args.mode == "sync":
        require_api_key(provider, api_key)
        return run_sync(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            input_path=input_path,
            output_path=output_path,
            start_row=args.start_row,
            max_rows=args.max_rows,
            sleep_seconds=args.sleep_seconds,
            append_output=args.append_output,
            skip_existing_output=args.skip_existing_output,
        )

    if args.mode == "batch-prepare":
        return prepare_batch(
            provider=provider,
            model=model,
            input_path=input_path,
            batch_input_path=batch_input_path,
            batch_metadata_path=batch_metadata_path,
            start_row=args.start_row,
            max_rows=args.max_rows,
        )

    require_api_key(provider, api_key)

    if args.mode == "batch-submit":
        return submit_batch(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            batch_input_path=batch_input_path,
            batch_metadata_path=batch_metadata_path,
        )
    if args.mode == "batch-status":
        if not args.batch_id:
            raise SystemExit("Provide --batch-id for --mode batch-status.")
        payload = retrieve_batch(provider=provider, api_key=api_key, base_url=base_url, batch_id=args.batch_id)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "batch-download":
        return download_batch_results(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            batch_id=args.batch_id,
            file_id=args.file_id,
            batch_metadata_path=batch_metadata_path,
            output_path=output_path,
        )

    raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
