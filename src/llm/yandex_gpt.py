from __future__ import annotations

import httpx

DEFAULT_API_URL = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'


def model_uri(folder_id: str, model: str) -> str:
    if model.startswith('gpt://'):
        return model
    return f'gpt://{folder_id}/{model}'


def to_yandex_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get('role', 'user')
        text = msg.get('text') or msg.get('content') or ''
        out.append({'role': role, 'text': text})
    return out


def complete(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    folder_id: str,
    model: str = 'yandexgpt-lite',
    temperature: float = 0.3,
    max_tokens: int = 1500,
    api_url: str = DEFAULT_API_URL,
    timeout: float = 60.0,
) -> str:
    payload = {
        'modelUri': model_uri(folder_id, model),
        'completionOptions': {
            'stream': False,
            'temperature': temperature,
            'maxTokens': str(max_tokens),
        },
        'messages': to_yandex_messages(messages),
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Api-Key {api_key}',
        'x-folder-id': folder_id,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(api_url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    try:
        return data['result']['alternatives'][0]['message']['text']
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f'Unexpected Yandex GPT response: {data!r}') from exc
