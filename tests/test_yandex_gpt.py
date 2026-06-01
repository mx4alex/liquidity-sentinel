from unittest.mock import MagicMock, patch

from ru_liquidity_sentinel.llm.yandex_gpt import complete, model_uri, to_yandex_messages


def test_model_uri():
    assert model_uri('b1abc', 'yandexgpt-lite') == 'gpt://b1abc/yandexgpt-lite'
    assert model_uri('b1abc', 'gpt://b1abc/yandexgpt') == 'gpt://b1abc/yandexgpt'


def test_to_yandex_messages_maps_content():
    msgs = to_yandex_messages([{'role': 'user', 'content': 'hi'}])
    assert msgs == [{'role': 'user', 'text': 'hi'}]


@patch('ru_liquidity_sentinel.llm.yandex_gpt.httpx.Client')
def test_complete_parses_response(mock_client_cls):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        'result': {'alternatives': [{'message': {'text': 'Ответ модели'}}]},
    }
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    text = complete(
        [{'role': 'user', 'text': 'вопрос'}],
        api_key='key',
        folder_id='folder1',
        model='yandexgpt-lite',
    )
    assert text == 'Ответ модели'
    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs['headers']['Authorization'] == 'Api-Key key'
    assert call_kwargs['headers']['x-folder-id'] == 'folder1'
    assert call_kwargs['json']['modelUri'] == 'gpt://folder1/yandexgpt-lite'
