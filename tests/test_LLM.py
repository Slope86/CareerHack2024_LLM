# test_LLM.py
from unittest.mock import MagicMock

import pytest

from LLM import embedding_all_doc


@pytest.fixture
def mock_doc2vector(mocker):
    # 使用 pytest-mock 提供的 mocker fixture 來 mock doc2vector 函式
    return mocker.patch("LLM.doc2vector", autospec=True)


def test_embedding_all_doc(mock_doc2vector):
    # 設定 doc2vector 的模擬返回值
    mock_doc2vector.side_effect = [MagicMock()] * 9  # 這裡根據你的 total_db 長度來調整

    # 執行你的函式
    total_db = embedding_all_doc()

    # 斷言
    assert isinstance(total_db, list)
    assert len(total_db) == 9

    for db in total_db:
        assert db is not None

    # 檢查 doc2vector 是否被呼叫了預期次數
    assert mock_doc2vector.call_count == 9

    # 你也可以檢查 doc2vector 的呼叫參數等等
    # 例如：mock_doc2vector.assert_called_with(expected_argument)
