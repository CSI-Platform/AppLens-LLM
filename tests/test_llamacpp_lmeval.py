from __future__ import annotations

from applens_llm.llamacpp_lmeval import normalize_chat_completion_response, strip_reasoning_blocks


def test_strip_reasoning_blocks_removes_empty_think_tags_before_answer() -> None:
    assert strip_reasoning_blocks("<think>\n\n</think>\n\n C") == "C"


def test_strip_reasoning_blocks_preserves_regular_answer() -> None:
    assert strip_reasoning_blocks("The best answer is C.") == "The best answer is C."


def test_normalize_chat_completion_response_strips_choice_message_content() -> None:
    payload = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think>hidden</think>\n\n C",
                },
            }
        ]
    }

    normalized = normalize_chat_completion_response(payload)

    assert normalized["choices"][0]["message"]["content"] == "C"
    assert payload["choices"][0]["message"]["content"] == "<think>hidden</think>\n\n C"
