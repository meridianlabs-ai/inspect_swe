from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser


def build_user_prompt(messages: list[ChatMessage]) -> tuple[str, bool]:
    if messages and isinstance(messages[-1], ChatMessageAssistant):
        raise ValueError("Messages input ends with an assistant messages.")

    last_assistant_idx = next(
        (
            i
            for i, m in reversed(list(enumerate(messages)))
            if isinstance(m, ChatMessageAssistant)
        ),
        None,
    )

    has_assistant_response = last_assistant_idx is not None
    start_idx = (last_assistant_idx + 1) if last_assistant_idx is not None else 0

    prompt = "\n\n".join(
        m.text for m in messages[start_idx:] if isinstance(m, ChatMessageUser)
    )

    return prompt, has_assistant_response
