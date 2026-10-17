"""Read-only personal environment summaries."""

import unicodedata


def display_text(value: object, limit: int = 120) -> str:
    """Keep saved metadata on one bounded line without terminal control sequences."""
    text = "".join(
        " " if unicodedata.category(char).startswith("C") else char for char in str(value)
    )
    return text if len(text) <= limit else text[: limit - 1] + "…"


def show_environment_status(status: dict) -> None:
    labels = {
        "ready": "준비 완료",
        "install-needed": "설치 필요",
        "update-needed": "환경 갱신 필요",
    }
    print(f"상태: {labels.get(status['state'], status['state'])}")
    if status.get("python"):
        print(f"Python: {display_text(str(status['python']), 500)}")
    if status.get("reason"):
        print(display_text(status["reason"], 500))
    for item in status.get("active", []):
        print(
            f"사용 중: {display_text(item.get('experiment', 'unknown'))} · "
            f"{display_text(item.get('kind', 'process'))} · PID {item.get('pid', '?')}"
        )
    if status.get("next_command"):
        print(f"다음: {status['next_command']}")
