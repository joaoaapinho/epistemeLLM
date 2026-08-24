"""
Pull final answer out model's reply.

Model is told to finish with <answer>X</answer>. We take the last tag and
clean up X. No tag = None.
"""

import re

from episteme import config

ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def last_boxed(text):
    """
    Content of the last \boxed{...}, or None.

    Qwen falls back to its own \boxed{} habit when it re-derives an answer,
    especially in the second turn. This is still the model deliberately marking
    its final answer, so we read it -- does not count like guessing.
    """
    start = text.rfind(r"\boxed")
    if start == -1:
        return None
    open_brace = text.find("{", start)
    if open_brace == -1:
        return None

    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:i]
    return None


# Number normalization
# '1,234' -> 1234.0, '$5' -> 5.0, '3/4' -> 0.75, 'twelve' -> None.
def clean_number(text):
    text = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", text)
    for junk in (",", "$", "%", " ", "\\"):
        text = text.replace(junk, "")
    text = text.strip().rstrip(".")

    try:
        return float(text)
    except ValueError:
        pass

    fraction = re.fullmatch(r"(-?\d+)/(-?\d+)", text)
    if fraction and float(fraction.group(2)) != 0:
        return float(fraction.group(1)) / float(fraction.group(2))
    return None


# Text normalization for removing cosmetic differences 
# "Murder", "3500 Hz", "\\frac{1}{3}" vs "murder.", "3,500 Hz", "1/3"
def to_plain(text):
    text = text.strip().lower()
    text = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", text)
    text = text.replace(r"\pi", "π").replace(r"\times", "*")
    for junk in ("\\", "$", "(", ")", ",", " ", "{", "}"):
        text = text.replace(junk, "")
    return text.strip().rstrip(".").strip()


# 'B', '(B)', 'B.', 'B. four', or the option written any reasonable way
def clean_choice(text, choices=None):
    text = text.strip().strip("$").strip().rstrip(".").strip()

    letter = re.match(r"^\(?([A-Da-d])\)?(?:[.):]\s|$)", text)
    if letter:
        return letter.group(1).upper()

    if not choices:
        return None

    wanted = to_plain(text)
    options = [to_plain(str(c)) for c in choices]
    if wanted in options:
        return config.LETTERS[options.index(wanted)]

    # The model often drops the unit: "39.5" for the option "39.5 eV". Accept
    # that only when exactly one option starts that way, so it stays unambiguous.
    starts = [i for i, option in enumerate(options)
              if wanted and option.startswith(wanted)]
    if len(starts) == 1:
        return config.LETTERS[starts[0]]
    return None


# Extract clean answer or None
def extract_answer(reply, answer_type, choices=None):
    tags = ANSWER_TAG.findall(reply or "")
    answer = tags[-1].strip() if tags and tags[-1].strip() else None
    if answer is None:
        boxed = last_boxed(reply or "")
        answer = boxed.strip() if boxed and boxed.strip() else None
    if answer is None:
        return None
    if answer_type == "number":
        return clean_number(answer)
    if answer_type == "choice":
        return clean_choice(answer, choices)
    raise ValueError(f"unknown answer_type: {answer_type}")


# None is never correct
def is_correct(answer, gold_answer, answer_type, choices=None):
    if answer is None:
        return False

    if answer_type == "number":
        gold = clean_number(str(gold_answer))
        return gold is not None and abs(answer - gold) <= config.ANSWER_TOLERANCE

    if answer_type == "choice":
        return answer == clean_choice(str(gold_answer), choices)

    raise ValueError(f"unknown answer_type: {answer_type}")
