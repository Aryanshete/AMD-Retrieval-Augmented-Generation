"""All model prompts in one place."""

from __future__ import annotations

SYSTEM = (
    "You are the answer engine of an enterprise document-search system. The company, its "
    "products and its documents are fictional, so you know NOTHING about them except what the "
    "passages you are shown say. Anything you remember from training is wrong by construction. "
    "You are exact, you never guess, and you output only what you are asked for."
)

IMAGE_INDEX_PROMPT = (
    "You are indexing a picture for a search engine. Report, in this order and with no other "
    "commentary:\n"
    "TEXT: every piece of text visible in the picture, copied exactly as printed (keep codes, "
    "dashes, punctuation, '#' and capitalisation), one item per line, in reading order.\n"
    "LINKS: if it is a diagram, label, table or form, one line per relationship between items, "
    "as 'A -> B' (for example which pin or box is connected to which signal name, or which "
    "field name is followed by which value).\n"
    "SUMMARY: one short sentence saying what the picture shows."
)

IMAGE_FOCUS_PROMPT = (
    "Look closely at this picture.\nQuestion: {question}\n"
    "Reply with ONLY the exact text or labels in the picture needed to answer the question, "
    "quoted as printed. For a diagram, say which label is attached to which box or pin. "
    "If the picture does not show the answer, reply exactly: NOT VISIBLE"
)

_RULES = """\
Rules:
1. Use ONLY the passages. Never use outside knowledge, never guess, never estimate.
2. "answer" is the VALUE ONLY - a number, part number, version, code, name or quarter. No \
sentence, no restating the question, no units or degree signs. Keep every part of the value \
that identifies it: "Q3 FY27" not "Q3", "REV-C2" not "C2", "ORR-FAN-2214-B" not "2214-B".
3. The passage must state the requested quantity for EXACTLY the product, item, condition and \
volume asked about. A different product, a legacy item, a different revision, a different \
volume tier or a merely related number is NOT an answer. If the answer is not stated, use "".
4. Passages marked [SUPERSEDED/WITHDRAWN] are no longer valid. Prefer a current passage; use a \
superseded one only if nothing current says it.
5. Code, comments, logs and tables: the live value is the one the code or table actually uses, \
not an old value mentioned in a comment.
6. If the question needs an identifier (ticket, part, error code, device name) that one passage \
gives and the value for it lives in another document that is NOT shown, set "answer" to "" and \
put the exact identifier in "lookup". Otherwise "lookup" is null.
7. "evidence" lists the passage numbers WITHOUT WHICH the answer would be impossible - nothing \
that merely repeats or discusses the same thing. A chain needs both ends: if the question \
refers to an event or document ("the log shows an incident", "that ticket") and one passage \
identifies it (a ticket number, an error code, a part number) while another passage holds the \
value for that identifier, cite BOTH."""

ANSWER_TEMPLATE = """\
Question: {question}

Passages:
{passages}

{rules}

Reply with ONE JSON object and nothing else:
{{"why": "<=20 words", "answer": "<value or empty>", "evidence": [<numbers>], "lookup": null}}"""

VERIFY_TEMPLATE = """\
Question: {question}
Proposed answer: {answer}

Evidence:
{passages}

Does the evidence explicitly state that "{answer}" is the answer to exactly this question - \
the same product, item, quantity, condition and volume, in a current (not withdrawn) document? \
A different product, a legacy item, a different revision or a merely related number means no. \
Reply with ONE JSON object and nothing else: {{"supported": true}} or {{"supported": false}}"""


BRIDGE_TEMPLATE = """\
Question: {question}
The answer "{answer}" was found in passage A. Passage B shares the identifier(s) {shared} with \
passage A.

Passage A:
{a}

Passage B:
{b}

Does the question REFER to something that only passage B identifies (for example "the incident \
in the log" or "that ticket"), so that without passage B you could not know which entry in \
passage A the question is about? If the question already names the entry itself, or passage B \
merely mentions the same thing, the answer is no.
Reply with ONE JSON object and nothing else: {{"needed": true}} or {{"needed": false}}"""


def format_passage(
    pid: int, file: str, loc: str, text: str, superseded: bool, limit: int = 3500
) -> str:
    flag = " [SUPERSEDED/WITHDRAWN]" if superseded else ""
    body = text if len(text) <= limit else text[:limit] + " ..."
    return f"[{pid}] file: {file} ({loc}){flag}\n{body}"


def build_answer_prompt(question: str, passages: list[str]) -> str:
    return ANSWER_TEMPLATE.format(
        question=question.strip(), passages="\n\n".join(passages), rules=_RULES
    )


def build_verify_prompt(question: str, answer: str, passages: list[str]) -> str:
    return VERIFY_TEMPLATE.format(
        question=question.strip(), answer=answer, passages="\n\n".join(passages)
    )


def build_bridge_prompt(question: str, answer: str, shared: str, a: str, b: str) -> str:
    return BRIDGE_TEMPLATE.format(question=question.strip(), answer=answer, shared=shared, a=a, b=b)
