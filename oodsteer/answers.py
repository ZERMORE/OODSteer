"""Versioned, gold-independent scoring and complete-answer stopping rules."""
import re
from decimal import Decimal, InvalidOperation

SCORING_VERSION = "compatible-v2"
QUESTION = re.compile(r"(?<!\w)Q:\s*")
NUMBER = r"[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
MARKER = r"(?:so\s*,?\s*)?(?:the\s+)?(?:(?:final|correct)\s+)?answer\s+(?:is|must\s+be)\s*:?\s*(?:\*\*)?"


def first_response(text):
    m=QUESTION.search(text)
    return text[:m.start()] if m else text


def number(value):
    value=value.strip().replace(r"\,", "").replace(r"\$", "").replace("$", "")
    if not re.fullmatch(NUMBER,value):
        return None
    try:
        d=Decimal(value.replace(",", ""))
        return str(d.normalize()) if d.is_finite() else None
    except InvalidOperation:
        return None


def events(text,task,complete=False):
    text=first_response(text)
    found=[]
    if task=="gsm8k":
        for m in re.finditer(r"\\boxed\s*\{([^{}]*)\}",text):
            value=number(m[1])
            if value is not None:
                found.append((m.start(),m.end(),value,"boxed"))
        for m in re.finditer(MARKER+r"(?:\$|\\\(|\\\[)?\s*("+NUMBER+r")",text,re.I):
            # Never accept a valid prefix of an unfinished numeric token/exponent.
            tail=text[m.end():]
            if tail and tail[0] in "0123456789.,eE/":
                if not re.match(r"\.\s",tail) and not (tail=="." and not complete):
                    continue
            if complete and not re.match(r"(?:\s|\$|\\[)\]]|\.\s)",tail):
                continue
            value=number(m[1])
            if value is not None:
                found.append((m.start(),m.end(),value,"number-marker"))
    else:
        value=r"(?:\(([a-e])\)|(?-i:([A-E]))(?=[\s.,;:]|$))" if task in ("csqa","aqua") else r"(yes|no)\b"
        for m in re.finditer(MARKER+value,text,re.I):
            tail=text[m.end():]
            if task=="strategyqa":
                if complete and not tail:
                    continue
                answer=m[1].lower()
            else:
                # Parenthesized choices are closed; a bare letter needs a delimiter.
                if complete and m[1] is None and not tail:
                    continue
                answer=(m[1] or m[2]).upper()
            found.append((m.start(),m.end(),answer,"choice-marker"))
    return sorted(found)


def extract(text,task):
    matches=events(text,task)
    # First explicit answer agrees with the answer-stop generation policy.
    return matches[0][2] if matches else None


def stop_reason(text,task,policy="answer"):
    if policy=="eos":
        return None
    if QUESTION.search(text):
        return "next-question"
    if policy=="answer" and events(text,task,complete=True):
        return "complete-answer"
    return None


def make_stopper(tokenizer,prompt_length,task,policy):
    from transformers import StoppingCriteria, StoppingCriteriaList
    class AnswerStopper(StoppingCriteria):
        def __init__(self):
            self.reason=None
        def __call__(self,input_ids,scores,**kwargs):
            if input_ids.shape[0]!=1:
                raise ValueError("Answer stopping currently supports batch size 1")
            text=tokenizer.decode(input_ids[0,prompt_length:],skip_special_tokens=True)
            self.reason=stop_reason(text,task,policy)
            return self.reason is not None
    stopper=AnswerStopper()
    return stopper,StoppingCriteriaList([stopper])
