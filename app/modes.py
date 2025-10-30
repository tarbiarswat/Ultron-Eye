from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Mode1Inputs:
    left_term: str
    right_term: str
    sources: List[str]

@dataclass
class Mode2Inputs:
    terms: List[str]
    sources: List[str]

def validate_mode1(left: str, right: str) -> Tuple[bool, str]:
    if not left or not right:
        return False, "Please enter both comparison terms."
    if left.strip().lower() == right.strip().lower():
        return False, "Terms must be different."
    return True, ""

def validate_mode2(terms: List[str]) -> Tuple[bool, str]:
    terms = [t for t in (terms or []) if t.strip()]
    if len(terms) < 1:
        return False, "Enter at least one term."
    return True, ""