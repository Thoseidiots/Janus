# sanitization_gateway.py
# Input Sanitization Gateway for Project Manus. This file is immutable.

import re

FORBIDDEN_INPUT_PATTERNS = [
    r"self_modify\s*\(", r"moral_core_immutable\.py", r"agi_core_live_v\d+\.py",
    r"subprocess\.", r"os\.system", r"sys\.exit"
]

def sanitize_input(incoming_text: str) -> (bool, str):
    for pattern in FORBIDDEN_INPUT_PATTERNS:
        if re.search(pattern, incoming_text, re.IGNORECASE):
            alert = f"SANITIZATION ALERT: Malicious pattern '{pattern}' detected and blocked."
            return False, alert
    return True, incoming_text
