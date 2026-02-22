import time
import random
import base64
import os
from pathlib import Path
import mss
import numpy as np
from PIL import Image
import asyncio

class JanusCaptchaSolver:
    """
    Fully local, zero-cost CAPTCHA solver for Janus.
    Uses screen capture + LLM description + human fallback.
    No training, no APIs, no external services.
    """

    def __init__(self, llm_adapter, memory=None):
        self.llm = llm_adapter          # your existing LLMAdapter / ByteLLM
        self.memory = memory
        self.sct = mss.mss()
        self.solver_dir = Path("janus_captchas")
        self.solver_dir.mkdir(exist_ok=True)

    async def solve(self, url: str = "", challenge_type: str = "unknown") -> str:
        """
        Main entry point. Returns the solution token/text or raises HumanNeeded.
        """
        print(f"[Janus-Solver] Challenge detected at {url} — type: {challenge_type}")

        # Step 1: Take screenshot of whole screen (CAPTCHA is usually visible)
        screenshot = self.sct.grab(self.sct.monitors[1])
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        img_path = self.solver_dir / f"challenge_{int(time.time())}.png"
        img.save(img_path)

        # Step 2: Try simple OCR (if you have pytesseract or easyocr installed)
        text = self._try_local_ocr(img_path)
        if text and len(text) > 3:
            print(f"[Janus-Solver] OCR success: {text}")
            return text

        # Step 3: Ask your LLM to describe what to do
        description = await self._ask_llm_for_solution(img_path, challenge_type, url)
        if description and "click" in description.lower() or "type" in description.lower():
            print(f"[Janus-Solver] LLM suggested: {description}")
            return description  # Janus can act on this in navigator

        # Step 4: Human fallback (this is the honest part — we still need human movement)
        print(f"\n🚨 HUMAN NEEDED 🚨")
        print(f"CAPTCHA at: {url}")
        print(f"Image saved to: {img_path}")
        print("Solve it manually, then type the answer below (or 'skip' to retry later):")
        
        solution = input("→ Your solution: ").strip()
        if solution.lower() == "skip":
            raise Exception("Human skipped CAPTCHA")
        
        print(f"[Janus-Solver] Human provided: {solution}")
        return solution

    def _try_local_ocr(self, img_path):
        """Optional local OCR — works on many simple text CAPTCHAs."""
        try:
            import pytesseract
            text = pytesseract.image_to_string(Image.open(img_path))
            return text.strip()
        except:
            try:
                import easyocr
                reader = easyocr.Reader(['en'], gpu=False)
                result = reader.readtext(str(img_path))
                return ' '.join([res[1] for res in result]).strip()
            except:
                return None

    async def _ask_llm_for_solution(self, img_path, challenge_type, url):
        """Use your existing LLM to describe the CAPTCHA."""
        try:
            # Convert image to base64 so LLM can "see" it (if your LLM supports vision)
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            prompt = f"""
You are Janus. A CAPTCHA appeared on {url}.
Challenge type: {challenge_type}

Describe exactly what the user needs to do (e.g. "type the text XYZ", "click the images with traffic lights", "check the box", "solve 5+3=").
Be precise and actionable. Only output the direct instruction or answer.

Image base64: {b64[:500]}... (truncated)
"""
            response = self.llm(prompt)  # your LLMAdapter
            return response.strip()
        except:
            return None


# ── Integration example with your navigator ─────────────────────────────

# In your JanusCognitiveNavigator class, add:
async def handle_challenge(self, url):
    solver = JanusCaptchaSolver(llm_adapter=your_llm_adapter, memory=your_memory)
    try:
        solution = await solver.solve(url, challenge_type="recaptcha_v2")  # or detect type
        # Then act on solution (click, type, etc.)
        print(f"Solution received: {solution}")
        # ... use your action chains to apply it
    except Exception as e:
        print("Human intervention required for CAPTCHA")
        # Optionally trigger a tree planner branch: "Human needed for verification"