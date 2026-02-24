from playwright.sync_api import sync_playwright
import time
import os

class JanusWebAutonomy:
    def __init__(self, headless=True):
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    def start(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        self.page = self.context.new_page()
        print("Janus Web Autonomy started.")

    def navigate(self, url):
        print(f"Navigating to: {url}")
        self.page.goto(url, wait_until="networkidle")
        return self.page.content()

    def click(self, selector):
        print(f"Clicking: {selector}")
        self.page.click(selector)
        self.page.wait_for_load_state("networkidle")

    def fill(self, selector, value):
        print(f"Filling {selector} with value.")
        self.page.fill(selector, value)

    def get_text(self, selector):
        return self.page.inner_text(selector)

    def screenshot(self, path="screenshot.png"):
        self.page.screenshot(path=path)
        print(f"Screenshot saved to {path}")

    def stop(self):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        print("Janus Web Autonomy stopped.")

if __name__ == "__main__":
    # Test the web autonomy layer
    web = JanusWebAutonomy()
    try:
        web.start()
        web.navigate("https://www.google.com")
        web.screenshot("/home/ubuntu/Janus/google_test.png")
        print("Web autonomy test successful.")
    finally:
        web.stop()
