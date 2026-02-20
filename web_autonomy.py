import subprocess
import time
import urllib.request

class JanusWeb:
    def __init__(self, os_interface):
        self.os = os_interface

    def browse_to(self, url):
        # Launch browser via shell (Windows handles protocol)
        subprocess.Popen(f"start {url}", shell=True)
        time.sleep(3) # Wait for load
        # Ensure focus and confirm URL
        self.os.type_string(url)
        self.os.press_key(0x0D) # VK_RETURN

    def fetch_headless(self, url):
        try:
            with urllib.request.urlopen(url) as response:
                return response.read().decode('utf-8', errors='ignore')
        except:
            return ""
