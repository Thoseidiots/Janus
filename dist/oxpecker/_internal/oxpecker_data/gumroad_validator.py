"""
Gumroad License Validation Integration
======================================
Validates license keys against Gumroad's public API.
Falls back to local validation if offline.
"""
from __future__ import annotations

import os
import json
import hashlib
import urllib.request
import urllib.error
from typing import Tuple, Optional
from datetime import datetime, timedelta

class GumroadValidator:
    """
    Validate Oxpecker licenses against Gumroad.
    
    Gumroad license keys can be validated via:
    https://api.gumroad.com/v2/licenses/verify
    
    Required params:
    - product_id: Your Gumroad product ID
    - license_key: The user's license key
    - increment_uses_count: (optional) whether to track activations
    """
    
    GUMROAD_API_URL = "https://api.gumroad.com/v2/licenses/verify"
    
    def __init__(self, product_id: str, product_token: str):
        """
        Initialize Gumroad validator.
        
        Args:
            product_id: Your Gumroad product ID (find in product settings)
            product_token: Your Gumroad product token (for server-side verification)
        """
        self.product_id = product_id
        self.product_token = product_token
    
    def validate_key(self, license_key: str, email: str = "") -> Tuple[bool, str, dict]:
        """
        Validate a license key with Gumroad.
        
        Returns:
            (is_valid, message, metadata)
        """
        if not license_key or len(license_key) < 8:
            return False, "Invalid key format.", {}
        
        try:
            # Prepare request to Gumroad API
            data = {
                "product_id": self.product_id,
                "license_key": license_key,
                "increment_uses_count": "false",  # Don't count every check
            }
            
            if email:
                data["email"] = email
            
            # URL encode the data
            encoded_data = urllib.parse.urlencode(data).encode('utf-8')
            
            # Make request
            req = urllib.request.Request(
                self.GUMROAD_API_URL,
                data=encoded_data,
                method='POST'
            )
            req.add_header('Authorization', f'Bearer {self.product_token}')
            
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            if result.get('success'):
                # License is valid
                metadata = {
                    'license_key': license_key,
                    'email': result.get('purchase', {}).get('email', email),
                    'product_name': result.get('product', {}).get('name'),
                    'purchase_date': result.get('purchase', {}).get('created_at'),
                    'variant': result.get('variant', {}).get('name', 'standard'),
                    'uses_count': result.get('uses_count', 0),
                }
                return True, "License valid.", metadata
            else:
                # License invalid
                reason = result.get('message', 'License not found.')
                return False, reason, {}
        
        except urllib.error.URLError as e:
            # Network error — fall back to local validation
            return self._fallback_local_check(license_key)
        except Exception as e:
            # Other error
            return False, f"Validation error: {str(e)}", {}
    
    def _fallback_local_check(self, license_key: str) -> Tuple[bool, str, dict]:
        """
        Local fallback when offline.
        Just checks format; doesn't actually validate against Gumroad.
        """
        if len(license_key) >= 8:
            return True, "License format valid (offline mode). Verification will occur when online.", {
                'offline_mode': True,
                'license_key': license_key,
            }
        else:
            return False, "Invalid license key format.", {}


# Example usage for Oxpecker
def example_validate_oxpecker_license(license_key: str, email: str = "") -> Tuple[bool, str]:
    """
    Validate an Oxpecker license.
    
    NOTE: Replace PRODUCT_ID and PRODUCT_TOKEN with actual Gumroad values.
    """
    # TODO: Set these from environment or config
    OXPECKER_PRODUCT_ID = os.getenv("OXPECKER_GUMROAD_PRODUCT_ID", "example_product_id")
    OXPECKER_PRODUCT_TOKEN = os.getenv("OXPECKER_GUMROAD_TOKEN", "example_token")
    
    validator = GumroadValidator(OXPECKER_PRODUCT_ID, OXPECKER_PRODUCT_TOKEN)
    is_valid, message, metadata = validator.validate_key(license_key, email)
    
    return is_valid, message


if __name__ == "__main__":
    # Test
    print("Gumroad Validator initialized.")
    print("Set env vars: OXPECKER_GUMROAD_PRODUCT_ID, OXPECKER_GUMROAD_TOKEN")
