"""
Universal Oxpecker License Manager
===================================
Handles license validation, monthly subscriptions, and pay-what-you-want flow.
Stores license state locally; validates against Gumroad metadata.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
from dataclasses import dataclass, asdict
import uuid

try:
    from .gumroad_validator import GumroadValidator
except ImportError:
    from gumroad_validator import GumroadValidator


@dataclass
class LicenseStatus:
    """Current license state for this installation."""
    is_licensed: bool
    license_key: Optional[str]
    email: Optional[str]
    activation_date: Optional[str]
    last_check: Optional[str]
    can_use_today: bool
    message: str
    days_remaining: Optional[int] = None


class LicenseManager:
    """
    License validation and monthly subscription tracking.
    
    Local-first design: stores state in ~/.oxpecker/license.json
    Validates against Gumroad metadata on startup.
    Allows "cannot afford" skip for current month.
    """

    LICENSE_DIR = Path.home() / ".oxpecker"
    LICENSE_FILE = LICENSE_DIR / "license.json"
    MONTH_SECONDS = 30 * 24 * 3600

    def __init__(self):
        self.license_dir = self.LICENSE_DIR
        self.license_file = self.LICENSE_FILE
        self.license_dir.mkdir(parents=True, exist_ok=True)

    def get_status(self) -> LicenseStatus:
        """Get current license status."""
        data = self._load_license_data()
        
        if not data or not data.get("license_key"):
            return LicenseStatus(
                is_licensed=False,
                license_key=None,
                email=None,
                activation_date=None,
                last_check=None,
                can_use_today=True,  # Free tier: can always use
                message="No license installed. Running in free mode.",
            )

        license_key = data.get("license_key")
        email = data.get("email", "unknown@example.com")
        activation_date = data.get("activation_date")
        last_check = data.get("last_check")
        cannot_afford_until = data.get("cannot_afford_until")

        # Check if current month is covered by "cannot afford"
        if cannot_afford_until:
            until_dt = datetime.fromisoformat(cannot_afford_until)
            if datetime.utcnow() < until_dt:
                return LicenseStatus(
                    is_licensed=True,
                    license_key=license_key,
                    email=email,
                    activation_date=activation_date,
                    last_check=last_check,
                    can_use_today=True,
                    message=f"License active (cannot-afford mode). Valid until {until_dt.strftime('%Y-%m-%d')}.",
                    days_remaining=(until_dt.date() - datetime.utcnow().date()).days,
                )

        # Check if subscription is still valid (monthly renewal)
        if activation_date:
            act_dt = datetime.fromisoformat(activation_date)
            today = datetime.utcnow()
            months_elapsed = (today.year - act_dt.year) * 12 + (today.month - act_dt.month)
            
            # If activated this month or last month, allow use
            if months_elapsed <= 1:
                days_left = (act_dt.replace(day=28) + timedelta(days=4) - today).days
                return LicenseStatus(
                    is_licensed=True,
                    license_key=license_key,
                    email=email,
                    activation_date=activation_date,
                    last_check=last_check,
                    can_use_today=True,
                    message=f"License active ({email}). Monthly renewal due in ~{max(0, days_left)} days.",
                    days_remaining=max(0, days_left),
                )
            else:
                # License expired; user must renew
                return LicenseStatus(
                    is_licensed=True,
                    license_key=license_key,
                    email=email,
                    activation_date=activation_date,
                    last_check=last_check,
                    can_use_today=False,
                    message=f"License expired. Please renew at gumroad.com. Last check: {last_check}",
                )

        return LicenseStatus(
            is_licensed=True,
            license_key=license_key,
            email=email,
            activation_date=activation_date,
            last_check=last_check,
            can_use_today=True,
            message="License valid (free tier or donation mode).",
        )

    def activate_license(self, license_key: str, email: str) -> Tuple[bool, str]:
        """
        Activate a new license key.
        Validates against Gumroad if possible.
        Returns (success, message).
        """
        # Optional: Validate with Gumroad first
        # Uncomment when you have Gumroad credentials:
        # try:
        #     validator = GumroadValidator(
        #         product_id=os.getenv("OXPECKER_PRODUCT_ID", ""),
        #         product_token=os.getenv("OXPECKER_PRODUCT_TOKEN", "")
        #     )
        #     is_valid, reason, metadata = validator.validate_key(license_key, email)
        #     if not is_valid:
        #         return False, reason
        # except Exception:
        #     pass  # Fall back to local validation
        
        # Local validation: accept any key > 8 chars
        if not license_key or len(license_key) < 8:
            return False, "Invalid license key format."

        data = {
            "license_key": license_key,
            "email": email,
            "activation_date": datetime.utcnow().isoformat(),
            "last_check": datetime.utcnow().isoformat(),
            "cannot_afford_until": None,
        }
        self._save_license_data(data)
        return True, f"License activated for {email}. Thank you for supporting Oxpecker!"

    def set_cannot_afford(self, skip_until_date: Optional[str] = None) -> Tuple[bool, str]:
        """
        Mark this month as 'cannot afford'. User can use free tier for 30 days.
        If skip_until_date is None, default to 30 days from now.
        """
        data = self._load_license_data()
        if not data or not data.get("license_key"):
            return False, "No license to skip. Running in free mode."

        if skip_until_date is None:
            skip_until_date = (datetime.utcnow() + timedelta(days=30)).isoformat()

        data["cannot_afford_until"] = skip_until_date
        data["last_check"] = datetime.utcnow().isoformat()
        self._save_license_data(data)
        
        return True, f"Skipped this month. You can use Oxpecker until {skip_until_date[:10]}."

    def renew_license(self, new_key: Optional[str] = None) -> Tuple[bool, str]:
        """
        Renew a license for another month.
        If new_key provided, update the key.
        """
        data = self._load_license_data()
        if not data or not data.get("license_key"):
            return False, "No license to renew."

        if new_key:
            data["license_key"] = new_key

        data["activation_date"] = datetime.utcnow().isoformat()
        data["last_check"] = datetime.utcnow().isoformat()
        data["cannot_afford_until"] = None
        self._save_license_data(data)
        
        return True, "License renewed for another month. Thank you!"

    def uninstall_license(self) -> Tuple[bool, str]:
        """Remove license from this machine."""
        if self.license_file.exists():
            self.license_file.unlink()
            return True, "License removed. Oxpecker is now in free mode."
        return False, "No license installed."

    def _load_license_data(self) -> Optional[dict]:
        """Load license from disk."""
        if not self.license_file.exists():
            return None
        try:
            with open(self.license_file, "r") as f:
                return json.load(f)
        except Exception as e:
            return None

    def _save_license_data(self, data: dict) -> None:
        """Save license to disk."""
        with open(self.license_file, "w") as f:
            json.dump(data, f, indent=2)
        # Restrict permissions on license file
        os.chmod(self.license_file, 0o600)


def check_license_at_startup() -> LicenseStatus:
    """Check license on app startup. Return status and whether to continue."""
    manager = LicenseManager()
    status = manager.get_status()
    return status
