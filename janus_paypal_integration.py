"""
janus_paypal_integration.py
============================
PayPal integration for Janus Wallet

Allows Janus to:
- Check PayPal balance
- Receive payments via PayPal
- Send payments via PayPal
- Track PayPal transaction history

Setup Instructions:
1. Go to https://developer.paypal.com/
2. Log in with your Gmail account
3. Create an app to get Client ID and Secret
4. Set environment variables:
   - PAYPAL_CLIENT_ID=your_client_id
   - PAYPAL_CLIENT_SECRET=your_client_secret
   - PAYPAL_MODE=sandbox (for testing) or live (for real money)
"""

import os
import json
import time
import logging
import requests
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PayPalTransaction:
    """PayPal transaction"""
    transaction_id: str
    amount: Decimal
    currency: str
    status: str
    payer_email: str
    payee_email: str
    timestamp: datetime
    description: str = ""
    fee: Decimal = Decimal("0")


# ═══════════════════════════════════════════════════════════════════════════════
# PAYPAL CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class PayPalClient:
    """PayPal API client"""
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        mode: str = "sandbox"
    ):
        """
        Initialize PayPal client
        
        Args:
            client_id: PayPal Client ID
            client_secret: PayPal Client Secret
            mode: "sandbox" for testing or "live" for production
        """
        self.client_id = client_id or os.getenv("PAYPAL_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("PAYPAL_CLIENT_SECRET")
        mode_env = os.getenv("PAYPAL_MODE", "sandbox")
        # Normalize mode to lowercase
        self.mode = (mode or mode_env).lower()
        
        if not self.client_id or not self.client_secret:
            logger.warning("PayPal credentials not configured. Set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET")
        
        # Set API base URL
        if self.mode == "live":
            self.base_url = "https://api-m.paypal.com"
            logger.info("PayPal client initialized in LIVE mode (real money)")
        else:
            self.base_url = "https://api-m.sandbox.paypal.com"
            logger.info("PayPal client initialized in sandbox mode (test money)")
        
        self.access_token = None
        self.token_expires_at = 0
    
    def _get_access_token(self) -> Optional[str]:
        """Get OAuth access token"""
        # Check if token is still valid
        if self.access_token and time.time() < self.token_expires_at:
            return self.access_token
        
        try:
            # Request new token
            url = f"{self.base_url}/v1/oauth2/token"
            headers = {
                "Accept": "application/json",
                "Accept-Language": "en_US"
            }
            data = {"grant_type": "client_credentials"}
            
            response = requests.post(
                url,
                headers=headers,
                data=data,
                auth=(self.client_id, self.client_secret),
                timeout=10
            )
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self.token_expires_at = time.time() + expires_in - 60  # Refresh 1 min early
            
            logger.info("PayPal access token obtained")
            return self.access_token
            
        except Exception as e:
            logger.error(f"Error getting PayPal access token: {e}")
            return None
    
    def get_balance(self) -> Decimal:
        """
        Get PayPal balance
        
        Note: PayPal doesn't provide a direct balance API.
        This would require using PayPal's reporting API or webhooks.
        For now, returns 0 and logs a warning.
        """
        logger.warning("PayPal balance API not available. Use PayPal dashboard to check balance.")
        return Decimal("0")
    
    def create_payment_link(
        self,
        amount: Decimal,
        currency: str = "USD",
        description: str = "Payment to Janus"
    ) -> Optional[str]:
        """
        Create a payment link for receiving money
        
        Returns:
            Payment link URL that can be shared with payers
        """
        token = self._get_access_token()
        if not token:
            logger.warning("Could not get access token. Generating PayPal.me link instead.")
            # Fallback: Generate PayPal.me link (works without API)
            if self.mode == "live":
                # You'll need to set up PayPal.me username in dashboard
                return f"https://paypal.me/TwoDoorsMedia/{amount}USD"
            return None
        
        try:
            url = f"{self.base_url}/v2/checkout/orders"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "Prefer": "return=representation"
            }
            
            data = {
                "intent": "CAPTURE",
                "purchase_units": [{
                    "amount": {
                        "currency_code": currency,
                        "value": f"{amount:.2f}"
                    },
                    "description": description
                }]
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            
            order_data = response.json()
            order_id = order_data["id"]
            
            # Get approval link
            for link in order_data.get("links", []):
                if link["rel"] == "approve":
                    payment_link = link["href"]
                    logger.info(f"Payment link created: {payment_link}")
                    return payment_link
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating payment link: {e}")
            # Fallback to PayPal.me
            if self.mode == "live":
                fallback_link = f"https://paypal.me/TwoDoorsMedia/{amount}USD"
                logger.info(f"Using PayPal.me fallback: {fallback_link}")
                return fallback_link
            return None
    
    def send_payment(
        self,
        recipient_email: str,
        amount: Decimal,
        currency: str = "USD",
        note: str = "Payment from Janus"
    ) -> Optional[str]:
        """
        Send payment to an email address
        
        Returns:
            Payout batch ID or None if failed
        """
        token = self._get_access_token()
        if not token:
            return None
        
        try:
            url = f"{self.base_url}/v1/payments/payouts"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
            
            data = {
                "sender_batch_header": {
                    "sender_batch_id": f"janus_{int(time.time())}",
                    "email_subject": "You have a payment from Janus",
                    "email_message": note
                },
                "items": [{
                    "recipient_type": "EMAIL",
                    "amount": {
                        "value": str(amount),
                        "currency": currency
                    },
                    "receiver": recipient_email,
                    "note": note,
                    "sender_item_id": f"item_{int(time.time())}"
                }]
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            
            payout_data = response.json()
            batch_id = payout_data["batch_header"]["payout_batch_id"]
            
            logger.info(f"Payment sent to {recipient_email}: ${amount} {currency}")
            return batch_id
            
        except Exception as e:
            logger.error(f"Error sending payment: {e}")
            return None
    
    def get_transaction_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10
    ) -> List[PayPalTransaction]:
        """
        Get transaction history
        
        Note: Requires PayPal Transaction Search API access
        """
        token = self._get_access_token()
        if not token:
            return []
        
        try:
            # Default to last 30 days
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            url = f"{self.base_url}/v1/reporting/transactions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
            
            params = {
                "start_date": start_date.strftime("%Y-%m-%dT%H:%M:%S-0700"),
                "end_date": end_date.strftime("%Y-%m-%dT%H:%M:%S-0700"),
                "fields": "all",
                "page_size": limit
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            transactions = []
            
            for tx in data.get("transaction_details", []):
                transactions.append(PayPalTransaction(
                    transaction_id=tx.get("transaction_info", {}).get("transaction_id", ""),
                    amount=Decimal(tx.get("transaction_info", {}).get("transaction_amount", {}).get("value", "0")),
                    currency=tx.get("transaction_info", {}).get("transaction_amount", {}).get("currency_code", "USD"),
                    status=tx.get("transaction_info", {}).get("transaction_status", ""),
                    payer_email=tx.get("payer_info", {}).get("email_address", ""),
                    payee_email=tx.get("payee_info", {}).get("email_address", ""),
                    timestamp=datetime.fromisoformat(tx.get("transaction_info", {}).get("transaction_initiation_date", "")),
                    description=tx.get("transaction_info", {}).get("transaction_subject", ""),
                    fee=Decimal(tx.get("transaction_info", {}).get("fee_amount", {}).get("value", "0"))
                ))
            
            logger.info(f"Retrieved {len(transactions)} PayPal transactions")
            return transactions
            
        except Exception as e:
            logger.error(f"Error getting transaction history: {e}")
            return []
    
    def verify_webhook(self, webhook_data: Dict, webhook_id: str, transmission_id: str, 
                      transmission_time: str, cert_url: str, auth_algo: str, 
                      transmission_sig: str) -> bool:
        """
        Verify PayPal webhook signature
        
        This ensures the webhook actually came from PayPal
        """
        token = self._get_access_token()
        if not token:
            return False
        
        try:
            url = f"{self.base_url}/v1/notifications/verify-webhook-signature"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
            
            data = {
                "transmission_id": transmission_id,
                "transmission_time": transmission_time,
                "cert_url": cert_url,
                "auth_algo": auth_algo,
                "transmission_sig": transmission_sig,
                "webhook_id": webhook_id,
                "webhook_event": webhook_data
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            verification_status = result.get("verification_status", "")
            
            if verification_status == "SUCCESS":
                logger.info("PayPal webhook verified successfully")
                return True
            else:
                logger.warning(f"PayPal webhook verification failed: {verification_status}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying webhook: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# PAYPAL WALLET INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class PayPalWallet:
    """PayPal wallet for Janus"""
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None, mode: Optional[str] = None):
        """Initialize PayPal wallet"""
        # Get mode from parameter or environment
        if mode is None:
            mode = os.getenv("PAYPAL_MODE", "sandbox")
        
        self.client = PayPalClient(client_id, client_secret, mode)
        self.email = os.getenv("PAYPAL_EMAIL", "")
        logger.info(f"PayPal wallet initialized for {self.email or 'unknown email'} in {self.client.mode} mode")
    
    def get_balance(self) -> Decimal:
        """Get PayPal balance"""
        return self.client.get_balance()
    
    def create_payment_request(self, amount: Decimal, description: str = "Payment to Janus") -> Optional[str]:
        """Create a payment request link"""
        return self.client.create_payment_link(amount, "USD", description)
    
    def send_payment(self, recipient_email: str, amount: Decimal, note: str = "") -> Optional[str]:
        """Send payment to email"""
        return self.client.send_payment(recipient_email, amount, "USD", note)
    
    def get_recent_transactions(self, days: int = 30, limit: int = 10) -> List[PayPalTransaction]:
        """Get recent transactions"""
        start_date = datetime.now() - timedelta(days=days)
        return self.client.get_transaction_history(start_date, datetime.now(), limit)


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP GUIDE
# ═══════════════════════════════════════════════════════════════════════════════

def print_setup_guide():
    """Print setup instructions"""
    print("\n" + "="*70)
    print("PAYPAL INTEGRATION SETUP GUIDE")
    print("="*70)
    print("\nTo connect your PayPal account to Janus Wallet:")
    print("\n1. Go to https://developer.paypal.com/")
    print("   - Log in with your Gmail account")
    print("\n2. Create an App:")
    print("   - Click 'Dashboard' → 'My Apps & Credentials'")
    print("   - Click 'Create App'")
    print("   - Name it 'Janus Wallet'")
    print("   - Select 'Merchant' as app type")
    print("\n3. Get Your Credentials:")
    print("   - Copy the 'Client ID'")
    print("   - Click 'Show' under 'Secret' and copy it")
    print("\n4. Set Environment Variables:")
    print("   Windows (PowerShell):")
    print("     $env:PAYPAL_CLIENT_ID='your_client_id_here'")
    print("     $env:PAYPAL_CLIENT_SECRET='your_client_secret_here'")
    print("     $env:PAYPAL_MODE='sandbox'  # Use 'live' for real money")
    print("     $env:PAYPAL_EMAIL='your@gmail.com'")
    print("\n5. Test the Integration:")
    print("   python janus_paypal_integration.py")
    print("\n6. For Production (Real Money):")
    print("   - Go to your app settings")
    print("   - Switch to 'Live' mode")
    print("   - Get Live credentials")
    print("   - Set PAYPAL_MODE='live'")
    print("\n" + "="*70)
    print("\nNOTE: Start with sandbox mode to test without real money!")
    print("="*70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO / TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    """Demo PayPal integration"""
    print("\n" + "="*60)
    print("JANUS PAYPAL INTEGRATION DEMO")
    print("="*60)
    
    # Check if credentials are configured
    if not os.getenv("PAYPAL_CLIENT_ID") or not os.getenv("PAYPAL_CLIENT_SECRET"):
        print("\n⚠️  PayPal credentials not configured!")
        print_setup_guide()
        return
    
    # Initialize PayPal wallet
    wallet = PayPalWallet()
    
    print(f"\nPayPal Email: {wallet.email or 'Not set'}")
    print(f"Mode: {wallet.client.mode}")
    
    # Try to get balance
    print("\nChecking balance...")
    balance = wallet.get_balance()
    print(f"Balance: ${balance} (Note: PayPal doesn't provide direct balance API)")
    
    # Create a payment request
    print("\nCreating payment request for $10...")
    payment_link = wallet.create_payment_request(Decimal("10.00"), "Test payment to Janus")
    if payment_link:
        print(f"Payment link: {payment_link}")
        print("Share this link to receive payment!")
    else:
        print("Failed to create payment link")
    
    # Get recent transactions
    print("\nFetching recent transactions...")
    transactions = wallet.get_recent_transactions(days=30, limit=5)
    if transactions:
        print(f"Found {len(transactions)} transactions:")
        for tx in transactions:
            print(f"  {tx.timestamp.strftime('%Y-%m-%d')} | ${tx.amount} | {tx.status} | {tx.description}")
    else:
        print("No transactions found (or API access not configured)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    demo()
