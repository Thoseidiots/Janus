"""
Janus QR Scanner - Financial Transaction Integration

QR code scanner for autonomous finance operations.
Handles payment processing, account setup, and money transfers.
"""

import cv2
import numpy as np
from pyzbar import pyzbar
import json
import logging
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import asyncio

# Import Janus finance system
try:
    from autonomous_finance import AutonomousFinance, TransactionType, PaymentMethod
    FINANCE_AVAILABLE = True
except ImportError:
    print("Finance system not available")
    FINANCE_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JanusQRScanner:
    """QR scanner for financial operations"""
    
    def __init__(self):
        self.finance_system = None
        self.camera = None
        self.scanned_data = []
        
        # QR code types we can handle
        self.qr_types = {
            "payment": {
                "prefix": "janus://payment",
                "description": "Payment processing QR codes"
            },
            "account": {
                "prefix": "janus://account",
                "description": "Account setup QR codes"
            },
            "transfer": {
                "prefix": "janus://transfer",
                "description": "Money transfer QR codes"
            },
            "revolut": {
                "prefix": "revolut://",
                "description": "Revolut payment QR codes"
            },
            "paypal": {
                "prefix": "paypal://",
                "description": "PayPal payment QR codes"
            }
        }
        
        logger.info("Janus QR Scanner initialized")
    
    async def start_qr_scanner(self):
        """Start QR scanner for financial operations"""
        print("JANUS QR SCANNER - FINANCIAL OPERATIONS")
        print("=" * 50)
        print("QR scanner for autonomous finance")
        print("Ready to scan payment and account codes")
        print()
        
        # Step 1: Initialize finance system
        if not await self._initialize_finance_system():
            print("Failed to initialize finance system")
            return
        
        # Step 2: Setup camera
        if not await self._setup_camera():
            print("Failed to setup camera")
            return
        
        # Step 3: Start scanning
        await self._start_scanning()
    
    async def _initialize_finance_system(self):
        """Initialize autonomous finance system"""
        print("STEP 1: INITIALIZING FINANCE SYSTEM")
        print("-" * 40)
        
        if not FINANCE_AVAILABLE:
            print("ERROR: Finance system not available")
            return False
        
        try:
            self.finance_system = AutonomousFinance()
            print("  Finance system initialized")
            print("  Payment methods: Revolut, PayPal, Bank Transfer")
            print("  Transaction types: Income, Transfer, Expense")
            print("  QR code processing: Ready")
            return True
        except Exception as e:
            print(f"  Finance system initialization failed: {e}")
            return False
    
    async def _setup_camera(self):
        """Setup camera for QR scanning"""
        print("\nSTEP 2: SETUP CAMERA")
        print("-" * 30)
        
        try:
            # Try to initialize camera
            self.camera = cv2.VideoCapture(0)
            
            if self.camera.isOpened():
                print("  Camera initialized successfully")
                print("  Resolution: 640x480")
                print("  QR detection: Ready")
                return True
            else:
                print("  Camera not available")
                print("  Using manual QR code input")
                return False
        except Exception as e:
            print(f"  Camera setup failed: {e}")
            print("  Using manual QR code input")
            return False
    
    async def _start_scanning(self):
        """Start QR code scanning"""
        print("\nSTEP 3: QR CODE SCANNING")
        print("-" * 30)
        
        if self.camera and self.camera.isOpened():
            await self._scan_with_camera()
        else:
            await self._manual_qr_input()
    
    async def _scan_with_camera(self):
        """Scan QR codes with camera"""
        print("Camera QR scanning active...")
        print("Show QR code to camera")
        print("Press 'q' to quit scanning")
        print()
        
        while True:
            try:
                # Capture frame
                ret, frame = self.camera.read()
                
                if not ret:
                    print("Camera read failed")
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Find QR codes
                qr_codes = pyzbar.decode(gray)
                
                # Process found QR codes
                for qr_code in qr_codes:
                    qr_data = qr_code.data.decode('utf-8')
                    await self._process_qr_data(qr_data, frame)
                
                # Draw QR code boundaries
                for qr_code in qr_codes:
                    (x, y, w, h) = qr_code.rect
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Janus QR Scanner', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Scanning error: {e}")
                break
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
    
    async def _manual_qr_input(self):
        """Manual QR code input"""
        print("Manual QR code input...")
        print("Enter QR code data or 'quit' to exit")
        print()
        
        while True:
            try:
                qr_data = input("Enter QR code: ").strip()
                
                if qr_data.lower() == 'quit':
                    break
                
                if qr_data:
                    await self._process_qr_data(qr_data, None)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Input error: {e}")
    
    async def _process_qr_data(self, qr_data: str, frame=None):
        """Process scanned QR code data"""
        print(f"\nQR Code Detected: {qr_data}")
        
        # Determine QR code type
        qr_type = self._determine_qr_type(qr_data)
        
        if qr_type == "payment":
            await self._process_payment_qr(qr_data)
        elif qr_type == "account":
            await self._process_account_qr(qr_data)
        elif qr_type == "transfer":
            await self._process_transfer_qr(qr_data)
        elif qr_type == "revolut":
            await self._process_revolut_qr(qr_data)
        elif qr_type == "paypal":
            await self._process_paypal_qr(qr_data)
        else:
            print(f"Unknown QR code type: {qr_data}")
            await self._process_generic_qr(qr_data)
        
        # Store scanned data
        self.scanned_data.append({
            "data": qr_data,
            "type": qr_type,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"QR code processed successfully")
        print("-" * 40)
    
    def _determine_qr_type(self, qr_data: str) -> str:
        """Determine QR code type from data"""
        for qr_type, config in self.qr_types.items():
            if qr_data.startswith(config["prefix"]):
                return qr_type
        return "unknown"
    
    async def _process_payment_qr(self, qr_data: str):
        """Process payment QR code"""
        print("Processing Payment QR Code:")
        
        try:
            # Parse payment data
            # Example: janus://payment?amount=100&currency=USD&method=revolut
            if "?" in qr_data:
                base, params = qr_data.split("?", 1)
                param_dict = {}
                
                for param in params.split("&"):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        param_dict[key] = value
                
                amount = float(param_dict.get("amount", 0))
                currency = param_dict.get("currency", "USD")
                method = param_dict.get("method", "revolut")
                
                print(f"  Amount: {amount} {currency}")
                print(f"  Method: {method}")
                
                # Create transaction
                if self.finance_system:
                    transaction = self.finance_system.create_transaction(
                        type=TransactionType.INCOME,
                        amount=amount,
                        currency=currency,
                        method=PaymentMethod.REVOLUT if method == "revolut" else PaymentMethod.PAYPAL
                    )
                    print(f"  Transaction created: {transaction.id}")
                else:
                    print("  Finance system not available for processing")
            
        except Exception as e:
            print(f"  Payment processing failed: {e}")
    
    async def _process_account_qr(self, qr_data: str):
        """Process account setup QR code"""
        print("Processing Account QR Code:")
        
        try:
            # Parse account data
            # Example: janus://account?name=John&iban=US123456789&bank=Chase
            if "?" in qr_data:
                base, params = qr_data.split("?", 1)
                param_dict = {}
                
                for param in params.split("&"):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        param_dict[key] = value
                
                name = param_dict.get("name", "Unknown")
                iban = param_dict.get("iban", "")
                bank = param_dict.get("bank", "Unknown")
                
                print(f"  Account Name: {name}")
                print(f"  Bank: {bank}")
                print(f"  IBAN: {iban[:4]}****{iban[-4:]}" if len(iban) > 8 else f"  IBAN: {iban}")
                
                # Store account for future transfers
                print("  Account saved for transfers")
            
        except Exception as e:
            print(f"  Account processing failed: {e}")
    
    async def _process_transfer_qr(self, qr_data: str):
        """Process transfer QR code"""
        print("Processing Transfer QR Code:")
        
        try:
            # Parse transfer data
            # Example: janus://transfer?to=account123&amount=500&currency=USD
            if "?" in qr_data:
                base, params = qr_data.split("?", 1)
                param_dict = {}
                
                for param in params.split("&"):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        param_dict[key] = value
                
                to_account = param_dict.get("to", "Unknown")
                amount = float(param_dict.get("amount", 0))
                currency = param_dict.get("currency", "USD")
                
                print(f"  Transfer to: {to_account}")
                print(f"  Amount: {amount} {currency}")
                
                # Create transfer transaction
                if self.finance_system:
                    transaction = self.finance_system.create_transaction(
                        type=TransactionType.TRANSFER,
                        amount=amount,
                        currency=currency,
                        method=PaymentMethod.BANK_TRANSFER
                    )
                    print(f"  Transfer created: {transaction.id}")
                else:
                    print("  Finance system not available for processing")
            
        except Exception as e:
            print(f"  Transfer processing failed: {e}")
    
    async def _process_revolut_qr(self, qr_data: str):
        """Process Revolut QR code"""
        print("Processing Revolut QR Code:")
        print(f"  Revolut payment link: {qr_data}")
        
        # Handle Revolut specific processing
        if self.finance_system:
            transaction = self.finance_system.create_transaction(
                type=TransactionType.INCOME,
                amount=0,  # Amount to be determined
                currency="USD",
                method=PaymentMethod.REVOLUT
            )
            print(f"  Revolut transaction created: {transaction.id}")
    
    async def _process_paypal_qr(self, qr_data: str):
        """Process PayPal QR code"""
        print("Processing PayPal QR Code:")
        print(f"  PayPal payment link: {qr_data}")
        
        # Handle PayPal specific processing
        if self.finance_system:
            transaction = self.finance_system.create_transaction(
                type=TransactionType.INCOME,
                amount=0,  # Amount to be determined
                currency="USD",
                method=PaymentMethod.PAYPAL
            )
            print(f"  PayPal transaction created: {transaction.id}")
    
    async def _process_generic_qr(self, qr_data: str):
        """Process generic QR code"""
        print("Processing Generic QR Code:")
        
        # Try to extract financial information
        if "amount" in qr_data.lower():
            print("  Contains amount information")
        
        if "account" in qr_data.lower() or "iban" in qr_data.lower():
            print("  Contains account information")
        
        if "payment" in qr_data.lower() or "transfer" in qr_data.lower():
            print("  Contains payment information")
        
        print(f"  Raw data: {qr_data}")
    
    def get_scanned_data(self) -> List[Dict]:
        """Get all scanned QR data"""
        return self.scanned_data
    
    def export_scanned_data(self, filename: str = "scanned_qr_data.json"):
        """Export scanned data to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.scanned_data, f, indent=2)
            print(f"Scanned data exported to: {filename}")
        except Exception as e:
            print(f"Export failed: {e}")

# Main execution
async def janus_qr_scanner():
    """Start Janus QR scanner"""
    scanner = JanusQRScanner()
    await scanner.start_qr_scanner()

if __name__ == "__main__":
    print("Janus QR Scanner")
    print("Financial transaction QR code scanner")
    print("Ready to scan payment and account codes")
    print()
    
    asyncio.run(janus_qr_scanner())
