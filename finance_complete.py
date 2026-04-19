
"""
Complete Finance System - All Issues Fixed

Fully functional finance system with proper imports and all features.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Define enums at module level
class TransactionType:
    INCOME = "income"
    TRANSFER = "transfer"
    EXPENSE = "expense"

class PaymentMethod:
    REVOLUT = "revolut"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"
    CASH = "cash"

class FinancialTransaction:
    """Complete financial transaction model"""
    
    def __init__(self, transaction_id: str, transaction_type, amount: float, 
                 currency: str, method, description: str = "", client: str = ""):
        self.id = transaction_id
        self.type = transaction_type
        self.amount = amount
        self.currency = currency
        self.method = method
        self.description = description
        self.client = client
        self.timestamp = datetime.now()
        self.status = "pending"
        self.qr_data = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value if hasattr(self.type, 'value') else self.type,
            "amount": self.amount,
            "currency": self.currency,
            "method": self.method.value if hasattr(self.method, 'value') else self.method,
            "description": self.description,
            "client": self.client,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "qr_data": self.qr_data
        }

class StandaloneFinance:
    """Complete standalone finance system"""
    
    def __init__(self):
        self.transactions = []
        self.accounts = {}
        self.balance = {}
        self.payment_links = {}
        
        # Initialize with default accounts
        self._initialize_default_accounts()
        
        logger.info("Complete Standalone Finance System initialized")
    
    def _initialize_default_accounts(self):
        """Initialize default accounts"""
        self.accounts = {
            "revolut": {
                "name": "Revolut",
                "currency": "USD",
                "balance": 0.0,
                "active": True,
                "payment_link": "https://revolut.me/i_sears"
            },
            "paypal": {
                "name": "PayPal",
                "currency": "USD", 
                "balance": 0.0,
                "active": True,
                "payment_link": None
            },
            "bank": {
                "name": "Bank Account",
                "currency": "USD",
                "balance": 0.0,
                "active": True,
                "payment_link": None
            }
        }
        
        self.balance = {
            "USD": 0.0,
            "EUR": 0.0,
            "GBP": 0.0
        }
    
    def create_transaction(self, transaction_type, amount: float, currency: str, 
                        method, description: str = "", client: str = ""):
        """Create a new financial transaction"""
        
        transaction_id = f"TXN_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        transaction = FinancialTransaction(
            transaction_id=transaction_id,
            transaction_type=transaction_type,
            amount=amount,
            currency=currency,
            method=method,
            description=description,
            client=client
        )
        
        # Add QR data for Revolut
        if method == PaymentMethod.REVOLUT:
            transaction.qr_data = f"revolut://payment?id={transaction_id}&amount={amount}&currency={currency}"
        
        # Store transaction
        self.transactions.append(transaction)
        
        # Update balance
        self._update_balance(transaction)
        
        logger.info(f"Transaction created: {transaction_id}")
        return transaction
    
    def _update_balance(self, transaction):
        """Update account balance based on transaction"""
        if transaction.type == TransactionType.INCOME:
            if transaction.currency not in self.balance:
                self.balance[transaction.currency] = 0.0
            self.balance[transaction.currency] += transaction.amount
        
        elif transaction.type == TransactionType.EXPENSE:
            if transaction.currency not in self.balance:
                self.balance[transaction.currency] = 0.0
            self.balance[transaction.currency] -= transaction.amount
    
    def get_transaction(self, transaction_id: str) -> Optional[FinancialTransaction]:
        """Get transaction by ID"""
        for transaction in self.transactions:
            if transaction.id == transaction_id:
                return transaction
        return None
    
    def get_balance(self, currency: str = "USD") -> float:
        """Get balance for currency"""
        return self.balance.get(currency, 0.0)
    
    def get_total_income(self, currency: str = "USD") -> float:
        """Get total income for currency"""
        income_transactions = [t for t in self.transactions if t.type == TransactionType.INCOME]
        return sum(t.amount for t in income_transactions if t.currency == currency)
    
    def get_total_expenses(self, currency: str = "USD") -> float:
        """Get total expenses for currency"""
        expense_transactions = [t for t in self.transactions if t.type == TransactionType.EXPENSE]
        return sum(t.amount for t in expense_transactions if t.currency == currency)
    
    def export_transactions(self, filename: str = "transactions.json"):
        """Export transactions to file"""
        try:
            data = {
                "transactions": [t.to_dict() for t in self.transactions],
                "accounts": self.accounts,
                "balance": self.balance,
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Transactions exported to: {filename}")
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def get_financial_summary(self) -> Dict:
        """Get financial summary"""
        income = self.get_total_income()
        expenses = self.get_total_expenses()
        net = income - expenses
        
        return {
            "total_income": income,
            "total_expenses": expenses,
            "net_profit": net,
            "balance": self.balance,
            "total_transactions": len(self.transactions),
            "accounts": self.accounts
        }
