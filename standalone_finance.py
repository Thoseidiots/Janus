"""
Standalone Finance System - Complete Payment Processing

Independent finance system that works without external dependencies.
Handles all payment processing for Revolut integration.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from enum import Enum
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransactionType(Enum):
    INCOME = "income"
    TRANSFER = "transfer"
    EXPENSE = "expense"

class PaymentMethod(Enum):
    REVOLUT = "revolut"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"
    CASH = "cash"

class FinancialTransaction:
    """Financial transaction model"""
    
    def __init__(self, transaction_id: str, transaction_type: TransactionType, 
                 amount: float, currency: str, method: PaymentMethod,
                 description: str = "", client: str = ""):
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
            "type": self.type.value,
            "amount": self.amount,
            "currency": self.currency,
            "method": self.method.value,
            "description": self.description,
            "client": self.client,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "qr_data": self.qr_data
        }

class StandaloneFinance:
    """Standalone finance system for payment processing"""
    
    def __init__(self):
        self.transactions = []
        self.accounts = {}
        self.balance = {}
        self.payment_links = {}
        
        # Initialize with default accounts
        self._initialize_default_accounts()
        
        logger.info("Standalone Finance System initialized")
    
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
    
    def create_transaction(self, transaction_type: TransactionType, amount: float, 
                        currency: str, method: PaymentMethod, 
                        description: str = "", client: str = "") -> FinancialTransaction:
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
        
        logger.info(f"Transaction created: {transaction_id} - {transaction_type.value} {amount} {currency}")
        
        return transaction
    
    def _update_balance(self, transaction: FinancialTransaction):
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
    
    def update_transaction_status(self, transaction_id: str, status: str):
        """Update transaction status"""
        transaction = self.get_transaction(transaction_id)
        if transaction:
            transaction.status = status
            logger.info(f"Transaction {transaction_id} status updated to: {status}")
    
    def get_transactions_by_type(self, transaction_type: TransactionType) -> List[FinancialTransaction]:
        """Get transactions by type"""
        return [t for t in self.transactions if t.type == transaction_type]
    
    def get_transactions_by_method(self, method: PaymentMethod) -> List[FinancialTransaction]:
        """Get transactions by payment method"""
        return [t for t in self.transactions if t.method == method]
    
    def get_transactions_by_client(self, client: str) -> List[FinancialTransaction]:
        """Get transactions by client"""
        return [t for t in self.transactions if t.client == client]
    
    def get_balance(self, currency: str = "USD") -> float:
        """Get balance for currency"""
        return self.balance.get(currency, 0.0)
    
    def get_total_income(self, currency: str = "USD") -> float:
        """Get total income for currency"""
        income_transactions = self.get_transactions_by_type(TransactionType.INCOME)
        return sum(t.amount for t in income_transactions if t.currency == currency)
    
    def get_total_expenses(self, currency: str = "USD") -> float:
        """Get total expenses for currency"""
        expense_transactions = self.get_transactions_by_type(TransactionType.EXPENSE)
        return sum(t.amount for t in expense_transactions if t.currency == currency)
    
    def get_account_balance(self, account_name: str) -> float:
        """Get account balance"""
        if account_name in self.accounts:
            return self.accounts[account_name]["balance"]
        return 0.0
    
    def generate_payment_link(self, method: PaymentMethod, amount: float, 
                          currency: str, description: str = "") -> str:
        """Generate payment link"""
        if method == PaymentMethod.REVOLUT:
            return f"https://revolut.me/i_sears"
        elif method == PaymentMethod.PAYPAL:
            return f"https://paypal.me/janus/{amount}{currency}"
        else:
            return f"janus://payment?amount={amount}&currency={currency}&method={method.value}"
    
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
    
    def print_financial_summary(self):
        """Print financial summary"""
        summary = self.get_financial_summary()
        
        print("FINANCIAL SUMMARY")
        print("=" * 40)
        print(f"Total Income: ${summary['total_income']:.2f}")
        print(f"Total Expenses: ${summary['total_expenses']:.2f}")
        print(f"Net Profit: ${summary['net_profit']:.2f}")
        print(f"Total Transactions: {summary['total_transactions']}")
        
        print(f"\nBALANCES:")
        for currency, balance in summary['balance'].items():
            print(f"  {currency}: ${balance:.2f}")
        
        print(f"\nACCOUNTS:")
        for account_name, account_info in summary['accounts'].items():
            status = "Active" if account_info['active'] else "Inactive"
            print(f"  {account_info['name']}: ${account_info['balance']:.2f} ({status})")
            if account_info.get('payment_link'):
                print(f"    Payment Link: {account_info['payment_link']}")

# Test the standalone finance system
if __name__ == "__main__":
    print("Testing Standalone Finance System")
    print("=" * 50)
    
    finance = StandaloneFinance()
    
    # Test transaction creation
    print("Creating test transactions...")
    
    # Income transaction
    income_txn = finance.create_transaction(
        transaction_type=TransactionType.INCOME,
        amount=100.0,
        currency="USD",
        method=PaymentMethod.REVOLUT,
        description="Content writing service",
        client="Tech Company"
    )
    
    # Another income transaction
    income_txn2 = finance.create_transaction(
        transaction_type=TransactionType.INCOME,
        amount=250.0,
        currency="USD",
        method=PaymentMethod.REVOLUT,
        description="Code generation service",
        client="Startup Inc"
    )
    
    # Expense transaction
    expense_txn = finance.create_transaction(
        transaction_type=TransactionType.EXPENSE,
        amount=50.0,
        currency="USD",
        method=PaymentMethod.BANK_TRANSFER,
        description="Software subscription",
        client="Software Provider"
    )
    
    print(f"Created {len(finance.transactions)} transactions")
    
    # Print summary
    finance.print_financial_summary()
    
    # Export transactions
    finance.export_transactions("test_transactions.json")
    
    print(f"\nStandalone Finance System Test Complete")
