
"""
Finance System Wrapper - Fixed Imports

Wrapper for standalone_finance with proper TransactionType and PaymentMethod imports.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid
import json

logger = logging.getLogger(__name__)

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
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value if hasattr(self.type, 'value') else self.type,
            "amount": self.amount,
            "currency": self.currency,
            "method": self.method.value if hasattr(self.method, 'value') else self.method,
            "description": self.description,
            "client": self.client,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status
        }

class StandaloneFinance:
    def __init__(self):
        self.transactions = []
        self.balance = {"USD": 0.0, "EUR": 0.0, "GBP": 0.0}
        logger.info("Standalone Finance System initialized")
    
    def create_transaction(self, transaction_type, amount: float, currency: str, 
                        method, description: str = "", client: str = ""):
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
        
        self.transactions.append(transaction)
        self._update_balance(transaction)
        
        logger.info(f"Transaction created: {transaction_id}")
        return transaction
    
    def _update_balance(self, transaction):
        if transaction.type == TransactionType.INCOME:
            if transaction.currency not in self.balance:
                self.balance[transaction.currency] = 0.0
            self.balance[transaction.currency] += transaction.amount
    
    def get_balance(self, currency: str = "USD") -> float:
        return self.balance.get(currency, 0.0)
    
    def get_total_income(self, currency: str = "USD") -> float:
        income_transactions = [t for t in self.transactions if t.type == TransactionType.INCOME]
        return sum(t.amount for t in income_transactions if t.currency == currency)
