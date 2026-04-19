"""
janus_wallet.py
===============
Autonomous Financial Wallet for Janus

Janus's personal wallet to manage earnings, expenses, and financial goals.
Supports multiple currencies (USD, BTC, ETH, USDC) and provides financial intelligence.

Features:
- Multi-currency balance tracking
- Transaction recording and categorization
- Cryptocurrency wallet support
- Budget management
- Financial reporting and analytics
- Autonomous financial decision-making
"""

import os
import sys
import json
import time
import sqlite3
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_DOWN
from enum import Enum
import uuid
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Crypto wallet support
try:
    from janus_crypto_wallets import CryptoWalletManager, BitcoinWallet, EthereumWallet, USDCWallet
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logging.warning("Crypto wallet support not available")

# PayPal support
try:
    from janus_paypal_integration import PayPalWallet
    HAS_PAYPAL = True
except ImportError:
    HAS_PAYPAL = False
    logging.warning("PayPal support not available")

# Financial intelligence
try:
    from janus_financial_intelligence import FinancialIntelligence, FinancialGoal
    HAS_FINANCIAL_INTELLIGENCE = True
except ImportError:
    HAS_FINANCIAL_INTELLIGENCE = False
    logging.warning("Financial intelligence not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class CurrencyType(Enum):
    """Currency type enum"""
    FIAT = "fiat"
    CRYPTO = "crypto"


@dataclass
class Currency:
    """Represents a currency"""
    code: str  # "USD", "BTC", "ETH", "USDC"
    type: CurrencyType
    decimals: int
    symbol: str
    name: str = ""
    
    def format_amount(self, amount: Decimal) -> str:
        """Format amount with currency symbol"""
        if self.type == CurrencyType.FIAT:
            return f"{self.symbol}{amount:,.2f}"
        else:
            return f"{amount:.{self.decimals}f} {self.code}"


# Predefined currencies
CURRENCIES = {
    "USD": Currency("USD", CurrencyType.FIAT, 2, "$", "US Dollar"),
    "BTC": Currency("BTC", CurrencyType.CRYPTO, 8, "₿", "Bitcoin"),
    "ETH": Currency("ETH", CurrencyType.CRYPTO, 18, "Ξ", "Ethereum"),
    "USDC": Currency("USDC", CurrencyType.CRYPTO, 6, "$", "USD Coin"),
}


@dataclass
class Balance:
    """Represents a balance in a currency"""
    currency: Currency
    amount: Decimal
    pending_amount: Decimal = Decimal("0")
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def available(self) -> Decimal:
        """Available balance (excluding pending)"""
        return self.amount
    
    @property
    def total(self) -> Decimal:
        """Total balance (including pending)"""
        return self.amount + self.pending_amount
    
    def format(self) -> str:
        """Format balance for display"""
        return self.currency.format_amount(self.amount)


class TransactionType(Enum):
    """Transaction type enum"""
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"


class TransactionStatus(Enum):
    """Transaction status enum"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Category(Enum):
    """Transaction category enum"""
    # Income categories
    FREELANCE_EARNINGS = "freelance_earnings"
    CRYPTO_EARNINGS = "crypto_earnings"
    AFFILIATE_COMMISSION = "affiliate_commission"
    PASSIVE_INCOME = "passive_income"
    OTHER_INCOME = "other_income"
    
    # Expense categories
    COMPUTE_RESOURCES = "compute_resources"
    API_COSTS = "api_costs"
    TRAINING_DATA = "training_data"
    INFRASTRUCTURE = "infrastructure"
    DEVELOPMENT_TOOLS = "development_tools"
    SAVINGS = "savings"
    INVESTMENT = "investment"
    OTHER_EXPENSE = "other_expense"
    
    @classmethod
    def is_income(cls, category: 'Category') -> bool:
        """Check if category is income"""
        income_categories = [
            cls.FREELANCE_EARNINGS,
            cls.CRYPTO_EARNINGS,
            cls.AFFILIATE_COMMISSION,
            cls.PASSIVE_INCOME,
            cls.OTHER_INCOME
        ]
        return category in income_categories
    
    @classmethod
    def is_expense(cls, category: 'Category') -> bool:
        """Check if category is expense"""
        return not cls.is_income(category)


@dataclass
class Transaction:
    """Represents a financial transaction"""
    tx_id: str
    type: TransactionType
    amount: Decimal
    currency: Currency
    category: Category
    source: str
    destination: str
    status: TransactionStatus
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tx_id": self.tx_id,
            "type": self.type.value,
            "amount": str(self.amount),
            "currency": self.currency.code,
            "category": self.category.value,
            "source": self.source,
            "destination": self.destination,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "verified": self.verified,
            "notes": self.notes
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ENCRYPTION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class EncryptionManager:
    """Manages encryption for sensitive data"""
    
    def __init__(self, password: Optional[str] = None):
        """Initialize encryption manager"""
        self.password = password or os.getenv("JANUS_WALLET_PASSWORD", "janus_default_password")
        self.key = self._derive_key(self.password)
        self.cipher = Fernet(self.key)
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"janus_wallet_salt",  # In production, use random salt
            iterations=100000,
        )
        key = kdf.derive(password.encode())
        # Convert to Fernet key format (base64 encoded)
        import base64
        return base64.urlsafe_b64encode(key)
    
    def encrypt(self, data: str) -> bytes:
        """Encrypt data"""
        return self.cipher.encrypt(data.encode())
    
    def decrypt(self, encrypted_data: bytes) -> str:
        """Decrypt data"""
        return self.cipher.decrypt(encrypted_data).decode()
    
    def encrypt_dict(self, data: Dict) -> bytes:
        """Encrypt dictionary"""
        json_str = json.dumps(data)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted_data: bytes) -> Dict:
        """Decrypt dictionary"""
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class WalletDatabase:
    """Manages wallet database"""
    
    def __init__(self, db_path: str = "janus_wallet.db"):
        """Initialize database"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Wallets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wallets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                currency TEXT NOT NULL UNIQUE,
                currency_type TEXT NOT NULL,
                balance TEXT NOT NULL DEFAULT '0',
                pending_balance TEXT NOT NULL DEFAULT '0',
                address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_id TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL,
                amount TEXT NOT NULL,
                currency TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT,
                destination TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                verified BOOLEAN DEFAULT FALSE,
                timestamp TIMESTAMP NOT NULL,
                metadata TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Financial goals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                target_amount TEXT NOT NULL,
                current_amount TEXT NOT NULL DEFAULT '0',
                deadline TIMESTAMP,
                priority INTEGER DEFAULT 0,
                category TEXT,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Budgets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                limit_amount TEXT NOT NULL,
                spent_amount TEXT NOT NULL DEFAULT '0',
                period TEXT NOT NULL,
                period_start TIMESTAMP NOT NULL,
                period_end TIMESTAMP NOT NULL,
                rollover BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Crypto keys table (encrypted)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crypto_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                currency TEXT NOT NULL UNIQUE,
                address TEXT NOT NULL,
                encrypted_private_key BLOB NOT NULL,
                public_key TEXT NOT NULL,
                derivation_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_currency ON transactions(currency)')
        
        conn.commit()
        conn.close()
        logger.info(f"Wallet database initialized: {self.db_path}")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)


# ═══════════════════════════════════════════════════════════════════════════════
# WALLET CORE
# ═══════════════════════════════════════════════════════════════════════════════

class WalletCore:
    """Core wallet management"""
    
    def __init__(self, db_path: str = "janus_wallet.db"):
        """Initialize wallet core"""
        self.db = WalletDatabase(db_path)
        self.balances: Dict[str, Balance] = {}
        self.encryption = EncryptionManager()
        self._load_balances()
        logger.info("Wallet core initialized")
    
    def _load_balances(self):
        """Load balances from database"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT currency, balance, pending_balance, updated_at FROM wallets')
        rows = cursor.fetchall()
        
        for row in rows:
            currency_code, balance_str, pending_str, updated_at = row
            if currency_code in CURRENCIES:
                currency = CURRENCIES[currency_code]
                self.balances[currency_code] = Balance(
                    currency=currency,
                    amount=Decimal(balance_str),
                    pending_amount=Decimal(pending_str),
                    last_updated=datetime.fromisoformat(updated_at) if updated_at else datetime.now()
                )
        
        conn.close()
        
        # Initialize default currencies if not present
        for currency_code in CURRENCIES:
            if currency_code not in self.balances:
                self._init_currency(currency_code)
    
    def _init_currency(self, currency_code: str):
        """Initialize a currency in the wallet"""
        if currency_code not in CURRENCIES:
            raise ValueError(f"Unknown currency: {currency_code}")
        
        currency = CURRENCIES[currency_code]
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO wallets (currency, currency_type, balance, pending_balance)
                VALUES (?, ?, '0', '0')
            ''', (currency_code, currency.type.value))
            conn.commit()
            
            self.balances[currency_code] = Balance(
                currency=currency,
                amount=Decimal("0"),
                pending_amount=Decimal("0")
            )
            
            logger.info(f"Initialized {currency_code} wallet")
        except sqlite3.IntegrityError:
            # Already exists
            pass
        finally:
            conn.close()
    
    def get_balance(self, currency_code: str) -> Balance:
        """Get balance for a currency"""
        if currency_code not in self.balances:
            self._init_currency(currency_code)
        return self.balances[currency_code]
    
    def get_all_balances(self) -> Dict[str, Balance]:
        """Get all balances"""
        return self.balances.copy()
    
    def get_total_balance_usd(self) -> Decimal:
        """Get total balance in USD (simplified - no conversion yet)"""
        # For now, just return USD balance
        # TODO: Add currency conversion
        return self.balances.get("USD", Balance(CURRENCIES["USD"], Decimal("0"))).amount
    
    def update_balance(self, currency_code: str, amount: Decimal, pending: Decimal = Decimal("0")):
        """Update balance in database"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE wallets 
            SET balance = ?, pending_balance = ?, updated_at = CURRENT_TIMESTAMP
            WHERE currency = ?
        ''', (str(amount), str(pending), currency_code))
        
        conn.commit()
        conn.close()
        
        # Update in-memory balance
        if currency_code in self.balances:
            self.balances[currency_code].amount = amount
            self.balances[currency_code].pending_amount = pending
            self.balances[currency_code].last_updated = datetime.now()
    
    def add_to_balance(self, currency_code: str, amount: Decimal):
        """Add to balance"""
        current_balance = self.get_balance(currency_code)
        new_balance = current_balance.amount + amount
        self.update_balance(currency_code, new_balance, current_balance.pending_amount)
        logger.info(f"Added {amount} {currency_code} to balance. New balance: {new_balance}")
    
    def subtract_from_balance(self, currency_code: str, amount: Decimal) -> bool:
        """Subtract from balance (returns False if insufficient funds)"""
        current_balance = self.get_balance(currency_code)
        
        if current_balance.amount < amount:
            logger.warning(f"Insufficient funds: {current_balance.amount} < {amount} {currency_code}")
            return False
        
        new_balance = current_balance.amount - amount
        self.update_balance(currency_code, new_balance, current_balance.pending_amount)
        logger.info(f"Subtracted {amount} {currency_code} from balance. New balance: {new_balance}")
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class TransactionManager:
    """Manages transactions"""
    
    def __init__(self, wallet_core: WalletCore):
        """Initialize transaction manager"""
        self.wallet_core = wallet_core
        self.db = wallet_core.db
    
    def record_transaction(self, transaction: Transaction) -> str:
        """Record a transaction"""
        # Verify transaction
        if not self._verify_transaction(transaction):
            logger.error(f"Transaction verification failed: {transaction.tx_id}")
            return ""
        
        # Check for duplicates
        if self._is_duplicate(transaction):
            logger.warning(f"Duplicate transaction detected: {transaction.tx_id}")
            return transaction.tx_id
        
        # Store in database
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO transactions 
                (tx_id, type, amount, currency, category, source, destination, 
                 status, verified, timestamp, metadata, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction.tx_id,
                transaction.type.value,
                str(transaction.amount),
                transaction.currency.code,
                transaction.category.value,
                transaction.source,
                transaction.destination,
                transaction.status.value,
                transaction.verified,
                transaction.timestamp.isoformat(),
                json.dumps(transaction.metadata),
                transaction.notes
            ))
            
            conn.commit()
            
            # Update balance if transaction is completed
            if transaction.status == TransactionStatus.COMPLETED:
                self._update_balance_for_transaction(transaction)
            
            logger.info(f"Transaction recorded: {transaction.tx_id} - {transaction.type.value} {transaction.amount} {transaction.currency.code}")
            return transaction.tx_id
            
        except sqlite3.IntegrityError as e:
            logger.error(f"Error recording transaction: {e}")
            return ""
        finally:
            conn.close()
    
    def _verify_transaction(self, transaction: Transaction) -> bool:
        """Verify transaction data"""
        # Check required fields
        if not transaction.tx_id or not transaction.amount:
            return False
        
        # Check amount is positive
        if transaction.amount <= 0:
            return False
        
        # Check currency is valid
        if transaction.currency.code not in CURRENCIES:
            return False
        
        # For expenses, check sufficient balance
        if transaction.type == TransactionType.EXPENSE:
            balance = self.wallet_core.get_balance(transaction.currency.code)
            if balance.amount < transaction.amount:
                logger.warning(f"Insufficient balance for expense: {balance.amount} < {transaction.amount}")
                return False
        
        return True
    
    def _is_duplicate(self, transaction: Transaction) -> bool:
        """Check if transaction is duplicate"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM transactions WHERE tx_id = ?', (transaction.tx_id,))
        count = cursor.fetchone()[0]
        
        conn.close()
        return count > 0
    
    def _update_balance_for_transaction(self, transaction: Transaction):
        """Update balance based on transaction"""
        if transaction.type == TransactionType.INCOME:
            self.wallet_core.add_to_balance(transaction.currency.code, transaction.amount)
        elif transaction.type == TransactionType.EXPENSE:
            self.wallet_core.subtract_from_balance(transaction.currency.code, transaction.amount)
    
    def get_transaction(self, tx_id: str) -> Optional[Transaction]:
        """Get transaction by ID"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM transactions WHERE tx_id = ?', (tx_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return self._row_to_transaction(row)
    
    def get_transactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        category: Optional[Category] = None,
        transaction_type: Optional[TransactionType] = None,
        limit: int = 100
    ) -> List[Transaction]:
        """Get transactions with filters"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM transactions WHERE 1=1'
        params = []
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date.isoformat())
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date.isoformat())
        
        if category:
            query += ' AND category = ?'
            params.append(category.value)
        
        if transaction_type:
            query += ' AND type = ?'
            params.append(transaction_type.value)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_transaction(row) for row in rows]
    
    def _row_to_transaction(self, row: tuple) -> Transaction:
        """Convert database row to Transaction object"""
        (id, tx_id, type_str, amount_str, currency_code, category_str,
         source, destination, status_str, verified, timestamp_str,
         metadata_str, notes, created_at) = row
        
        return Transaction(
            tx_id=tx_id,
            type=TransactionType(type_str),
            amount=Decimal(amount_str),
            currency=CURRENCIES[currency_code],
            category=Category(category_str),
            source=source or "",
            destination=destination or "",
            status=TransactionStatus(status_str),
            timestamp=datetime.fromisoformat(timestamp_str),
            metadata=json.loads(metadata_str) if metadata_str else {},
            verified=bool(verified),
            notes=notes or ""
        )
    
    def categorize_transaction(self, transaction: Transaction) -> Category:
        """Auto-categorize transaction based on source/destination"""
        # Simple rule-based categorization
        source_lower = transaction.source.lower()
        dest_lower = transaction.destination.lower()
        
        if transaction.type == TransactionType.INCOME:
            if "upwork" in source_lower or "fiverr" in source_lower:
                return Category.FREELANCE_EARNINGS
            elif "crypto" in source_lower or "btc" in source_lower or "eth" in source_lower:
                return Category.CRYPTO_EARNINGS
            elif "affiliate" in source_lower:
                return Category.AFFILIATE_COMMISSION
            else:
                return Category.OTHER_INCOME
        
        else:  # EXPENSE
            if "gpu" in dest_lower or "compute" in dest_lower or "aws" in dest_lower:
                return Category.COMPUTE_RESOURCES
            elif "api" in dest_lower or "openai" in dest_lower or "anthropic" in dest_lower:
                return Category.API_COSTS
            elif "data" in dest_lower or "dataset" in dest_lower:
                return Category.TRAINING_DATA
            elif "savings" in dest_lower or "save" in dest_lower:
                return Category.SAVINGS
            else:
                return Category.OTHER_EXPENSE


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WALLET CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class JanusWallet:
    """Main wallet class - unified interface"""
    
    # Business information
    BUSINESS_NAME = "Two Doors Media"
    BUSINESS_EMAIL = "Legac3y@gmail.com"
    
    def __init__(self, db_path: str = "janus_wallet.db"):
        """Initialize Janus wallet"""
        self.core = WalletCore(db_path)
        self.transactions = TransactionManager(self.core)
        
        # Initialize crypto wallet manager
        if HAS_CRYPTO:
            self.crypto = CryptoWalletManager(self.core.encryption)
            logger.info("Crypto wallet support enabled")
        else:
            self.crypto = None
            logger.warning("Crypto wallet support disabled")
        
        # Initialize PayPal wallet
        if HAS_PAYPAL:
            self.paypal = PayPalWallet()
            logger.info("PayPal support enabled")
        else:
            self.paypal = None
            logger.warning("PayPal support disabled")
        
        # Initialize financial intelligence
        if HAS_FINANCIAL_INTELLIGENCE:
            self.intelligence = FinancialIntelligence(self)
            logger.info("Financial intelligence enabled")
        else:
            self.intelligence = None
            logger.warning("Financial intelligence disabled")
        
        logger.info("Janus Wallet initialized successfully")
    
    def get_balance(self, currency: str = "USD") -> Decimal:
        """Get balance for currency"""
        return self.core.get_balance(currency).amount
    
    def get_all_balances(self) -> Dict[str, Decimal]:
        """Get all balances"""
        return {code: balance.amount for code, balance in self.core.get_all_balances().items()}
    
    def record_income(
        self,
        amount: float,
        currency: str = "USD",
        source: str = "",
        category: Optional[Category] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Record income"""
        tx = Transaction(
            tx_id=f"income_{uuid.uuid4().hex[:16]}",
            type=TransactionType.INCOME,
            amount=Decimal(str(amount)),
            currency=CURRENCIES[currency],
            category=category or Category.OTHER_INCOME,
            source=source,
            destination="janus_wallet",
            status=TransactionStatus.COMPLETED,
            timestamp=datetime.now(),
            metadata=metadata or {},
            verified=True
        )
        
        return self.transactions.record_transaction(tx)
    
    def record_expense(
        self,
        amount: float,
        currency: str = "USD",
        destination: str = "",
        category: Optional[Category] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Record expense"""
        tx = Transaction(
            tx_id=f"expense_{uuid.uuid4().hex[:16]}",
            type=TransactionType.EXPENSE,
            amount=Decimal(str(amount)),
            currency=CURRENCIES[currency],
            category=category or Category.OTHER_EXPENSE,
            source="janus_wallet",
            destination=destination,
            status=TransactionStatus.COMPLETED,
            timestamp=datetime.now(),
            metadata=metadata or {},
            verified=True
        )
        
        return self.transactions.record_transaction(tx)
    
    def get_recent_transactions(self, limit: int = 10) -> List[Transaction]:
        """Get recent transactions"""
        return self.transactions.get_transactions(limit=limit)
    
    def setup_crypto_wallets(self, load_existing: bool = False, btc_key: str = "", eth_key: str = ""):
        """Setup cryptocurrency wallets"""
        if not HAS_CRYPTO or not self.crypto:
            logger.error("Crypto wallet support not available")
            return False
        
        try:
            if load_existing and btc_key:
                # Load existing Bitcoin wallet
                self.crypto.load_bitcoin_wallet(btc_key, testnet=False)
            else:
                # Create new Bitcoin wallet
                self.crypto.create_bitcoin_wallet(testnet=False)
            
            if load_existing and eth_key:
                # Load existing Ethereum wallet
                self.crypto.load_ethereum_wallet(eth_key)
            else:
                # Create new Ethereum wallet
                self.crypto.create_ethereum_wallet()
            
            logger.info("Crypto wallets setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up crypto wallets: {e}")
            return False
    
    def get_crypto_addresses(self) -> Dict[str, str]:
        """Get all cryptocurrency addresses"""
        if not HAS_CRYPTO or not self.crypto:
            return {}
        return self.crypto.get_all_addresses()
    
    def get_crypto_balance(self, currency: str) -> Decimal:
        """Get cryptocurrency balance"""
        if not HAS_CRYPTO or not self.crypto:
            return Decimal("0")
        return self.crypto.get_balance(currency)
    
    def sync_crypto_balances(self):
        """Sync cryptocurrency balances with blockchain"""
        if not HAS_CRYPTO or not self.crypto:
            logger.warning("Crypto wallet support not available")
            return
        
        try:
            # Get balances from blockchain
            crypto_balances = self.crypto.get_all_balances()
            
            # Update wallet balances
            for currency, amount in crypto_balances.items():
                current_balance = self.core.get_balance(currency)
                if current_balance.amount != amount:
                    self.core.update_balance(currency, amount)
                    logger.info(f"Synced {currency} balance: {amount}")
            
        except Exception as e:
            logger.error(f"Error syncing crypto balances: {e}")
    
    def create_paypal_payment_link(self, amount: float, description: str = "Payment to Two Doors Media") -> Optional[str]:
        """Create PayPal payment link"""
        if not HAS_PAYPAL or not self.paypal:
            logger.error("PayPal support not available")
            return None
        
        return self.paypal.create_payment_request(Decimal(str(amount)), description)
    
    def send_paypal_payment(self, recipient_email: str, amount: float, note: str = "") -> Optional[str]:
        """Send PayPal payment"""
        if not HAS_PAYPAL or not self.paypal:
            logger.error("PayPal support not available")
            return None
        
        # Record as expense
        tx_id = self.record_expense(
            amount=amount,
            currency="USD",
            destination=recipient_email,
            category=Category.OTHER_EXPENSE,
            metadata={"note": note, "method": "paypal"}
        )
        
        # Send via PayPal
        batch_id = self.paypal.send_payment(recipient_email, Decimal(str(amount)), note)
        
        if batch_id:
            logger.info(f"PayPal payment sent: {batch_id}")
        
        return batch_id
    
    def get_paypal_email(self) -> str:
        """Get PayPal email"""
        if not HAS_PAYPAL or not self.paypal:
            return ""
        return self.paypal.email
    
    def create_financial_goal(
        self,
        name: str,
        target_amount: float,
        deadline_days: Optional[int] = None,
        category: str = "general"
    ) -> Optional[str]:
        """Create a financial goal"""
        if not HAS_FINANCIAL_INTELLIGENCE or not self.intelligence:
            logger.error("Financial intelligence not available")
            return None
        
        deadline = None
        if deadline_days:
            deadline = datetime.now() + timedelta(days=deadline_days)
        
        goal = self.intelligence.create_goal(
            name=name,
            target_amount=Decimal(str(target_amount)),
            deadline=deadline,
            category=category
        )
        
        return goal.goal_id
    
    def get_financial_report(self, days: int = 30):
        """Get and print financial report"""
        if not HAS_FINANCIAL_INTELLIGENCE or not self.intelligence:
            logger.error("Financial intelligence not available")
            return
        
        self.intelligence.print_report(days)
    
    def print_summary(self):
        """Print wallet summary"""
        print("\n" + "="*60)
        print(f"{self.BUSINESS_NAME.upper()} - WALLET SUMMARY")
        print("="*60)
        
        print("\nBALANCES:")
        print("-" * 30)
        for code, balance in self.core.get_all_balances().items():
            print(f"  {code}: {balance.format()}")
        
        # Show crypto addresses if available
        if HAS_CRYPTO and self.crypto:
            addresses = self.get_crypto_addresses()
            if addresses:
                print(f"\nCRYPTO ADDRESSES:")
                print("-" * 30)
                for currency, address in addresses.items():
                    print(f"  {currency}: {address}")
        
        # Show PayPal info if available
        if HAS_PAYPAL and self.paypal:
            paypal_email = self.get_paypal_email()
            if paypal_email:
                print(f"\nPAYPAL:")
                print("-" * 30)
                print(f"  Email: {paypal_email}")
                print(f"  Mode: {self.paypal.client.mode}")
        
        print(f"\nRECENT TRANSACTIONS:")
        print("-" * 30)
        recent = self.get_recent_transactions(5)
        for tx in recent:
            sign = "+" if tx.type == TransactionType.INCOME else "-"
            print(f"  {tx.timestamp.strftime('%Y-%m-%d %H:%M')} | {sign}{tx.amount} {tx.currency.code} | {tx.category.value}")
        
        print("\n" + "="*60)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO / TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    """Demo wallet functionality"""
    print("Initializing Janus Wallet...")
    wallet = JanusWallet()
    
    # Setup crypto wallets
    print("\nSetting up crypto wallets...")
    wallet.setup_crypto_wallets()
    
    # Create a financial goal
    print("\nCreating financial goal: Robot Body Fund...")
    wallet.create_financial_goal(
        name="Robot Body Fund",
        target_amount=50000.00,
        deadline_days=365,
        category="robot_body"
    )
    
    # Record some income
    print("\nRecording income...")
    wallet.record_income(299.99, "USD", "upwork_job_123", Category.FREELANCE_EARNINGS)
    wallet.record_income(0.001, "BTC", "crypto_payment", Category.CRYPTO_EARNINGS)
    
    # Record some expenses
    print("\nRecording expenses...")
    wallet.record_expense(50.00, "USD", "aws_compute", Category.COMPUTE_RESOURCES)
    wallet.record_expense(20.00, "USD", "openai_api", Category.API_COSTS)
    
    # Create PayPal payment link
    if HAS_PAYPAL:
        print("\nCreating PayPal payment link...")
        payment_link = wallet.create_paypal_payment_link(100.00, "Services from Two Doors Media")
        if payment_link:
            print(f"Payment link: {payment_link}")
    
    # Print summary
    wallet.print_summary()
    
    # Print financial report
    if HAS_FINANCIAL_INTELLIGENCE:
        wallet.get_financial_report(30)


if __name__ == "__main__":
    demo()
