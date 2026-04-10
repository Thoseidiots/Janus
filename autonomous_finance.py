
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
import qrcode # Assuming qrcode library is available or will be installed

# --- Configuration ---
LEDGER_FILE = "autonomous_finance_ledger.json"
REVOLUT_QR_PATH = "/home/ubuntu/upload/IMG_6058.PNG" # User-provided Revolut QR code

# --- Enums ---
class TransactionType(str, Enum):
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"

class TransactionStatus(str, Enum):
    PENDING = "pending"
    AUTHORIZED = "authorized"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class PaymentMethod(str, Enum):
    REVOLUT = "revolut"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"
    CRYPTO = "crypto"
    OTHER = "other"

# --- Data Models ---
@dataclass
class FinancialTransaction:
    id: str
    type: TransactionType
    amount: float
    currency: str
    description: str
    method: PaymentMethod
    status: TransactionStatus
    created_at: float
    completed_at: Optional[float] = None
    notes: str = ""
    qr_code_data: Optional[str] = None # Data to be encoded in QR for authorization/payment

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        d["method"] = self.method.value
        d["status"] = self.status.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "FinancialTransaction":
        d["type"] = TransactionType(d["type"])
        d["method"] = PaymentMethod(d["method"])
        d["status"] = TransactionStatus(d["status"])
        return FinancialTransaction(**d)

@dataclass
class HoldingAccountSummary:
    total_held: float = 0.0
    currency: str = "USD"
    last_updated: float = time.time()

# --- Core Financial Management System ---
class AutonomousFinance:
    def __init__(self, ledger_path: str = LEDGER_FILE):
        self.ledger_path = Path(ledger_path)
        self.transactions: Dict[str, FinancialTransaction] = {}
        self.holding_account: HoldingAccountSummary = HoldingAccountSummary()
        self._load_ledger()

    def _save_ledger(self) -> None:
        try:
            data = {
                "transactions": [t.to_dict() for t in self.transactions.values()],
                "holding_account": asdict(self.holding_account)
            }
            with open(self.ledger_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[AutonomousFinance] Ledger save failed: {e}")

    def _load_ledger(self) -> None:
        if not self.ledger_path.exists():
            return
        try:
            with open(self.ledger_path, "r") as f:
                data = json.load(f)
            for d in data.get("transactions", []):
                trans = FinancialTransaction.from_dict(d)
                self.transactions[trans.id] = trans
            if "holding_account" in data:
                self.holding_account = HoldingAccountSummary(**data["holding_account"])
            print(f"[AutonomousFinance] Loaded {len(self.transactions)} transactions.")
        except Exception as e:
            print(f"[AutonomousFinance] Ledger load failed: {e}")

    def create_transaction(
        self,
        type: TransactionType,
        amount: float,
        currency: str,
        description: str,
        method: PaymentMethod = PaymentMethod.OTHER,
        notes: str = "",
    ) -> FinancialTransaction:
        transaction_id = str(uuid.uuid4())
        status = TransactionStatus.PENDING
        qr_data = None

        # Logic for generating QR code data based on transaction type/method
        if type == TransactionType.INCOME and method == PaymentMethod.REVOLUT:
            # For Revolut income, use the provided static QR code
            qr_data = f"file://{REVOLUT_QR_PATH}"
        elif type == TransactionType.INCOME and method == PaymentMethod.PAYPAL:
            qr_data = f"janus://income?id={transaction_id}&amount={amount}&currency={currency}&method={method.value}"
        elif type == TransactionType.TRANSFER:
            # For transfers, QR could be for user authorization
            qr_data = f"janus://authorize_transfer?id={transaction_id}&amount={amount}&currency={currency}"

        trans = FinancialTransaction(
            id=transaction_id,
            type=type,
            amount=amount,
            currency=currency,
            description=description,
            method=method,
            status=status,
            created_at=time.time(),
            notes=notes,
            qr_code_data=qr_data
        )
        self.transactions[transaction_id] = trans
        self._save_ledger()
        print(f"[AutonomousFinance] Created {type.value} transaction {transaction_id[:8]} for {currency} {amount:.2f}")
        return trans

    def update_transaction_status(self, transaction_id: str, new_status: TransactionStatus, notes: str = "") -> bool:
        trans = self.transactions.get(transaction_id)
        if not trans:
            print(f"[AutonomousFinance] Transaction {transaction_id[:8]} not found.")
            return False

        trans.status = new_status
        if new_status == TransactionStatus.COMPLETED:
            trans.completed_at = time.time()
            # Update holding account balance
            if trans.type == TransactionType.INCOME:
                self.holding_account.total_held += trans.amount
            elif trans.type == TransactionType.EXPENSE or trans.type == TransactionType.TRANSFER:
                self.holding_account.total_held -= trans.amount
            self.holding_account.last_updated = time.time()

        if notes:
            trans.notes = notes
        self._save_ledger()
        print(f"[AutonomousFinance] Transaction {transaction_id[:8]} status updated to {new_status.value}")
        return True

    def get_transaction(self, transaction_id: str) -> Optional[FinancialTransaction]:
        return self.transactions.get(transaction_id)

    def get_all_transactions(self) -> List[FinancialTransaction]:
        return list(self.transactions.values())

    def get_holding_account_summary(self) -> HoldingAccountSummary:
        return self.holding_account

    def generate_qr_code_image(self, transaction_id: str, filename: str) -> Optional[str]:
        trans = self.transactions.get(transaction_id)
        if not trans or not trans.qr_code_data:
            print(f"[AutonomousFinance] No QR code data for transaction {transaction_id[:8]}")
            return None
        
        if trans.qr_code_data.startswith("file://"):
            return trans.qr_code_data.replace("file://", "")

        try:
            img = qrcode.make(trans.qr_code_data)
            file_path = f"/home/ubuntu/Janus/{filename}.png"
            img.save(file_path)
            print(f"[AutonomousFinance] QR code saved to {file_path}")
            return file_path
        except Exception as e:
            print(f"[AutonomousFinance] Failed to generate QR code: {e}")
            return None

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    finance = AutonomousFinance(ledger_path="test_autonomous_finance_ledger.json")

    # Simulate income via Revolut
    income_req = finance.create_transaction(
        type=TransactionType.INCOME,
        amount=100.00,
        currency="USD",
        description="Commission for project X",
        method=PaymentMethod.REVOLUT
    )
    if income_req.qr_code_data:
        finance.generate_qr_code_image(income_req.id, f"income_qr_{income_req.id[:8]}")
    finance.update_transaction_status(income_req.id, TransactionStatus.COMPLETED)

    # Simulate a transfer to user's main account (requires authorization)
    transfer_req = finance.create_transaction(
        type=TransactionType.TRANSFER,
        amount=50.00,
        currency="USD",
        description="Transfer to user's main account",
        method=PaymentMethod.BANK_TRANSFER
    )
    if transfer_req.qr_code_data:
        finance.generate_qr_code_image(transfer_req.id, f"transfer_qr_{transfer_req.id[:8]}")
    # User would scan QR and authorize, then Janus would call update_transaction_status
    finance.update_transaction_status(transfer_req.id, TransactionStatus.COMPLETED, notes="User authorized via QR")

    # Simulate an expense
    expense_req = finance.create_transaction(
        type=TransactionType.EXPENSE,
        amount=20.00,
        currency="USD",
        description="Software subscription",
        method=PaymentMethod.PAYPAL
    )
    finance.update_transaction_status(expense_req.id, TransactionStatus.COMPLETED)

    print("\n--- Current Holding Account Summary ---")
    summary = finance.get_holding_account_summary()
    print(f"Total Held: {summary.currency} {summary.total_held:.2f}")

    print("\n--- All Transactions ---")
    for trans in finance.get_all_transactions():
        print(f"ID: {trans.id[:8]}, Type: {trans.type.value}, Amount: {trans.currency} {trans.amount:.2f}, Status: {trans.status.value}")

    # Clean up test file
    import os
    if os.path.exists("test_autonomous_finance_ledger.json"):
        os.remove("test_autonomous_finance_ledger.json")
    if os.path.exists(f"income_qr_{income_req.id[:8]}.png"):
        os.remove(f"income_qr_{income_req.id[:8]}.png")
    if os.path.exists(f"transfer_qr_{transfer_req.id[:8]}.png"):
        os.remove(f"transfer_qr_{transfer_req.id[:8]}.png")
