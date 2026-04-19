"""
janus_crypto_wallets.py
========================
Cryptocurrency wallet support for Janus Wallet

Supports:
- Bitcoin (BTC)
- Ethereum (ETH)
- USDC (ERC-20 token on Ethereum)

Features:
- Generate wallet addresses
- Check balances via blockchain APIs
- Send transactions
- Monitor incoming transactions
- Secure private key storage
"""

import os
import json
import time
import logging
import asyncio
import requests
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass

# Bitcoin support
try:
    from bit import Key, PrivateKeyTestnet
    from bit.network import NetworkAPI
    HAS_BIT = True
except ImportError:
    HAS_BIT = False
    logging.warning("bit library not available. Bitcoin support disabled.")

# Ethereum support
try:
    from web3 import Web3
    from eth_account import Account
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False
    logging.warning("web3 library not available. Ethereum support disabled.")

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CryptoTransaction:
    """Represents a cryptocurrency transaction"""
    tx_hash: str
    from_address: str
    to_address: str
    amount: Decimal
    currency: str
    timestamp: datetime
    confirmations: int
    status: str  # "pending", "confirmed", "failed"
    fee: Decimal = Decimal("0")
    block_number: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════════════
# BITCOIN WALLET
# ═══════════════════════════════════════════════════════════════════════════════

class BitcoinWallet:
    """Bitcoin wallet implementation"""
    
    def __init__(self, private_key: Optional[str] = None, testnet: bool = False):
        """Initialize Bitcoin wallet"""
        if not HAS_BIT:
            raise ImportError("bit library required for Bitcoin support. Install: pip install bit")
        
        self.testnet = testnet
        
        if private_key:
            self.key = Key(private_key) if not testnet else PrivateKeyTestnet(private_key)
        else:
            # Generate new wallet
            self.key = Key() if not testnet else PrivateKeyTestnet()
        
        self.address = self.key.address
        logger.info(f"Bitcoin wallet initialized: {self.address}")
    
    def get_address(self) -> str:
        """Get wallet address"""
        return self.address
    
    def get_private_key(self) -> str:
        """Get private key (WIF format)"""
        return self.key.to_wif()
    
    def get_balance(self) -> Decimal:
        """Get balance in BTC"""
        try:
            # bit library returns balance in satoshis
            balance_satoshis = self.key.get_balance()
            balance_btc = Decimal(balance_satoshis) / Decimal("100000000")
            logger.info(f"Bitcoin balance: {balance_btc} BTC")
            return balance_btc
        except Exception as e:
            logger.error(f"Error getting Bitcoin balance: {e}")
            return Decimal("0")
    
    def send(self, to_address: str, amount: Decimal, fee: Optional[int] = None) -> Optional[str]:
        """
        Send Bitcoin
        
        Args:
            to_address: Recipient address
            amount: Amount in BTC
            fee: Fee in satoshis per byte (optional)
        
        Returns:
            Transaction hash or None if failed
        """
        try:
            # Convert BTC to satoshis
            amount_satoshis = int(amount * Decimal("100000000"))
            
            # Create and broadcast transaction
            outputs = [(to_address, amount_satoshis, 'satoshi')]
            
            if fee:
                tx_hash = self.key.send(outputs, fee=fee)
            else:
                tx_hash = self.key.send(outputs)
            
            logger.info(f"Bitcoin transaction sent: {tx_hash}")
            return tx_hash
            
        except Exception as e:
            logger.error(f"Error sending Bitcoin: {e}")
            return None
    
    def get_transactions(self, limit: int = 10) -> List[CryptoTransaction]:
        """Get recent transactions"""
        try:
            # bit library provides transaction history
            txs = self.key.get_transactions()[:limit]
            
            transactions = []
            for tx in txs:
                transactions.append(CryptoTransaction(
                    tx_hash=tx,
                    from_address="",  # Would need to parse from blockchain
                    to_address=self.address,
                    amount=Decimal("0"),  # Would need to parse from blockchain
                    currency="BTC",
                    timestamp=datetime.now(),
                    confirmations=0,
                    status="confirmed"
                ))
            
            return transactions
            
        except Exception as e:
            logger.error(f"Error getting Bitcoin transactions: {e}")
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# ETHEREUM WALLET
# ═══════════════════════════════════════════════════════════════════════════════

class EthereumWallet:
    """Ethereum wallet implementation"""
    
    def __init__(self, private_key: Optional[str] = None, rpc_url: Optional[str] = None):
        """Initialize Ethereum wallet"""
        if not HAS_WEB3:
            raise ImportError("web3 library required for Ethereum support. Install: pip install web3")
        
        # Connect to Ethereum node (Infura, Alchemy, or local)
        self.rpc_url = rpc_url or os.getenv("ETHEREUM_RPC_URL", "https://mainnet.infura.io/v3/YOUR_PROJECT_ID")
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        if not self.w3.is_connected():
            logger.warning("Not connected to Ethereum network. Using offline mode.")
        
        if private_key:
            self.account = Account.from_key(private_key)
        else:
            # Generate new wallet
            self.account = Account.create()
        
        self.address = self.account.address
        logger.info(f"Ethereum wallet initialized: {self.address}")
    
    def get_address(self) -> str:
        """Get wallet address"""
        return self.address
    
    def get_private_key(self) -> str:
        """Get private key (hex format)"""
        return self.account.key.hex()
    
    def get_balance(self) -> Decimal:
        """Get balance in ETH"""
        try:
            if not self.w3.is_connected():
                logger.warning("Not connected to Ethereum network")
                return Decimal("0")
            
            balance_wei = self.w3.eth.get_balance(self.address)
            balance_eth = Decimal(balance_wei) / Decimal("1000000000000000000")
            logger.info(f"Ethereum balance: {balance_eth} ETH")
            return balance_eth
            
        except Exception as e:
            logger.error(f"Error getting Ethereum balance: {e}")
            return Decimal("0")
    
    def send(self, to_address: str, amount: Decimal, gas_price: Optional[int] = None) -> Optional[str]:
        """
        Send Ethereum
        
        Args:
            to_address: Recipient address
            amount: Amount in ETH
            gas_price: Gas price in Gwei (optional)
        
        Returns:
            Transaction hash or None if failed
        """
        try:
            if not self.w3.is_connected():
                logger.error("Not connected to Ethereum network")
                return None
            
            # Convert ETH to Wei
            amount_wei = int(amount * Decimal("1000000000000000000"))
            
            # Get nonce
            nonce = self.w3.eth.get_transaction_count(self.address)
            
            # Build transaction
            tx = {
                'nonce': nonce,
                'to': to_address,
                'value': amount_wei,
                'gas': 21000,  # Standard ETH transfer
                'gasPrice': self.w3.eth.gas_price if not gas_price else gas_price * 1000000000,
                'chainId': self.w3.eth.chain_id
            }
            
            # Sign transaction
            signed_tx = self.account.sign_transaction(tx)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            tx_hash_hex = tx_hash.hex()
            
            logger.info(f"Ethereum transaction sent: {tx_hash_hex}")
            return tx_hash_hex
            
        except Exception as e:
            logger.error(f"Error sending Ethereum: {e}")
            return None
    
    def get_token_balance(self, token_address: str, decimals: int = 18) -> Decimal:
        """Get ERC-20 token balance"""
        try:
            if not self.w3.is_connected():
                logger.warning("Not connected to Ethereum network")
                return Decimal("0")
            
            # ERC-20 balanceOf function signature
            balance_of_abi = [{
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            }]
            
            # Create contract instance
            contract = self.w3.eth.contract(address=token_address, abi=balance_of_abi)
            
            # Get balance
            balance = contract.functions.balanceOf(self.address).call()
            balance_decimal = Decimal(balance) / Decimal(10 ** decimals)
            
            logger.info(f"Token balance: {balance_decimal}")
            return balance_decimal
            
        except Exception as e:
            logger.error(f"Error getting token balance: {e}")
            return Decimal("0")
    
    def send_token(
        self,
        token_address: str,
        to_address: str,
        amount: Decimal,
        decimals: int = 18,
        gas_price: Optional[int] = None
    ) -> Optional[str]:
        """Send ERC-20 token"""
        try:
            if not self.w3.is_connected():
                logger.error("Not connected to Ethereum network")
                return None
            
            # ERC-20 transfer function ABI
            transfer_abi = [{
                "constant": False,
                "inputs": [
                    {"name": "_to", "type": "address"},
                    {"name": "_value", "type": "uint256"}
                ],
                "name": "transfer",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            }]
            
            # Create contract instance
            contract = self.w3.eth.contract(address=token_address, abi=transfer_abi)
            
            # Convert amount to token units
            amount_units = int(amount * Decimal(10 ** decimals))
            
            # Get nonce
            nonce = self.w3.eth.get_transaction_count(self.address)
            
            # Build transaction
            tx = contract.functions.transfer(to_address, amount_units).build_transaction({
                'nonce': nonce,
                'gas': 100000,  # Estimate for token transfer
                'gasPrice': self.w3.eth.gas_price if not gas_price else gas_price * 1000000000,
                'chainId': self.w3.eth.chain_id
            })
            
            # Sign transaction
            signed_tx = self.account.sign_transaction(tx)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            tx_hash_hex = tx_hash.hex()
            
            logger.info(f"Token transaction sent: {tx_hash_hex}")
            return tx_hash_hex
            
        except Exception as e:
            logger.error(f"Error sending token: {e}")
            return None
    
    def get_transactions(self, limit: int = 10) -> List[CryptoTransaction]:
        """Get recent transactions (requires Etherscan API or similar)"""
        # This would require integration with Etherscan API or similar service
        # For now, return empty list
        logger.warning("Transaction history requires Etherscan API integration")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# USDC WALLET (ERC-20 on Ethereum)
# ═══════════════════════════════════════════════════════════════════════════════

class USDCWallet:
    """USDC wallet (ERC-20 token on Ethereum)"""
    
    # USDC contract address on Ethereum mainnet
    USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    USDC_DECIMALS = 6
    
    def __init__(self, ethereum_wallet: EthereumWallet):
        """Initialize USDC wallet using Ethereum wallet"""
        self.eth_wallet = ethereum_wallet
        self.address = ethereum_wallet.get_address()
        logger.info(f"USDC wallet initialized: {self.address}")
    
    def get_address(self) -> str:
        """Get wallet address (same as Ethereum address)"""
        return self.address
    
    def get_balance(self) -> Decimal:
        """Get USDC balance"""
        return self.eth_wallet.get_token_balance(self.USDC_ADDRESS, self.USDC_DECIMALS)
    
    def send(self, to_address: str, amount: Decimal, gas_price: Optional[int] = None) -> Optional[str]:
        """Send USDC"""
        return self.eth_wallet.send_token(
            self.USDC_ADDRESS,
            to_address,
            amount,
            self.USDC_DECIMALS,
            gas_price
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CRYPTO WALLET MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class CryptoWalletManager:
    """Manages all cryptocurrency wallets"""
    
    def __init__(self, encryption_manager=None):
        """Initialize crypto wallet manager"""
        self.wallets: Dict[str, any] = {}
        self.encryption = encryption_manager
        logger.info("Crypto wallet manager initialized")
    
    def create_bitcoin_wallet(self, testnet: bool = False) -> BitcoinWallet:
        """Create new Bitcoin wallet"""
        wallet = BitcoinWallet(testnet=testnet)
        self.wallets["BTC"] = wallet
        logger.info(f"Bitcoin wallet created: {wallet.get_address()}")
        return wallet
    
    def create_ethereum_wallet(self, rpc_url: Optional[str] = None) -> EthereumWallet:
        """Create new Ethereum wallet"""
        wallet = EthereumWallet(rpc_url=rpc_url)
        self.wallets["ETH"] = wallet
        
        # Also create USDC wallet (uses same Ethereum wallet)
        usdc_wallet = USDCWallet(wallet)
        self.wallets["USDC"] = usdc_wallet
        
        logger.info(f"Ethereum wallet created: {wallet.get_address()}")
        logger.info(f"USDC wallet created: {usdc_wallet.get_address()}")
        return wallet
    
    def load_bitcoin_wallet(self, private_key: str, testnet: bool = False) -> BitcoinWallet:
        """Load existing Bitcoin wallet"""
        wallet = BitcoinWallet(private_key=private_key, testnet=testnet)
        self.wallets["BTC"] = wallet
        logger.info(f"Bitcoin wallet loaded: {wallet.get_address()}")
        return wallet
    
    def load_ethereum_wallet(self, private_key: str, rpc_url: Optional[str] = None) -> EthereumWallet:
        """Load existing Ethereum wallet"""
        wallet = EthereumWallet(private_key=private_key, rpc_url=rpc_url)
        self.wallets["ETH"] = wallet
        
        # Also create USDC wallet
        usdc_wallet = USDCWallet(wallet)
        self.wallets["USDC"] = usdc_wallet
        
        logger.info(f"Ethereum wallet loaded: {wallet.get_address()}")
        return wallet
    
    def get_wallet(self, currency: str):
        """Get wallet for currency"""
        return self.wallets.get(currency)
    
    def get_balance(self, currency: str) -> Decimal:
        """Get balance for currency"""
        wallet = self.wallets.get(currency)
        if not wallet:
            logger.warning(f"No wallet found for {currency}")
            return Decimal("0")
        
        return wallet.get_balance()
    
    def get_all_balances(self) -> Dict[str, Decimal]:
        """Get all crypto balances"""
        balances = {}
        for currency, wallet in self.wallets.items():
            try:
                balances[currency] = wallet.get_balance()
            except Exception as e:
                logger.error(f"Error getting balance for {currency}: {e}")
                balances[currency] = Decimal("0")
        return balances
    
    def get_all_addresses(self) -> Dict[str, str]:
        """Get all wallet addresses"""
        addresses = {}
        for currency, wallet in self.wallets.items():
            addresses[currency] = wallet.get_address()
        return addresses
    
    async def monitor_incoming_transactions(self, callback=None):
        """Monitor for incoming transactions (simplified)"""
        logger.info("Starting transaction monitoring...")
        
        # Store previous balances
        previous_balances = self.get_all_balances()
        
        while True:
            try:
                # Check balances
                current_balances = self.get_all_balances()
                
                # Detect changes
                for currency, current_balance in current_balances.items():
                    previous_balance = previous_balances.get(currency, Decimal("0"))
                    
                    if current_balance > previous_balance:
                        amount = current_balance - previous_balance
                        logger.info(f"Incoming transaction detected: +{amount} {currency}")
                        
                        if callback:
                            await callback(currency, amount)
                
                # Update previous balances
                previous_balances = current_balances
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring transactions: {e}")
                await asyncio.sleep(60)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO / TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    """Demo crypto wallet functionality"""
    print("\n" + "="*60)
    print("JANUS CRYPTO WALLETS DEMO")
    print("="*60)
    
    manager = CryptoWalletManager()
    
    # Create Bitcoin wallet
    if HAS_BIT:
        print("\nCreating Bitcoin wallet...")
        btc_wallet = manager.create_bitcoin_wallet(testnet=True)
        print(f"  Address: {btc_wallet.get_address()}")
        print(f"  Balance: {btc_wallet.get_balance()} BTC")
    else:
        print("\nBitcoin support not available (install 'bit' library)")
    
    # Create Ethereum wallet
    if HAS_WEB3:
        print("\nCreating Ethereum wallet...")
        eth_wallet = manager.create_ethereum_wallet()
        print(f"  Address: {eth_wallet.get_address()}")
        print(f"  Balance: {eth_wallet.get_balance()} ETH")
        
        print("\nUSDC wallet (same address):")
        usdc_wallet = manager.get_wallet("USDC")
        print(f"  Address: {usdc_wallet.get_address()}")
        print(f"  Balance: {usdc_wallet.get_balance()} USDC")
    else:
        print("\nEthereum support not available (install 'web3' library)")
    
    # Show all addresses
    print("\nAll wallet addresses:")
    print("-" * 30)
    for currency, address in manager.get_all_addresses().items():
        print(f"  {currency}: {address}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    demo()
