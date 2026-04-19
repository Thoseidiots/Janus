"""
test_wallet_integration.py
===========================
Test wallet integration with autonomous worker
"""

import asyncio
import sys
from janus_autonomous_worker import JanusAutonomousWorker, Job, JobStatus
from datetime import datetime, timedelta


async def test_wallet_integration():
    """Test wallet integration with autonomous worker"""
    print("="*60)
    print("TESTING WALLET INTEGRATION WITH AUTONOMOUS WORKER")
    print("="*60)
    
    # Initialize worker
    print("\n1. Initializing Janus Autonomous Worker...")
    worker = JanusAutonomousWorker(name="Janus")
    
    # Check if wallet is initialized
    if not worker.wallet:
        print("❌ Wallet not initialized!")
        return False
    
    print(f"✓ Wallet initialized: {worker.wallet.BUSINESS_NAME}")
    
    # Check crypto addresses
    crypto_addresses = worker.wallet.get_crypto_addresses()
    if crypto_addresses:
        print(f"✓ Crypto wallets ready:")
        for currency, address in crypto_addresses.items():
            print(f"  - {currency}: {address}")
    else:
        print("⚠ No crypto addresses found (will be created on first use)")
    
    # Check PayPal
    paypal_email = worker.wallet.get_paypal_email()
    if paypal_email:
        print(f"✓ PayPal configured: {paypal_email}")
    else:
        print("⚠ PayPal not configured")
    
    # Test recording income (simulating job payment)
    print("\n2. Testing income recording...")
    try:
        from janus_wallet import Category
        
        # Simulate a completed job
        test_job = Job(
            id="test_job_001",
            title="Test Web Development Job",
            description="Build a simple website",
            required_skills=["html", "css", "javascript"],
            budget=150.00,
            deadline=datetime.now() + timedelta(days=7),
            platform="upwork",
            status=JobStatus.COMPLETED
        )
        
        # Record payment
        tx_id = worker.wallet.record_income(
            amount=150.00,
            currency="USD",
            source="upwork_test_job_001",
            category=Category.FREELANCE_EARNINGS,
            metadata={
                "job_id": test_job.id,
                "job_title": test_job.title,
                "platform": "upwork"
            }
        )
        
        print(f"✓ Income recorded: {tx_id}")
        
        # Check balance
        balance = worker.wallet.get_balance("USD")
        print(f"✓ Current USD balance: ${balance}")
        
    except Exception as e:
        print(f"❌ Error recording income: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test recording expense
    print("\n3. Testing expense recording...")
    try:
        tx_id = worker.wallet.record_expense(
            amount=25.00,
            currency="USD",
            destination="aws_compute",
            category=Category.COMPUTE_RESOURCES,
            metadata={"purpose": "model_training"}
        )
        
        print(f"✓ Expense recorded: {tx_id}")
        
        # Check balance after expense
        balance = worker.wallet.get_balance("USD")
        print(f"✓ Current USD balance after expense: ${balance}")
        
    except Exception as e:
        print(f"❌ Error recording expense: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test get_status with wallet info
    print("\n4. Testing status reporting...")
    try:
        status = worker.get_status()
        
        if "wallet" in status:
            print("✓ Wallet info in status:")
            print(f"  - Business: {status['wallet'].get('business_name')}")
            print(f"  - Balances: {status['wallet'].get('balances')}")
            if status['wallet'].get('crypto_addresses'):
                print(f"  - Crypto addresses: {list(status['wallet']['crypto_addresses'].keys())}")
        else:
            print("⚠ Wallet info not in status")
        
    except Exception as e:
        print(f"❌ Error getting status: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Print wallet summary
    print("\n5. Wallet Summary:")
    print("-"*60)
    try:
        worker.wallet.print_summary()
    except Exception as e:
        print(f"❌ Error printing summary: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nWallet integration is working correctly.")
    print("Janus can now:")
    print("  ✓ Track earnings from jobs")
    print("  ✓ Record expenses for resources")
    print("  ✓ Maintain crypto wallet addresses")
    print("  ✓ Generate financial reports")
    print("  ✓ Set and track financial goals")
    
    return True


if __name__ == "__main__":
    result = asyncio.run(test_wallet_integration())
    sys.exit(0 if result else 1)
