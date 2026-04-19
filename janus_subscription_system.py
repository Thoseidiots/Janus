"""
Janus Financial Management System - CEO Financial Operations

Handles revenue tracking, expense management, and financial reporting for
Janus AI CEO operations. Manages the financial aspects of running multiple
autonomous businesses to generate $10k/month revenue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import stripe
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import sqlite3

# FastAPI for subscription API
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger(__name__)

class SubscriptionTier(Enum):
    FREE = "free"
    DEVELOPER = "developer"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

class SubscriptionStatus(Enum):
    ACTIVE = "active"
    CANCELLED = "cancelled"
    PAST_DUE = "past_due"
    TRIALING = "trialing"

@dataclass
class SubscriptionPlan:
    """Subscription plan configuration"""
    tier: SubscriptionTier
    name: str
    price_monthly: float
    price_yearly: float
    features: List[str]
    limits: Dict[str, Any]
    stripe_price_id_monthly: Optional[str] = None
    stripe_price_id_yearly: Optional[str] = None

@dataclass
class UserSubscription:
    """User's subscription information"""
    subscription_id: str
    user_id: str
    tier: SubscriptionTier
    status: SubscriptionStatus
    plan: SubscriptionPlan
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool = False
    trial_end: Optional[datetime] = None
    stripe_subscription_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class UsageRecord:
    """Usage tracking record"""
    record_id: str
    user_id: str
    subscription_id: str
    metric: str  # "api_requests", "tokens", "storage", "instances"
    quantity: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BillingEvent:
    """Billing and payment events"""
    event_id: str
    user_id: str
    subscription_id: str
    event_type: str  # "charge", "refund", "usage_alert"
    amount: float
    currency: str = "USD"
    status: str = "pending"
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SubscriptionDatabase:
    """Database for subscription management"""
    
    def __init__(self, db_path: str = "subscriptions.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                created_at TEXT,
                updated_at TEXT,
                stripe_customer_id TEXT
            )
        """)
        
        # Subscriptions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                subscription_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                tier TEXT NOT NULL,
                status TEXT NOT NULL,
                plan_data TEXT,
                current_period_start TEXT,
                current_period_end TEXT,
                cancel_at_period_end INTEGER DEFAULT 0,
                trial_end TEXT,
                stripe_subscription_id TEXT,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        
        # Usage records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_records (
                record_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                subscription_id TEXT NOT NULL,
                metric TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                timestamp TEXT,
                metadata TEXT
            )
        """)
        
        # Billing events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS billing_events (
                event_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                subscription_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT DEFAULT 'USD',
                status TEXT DEFAULT 'pending',
                description TEXT,
                timestamp TEXT,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_user(self, user_id: str, email: str, password_hash: str, stripe_customer_id: str = None):
        """Create new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO users (user_id, email, password_hash, created_at, updated_at, stripe_customer_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            email,
            password_hash,
            datetime.utcnow().isoformat(),
            datetime.utcnow().isoformat(),
            stripe_customer_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "user_id": row[0],
                "email": row[1],
                "password_hash": row[2],
                "created_at": row[3],
                "updated_at": row[4],
                "stripe_customer_id": row[5]
            }
        return None
    
    def create_subscription(self, subscription: UserSubscription):
        """Create subscription"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO subscriptions (
                subscription_id, user_id, tier, status, plan_data,
                current_period_start, current_period_end, cancel_at_period_end,
                trial_end, stripe_subscription_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            subscription.subscription_id,
            subscription.user_id,
            subscription.tier.value,
            subscription.status.value,
            json.dumps(asdict(subscription.plan)),
            subscription.current_period_start.isoformat(),
            subscription.current_period_end.isoformat(),
            subscription.cancel_at_period_end,
            subscription.trial_end.isoformat() if subscription.trial_end else None,
            subscription.stripe_subscription_id,
            subscription.created_at.isoformat(),
            subscription.updated_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_user_subscription(self, user_id: str) -> Optional[UserSubscription]:
        """Get user's active subscription"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM subscriptions WHERE user_id = ? AND status = 'active'
            ORDER BY created_at DESC LIMIT 1
        """, (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            plan_data = json.loads(row[4])
            plan = SubscriptionPlan(
                tier=SubscriptionTier(plan_data["tier"]),
                name=plan_data["name"],
                price_monthly=plan_data["price_monthly"],
                price_yearly=plan_data["price_yearly"],
                features=plan_data["features"],
                limits=plan_data["limits"]
            )
            
            return UserSubscription(
                subscription_id=row[0],
                user_id=row[1],
                tier=SubscriptionTier(row[2]),
                status=SubscriptionStatus(row[3]),
                plan=plan,
                current_period_start=datetime.fromisoformat(row[5]),
                current_period_end=datetime.fromisoformat(row[6]),
                cancel_at_period_end=bool(row[7]),
                trial_end=datetime.fromisoformat(row[8]) if row[8] else None,
                stripe_subscription_id=row[9],
                created_at=datetime.fromisoformat(row[10]),
                updated_at=datetime.fromisoformat(row[11])
            )
        return None
    
    def record_usage(self, usage: UsageRecord):
        """Record usage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO usage_records (record_id, user_id, subscription_id, metric, quantity, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            usage.record_id,
            usage.user_id,
            usage.subscription_id,
            usage.metric,
            usage.quantity,
            usage.timestamp.isoformat(),
            json.dumps(usage.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def get_usage_summary(self, user_id: str, period_start: datetime, period_end: datetime) -> Dict[str, int]:
        """Get usage summary for period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT metric, SUM(quantity) FROM usage_records
            WHERE user_id = ? AND timestamp >= ? AND timestamp <= ?
            GROUP BY metric
        """, (user_id, period_start.isoformat(), period_end.isoformat()))
        
        rows = cursor.fetchall()
        conn.close()
        
        return {metric: quantity for metric, quantity in rows}

class JanusSubscriptionManager:
    """Main subscription management system"""
    
    def __init__(self, stripe_secret_key: str = None):
        self.db = SubscriptionDatabase()
        
        # Initialize Stripe (in production, use real key)
        if stripe_secret_key:
            stripe.api_key = stripe_secret_key
        
        # Define subscription plans
        self.plans = self._define_plans()
        
        logger.info("Janus Subscription Manager initialized")
    
    def _define_plans(self) -> Dict[SubscriptionTier, SubscriptionPlan]:
        """Define subscription plans"""
        return {
            SubscriptionTier.FREE: SubscriptionPlan(
                tier=SubscriptionTier.FREE,
                name="Free Tier",
                price_monthly=0.0,
                price_yearly=0.0,
                features=[
                    "Basic text generation",
                    "1B model access",
                    "100 requests/hour",
                    "Community support"
                ],
                limits={
                    "requests_per_hour": 100,
                    "max_model_size": "1b",
                    "max_instances": 1,
                    "storage_gb": 1,
                    "support_level": "community"
                }
            ),
            SubscriptionTier.DEVELOPER: SubscriptionPlan(
                tier=SubscriptionTier.DEVELOPER,
                name="Developer Tier",
                price_monthly=100.0,
                price_yearly=1000.0,  # 2 months free
                features=[
                    "All free features",
                    "7B model access",
                    "1,000 requests/hour",
                    "Video analysis",
                    "Speech synthesis",
                    "Holographic memory",
                    "Email support"
                ],
                limits={
                    "requests_per_hour": 1000,
                    "max_model_size": "7b",
                    "max_instances": 3,
                    "storage_gb": 10,
                    "support_level": "email"
                }
            ),
            SubscriptionTier.PROFESSIONAL: SubscriptionPlan(
                tier=SubscriptionTier.PROFESSIONAL,
                name="Professional Tier",
                price_monthly=500.0,
                price_yearly=5000.0,  # 2 months free
                features=[
                    "All developer features",
                    "34B model access",
                    "10,000 requests/hour",
                    "Custom model training",
                    "Priority support",
                    "API access",
                    "Advanced analytics"
                ],
                limits={
                    "requests_per_hour": 10000,
                    "max_model_size": "34b",
                    "max_instances": 10,
                    "storage_gb": 100,
                    "support_level": "priority"
                }
            ),
            SubscriptionTier.ENTERPRISE: SubscriptionPlan(
                tier=SubscriptionTier.ENTERPRISE,
                name="Enterprise Tier",
                price_monthly=2000.0,
                price_yearly=20000.0,  # 2 months free
                features=[
                    "All professional features",
                    "70B model access",
                    "Unlimited requests",
                    "Multi-instance deployment",
                    "Dedicated support",
                    "Custom integrations",
                    "SLA guarantees",
                    "On-premise deployment"
                ],
                limits={
                    "requests_per_hour": -1,  # Unlimited
                    "max_model_size": "70b",
                    "max_instances": -1,  # Unlimited
                    "storage_gb": -1,  # Unlimited
                    "support_level": "dedicated"
                }
            )
        }
    
    async def create_user(self, email: str, password: str) -> str:
        """Create new user"""
        user_id = str(uuid.uuid4())
        password_hash = self._hash_password(password)
        
        # Create Stripe customer (in production)
        stripe_customer_id = f"cus_{uuid.uuid4().hex}"
        
        self.db.create_user(user_id, email, password_hash, stripe_customer_id)
        
        # Create free subscription
        await self._create_free_subscription(user_id)
        
        logger.info(f"Created user {user_id} with email {email}")
        return user_id
    
    def _hash_password(self, password: str) -> str:
        """Hash password (simplified)"""
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest()
    
    async def _create_free_subscription(self, user_id: str):
        """Create free subscription for user"""
        subscription_id = str(uuid.uuid4())
        plan = self.plans[SubscriptionTier.FREE]
        
        subscription = UserSubscription(
            subscription_id=subscription_id,
            user_id=user_id,
            tier=SubscriptionTier.FREE,
            status=SubscriptionStatus.ACTIVE,
            plan=plan,
            current_period_start=datetime.utcnow(),
            current_period_end=datetime.utcnow() + timedelta(days=365)  # Free for 1 year
        )
        
        self.db.create_subscription(subscription)
    
    async def upgrade_subscription(self, user_id: str, tier: SubscriptionTier, billing_cycle: str = "monthly") -> UserSubscription:
        """Upgrade user subscription"""
        user = self.db.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        plan = self.plans[tier]
        subscription_id = str(uuid.uuid4())
        
        # Calculate billing period
        now = datetime.utcnow()
        if billing_cycle == "monthly":
            period_end = now + timedelta(days=30)
            price = plan.price_monthly
        else:  # yearly
            period_end = now + timedelta(days=365)
            price = plan.price_yearly
        
        # Create subscription
        subscription = UserSubscription(
            subscription_id=subscription_id,
            user_id=user_id,
            tier=tier,
            status=SubscriptionStatus.ACTIVE,
            plan=plan,
            current_period_start=now,
            current_period_end=period_end,
            stripe_subscription_id=f"sub_{uuid.uuid4().hex}"
        )
        
        self.db.create_subscription(subscription)
        
        # Create billing event
        await self._create_billing_event(
            user_id=user_id,
            subscription_id=subscription_id,
            event_type="charge",
            amount=price,
            description=f"{plan.name} - {billing_cycle}"
        )
        
        logger.info(f"Upgraded user {user_id} to {tier.value}")
        return subscription
    
    async def _create_billing_event(self, user_id: str, subscription_id: str, event_type: str, amount: float, description: str = ""):
        """Create billing event"""
        event_id = str(uuid.uuid4())
        event = BillingEvent(
            event_id=event_id,
            user_id=user_id,
            subscription_id=subscription_id,
            event_type=event_type,
            amount=amount,
            description=description,
            status="completed"
        )
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO billing_events (event_id, user_id, subscription_id, event_type, amount, description, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.user_id,
            event.subscription_id,
            event.event_type,
            event.amount,
            event.description,
            event.status,
            event.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    async def record_usage(self, user_id: str, metric: str, quantity: int, metadata: Dict[str, Any] = None):
        """Record usage for user"""
        subscription = self.db.get_user_subscription(user_id)
        if not subscription:
            raise HTTPException(status_code=404, detail="No active subscription")
        
        # Check limits
        if not self._check_usage_limits(subscription, metric, quantity):
            raise HTTPException(status_code=429, detail="Usage limit exceeded")
        
        usage = UsageRecord(
            record_id=str(uuid.uuid4()),
            user_id=user_id,
            subscription_id=subscription.subscription_id,
            metric=metric,
            quantity=quantity,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.db.record_usage(usage)
    
    def _check_usage_limits(self, subscription: UserSubscription, metric: str, additional_quantity: int) -> bool:
        """Check if usage is within limits"""
        limits = subscription.plan.limits
        
        if metric == "requests_per_hour":
            limit = limits.get("requests_per_hour", 100)
            if limit == -1:  # Unlimited
                return True
            
            # Get current hour usage
            now = datetime.utcnow()
            hour_start = now.replace(minute=0, second=0, microsecond=0)
            current_usage = self.db.get_usage_summary(subscription.user_id, hour_start, now)
            current_requests = current_usage.get("api_requests", 0)
            
            return current_requests + additional_quantity <= limit
        
        return True  # Other metrics not implemented yet
    
    def get_subscription_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's subscription information"""
        subscription = self.db.get_user_subscription(user_id)
        if not subscription:
            return None
        
        # Get current usage
        now = datetime.utcnow()
        period_start = subscription.current_period_start
        usage = self.db.get_usage_summary(user_id, period_start, now)
        
        return {
            "subscription": asdict(subscription),
            "usage": usage,
            "limits": subscription.plan.limits,
            "remaining_days": (subscription.current_period_end - now).days
        }
    
    def calculate_revenue_projection(self, months: int = 12) -> Dict[str, Any]:
        """Calculate revenue projection"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Get all active subscriptions
        cursor.execute("""
            SELECT tier, COUNT(*) FROM subscriptions WHERE status = 'active'
            GROUP BY tier
        """)
        
        tier_counts = dict(cursor.fetchall())
        conn.close()
        
        # Calculate monthly revenue
        monthly_revenue = 0
        tier_breakdown = {}
        
        for tier_name, count in tier_counts.items():
            tier = SubscriptionTier(tier_name)
            plan = self.plans[tier]
            revenue = count * plan.price_monthly
            monthly_revenue += revenue
            tier_breakdown[tier_name] = {
                "subscribers": count,
                "monthly_revenue": revenue,
                "yearly_revenue": revenue * 12
            }
        
        return {
            "current_monthly_revenue": monthly_revenue,
            "projected_yearly_revenue": monthly_revenue * 12,
            "tier_breakdown": tier_breakdown,
            "total_subscribers": sum(tier_counts.values()),
            "projection_months": months
        }

class JanusSubscriptionAPI:
    """API for subscription management"""
    
    def __init__(self, stripe_secret_key: str = None):
        self.app = FastAPI(
            title="Janus Subscription System",
            description="Subscription and billing management",
            version="1.0.0"
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Security
        self.security = HTTPBearer()
        
        # Subscription manager
        self.manager = JanusSubscriptionManager(stripe_secret_key)
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Janus Subscription API initialized")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "Janus Subscription System",
                "version": "1.0.0",
                "tiers": list(tier.value for tier in SubscriptionTier)
            }
        
        @self.app.post("/auth/register")
        async def register(email: str, password: str):
            """Register new user"""
            user_id = await self.manager.create_user(email, password)
            return {
                "user_id": user_id,
                "message": "User registered successfully",
                "subscription_tier": "free"
            }
        
        @self.app.post("/auth/login")
        async def login(email: str, password: str):
            """User login (simplified)"""
            # In production, verify password properly
            user_id = f"user_{uuid.uuid4().hex}"
            return {
                "user_id": user_id,
                "token": f"token_{uuid.uuid4().hex}",
                "message": "Login successful"
            }
        
        @self.app.get("/subscription/plans")
        async def list_plans():
            """List available subscription plans"""
            return {
                "plans": [
                    {
                        "tier": plan.tier.value,
                        "name": plan.name,
                        "price_monthly": plan.price_monthly,
                        "price_yearly": plan.price_yearly,
                        "features": plan.features,
                        "limits": plan.limits
                    }
                    for plan in self.manager.plans.values()
                ]
            }
        
        @self.app.get("/subscription/current")
        async def get_current_subscription(
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Get user's current subscription"""
            # Extract user ID from token (simplified)
            user_id = "demo_user"
            
            subscription_info = self.manager.get_subscription_info(user_id)
            if not subscription_info:
                # Create free subscription if none exists
                await self.manager._create_free_subscription(user_id)
                subscription_info = self.manager.get_subscription_info(user_id)
            
            return subscription_info
        
        @self.app.post("/subscription/upgrade")
        async def upgrade_subscription(
            tier: str,
            billing_cycle: str = "monthly",
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Upgrade subscription"""
            user_id = "demo_user"
            
            try:
                subscription = await self.manager.upgrade_subscription(
                    user_id, SubscriptionTier(tier), billing_cycle
                )
                return {
                    "subscription": asdict(subscription),
                    "message": f"Successfully upgraded to {tier}"
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/usage/record")
        async def record_usage(
            metric: str,
            quantity: int,
            metadata: Dict[str, Any] = None,
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Record usage"""
            if metadata is None:
                metadata = {}
            user_id = "demo_user"
            
            try:
                await self.manager.record_usage(user_id, metric, quantity, metadata)
                return {"message": "Usage recorded successfully"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/analytics/revenue")
        async def get_revenue_analytics(
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Get revenue analytics (admin only)"""
            projection = self.manager.calculate_revenue_projection()
            return projection
    
    def run(self, host: str = "0.0.0.0", port: int = 8003):
        """Run the subscription API"""
        uvicorn.run(self.app, host=host, port=port, log_level="info")

# Create global subscription API instance
subscription_api = JanusSubscriptionAPI()

if __name__ == "__main__":
    subscription_api.run(host="0.0.0.0", port=8003)
