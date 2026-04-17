"""
Janus CEO Dashboard - Business Management Interface

Provides management dashboard for Janus AI CEO to monitor and control
multiple autonomous businesses. Tracks revenue, performance metrics,
and operational efficiency across the CEO's business portfolio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import sqlite3

# FastAPI for dashboard API
from fastapi import FastAPI, HTTPException, Depends, Security, status, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger(__name__)

# Data Models
@dataclass
class EnterpriseInstance:
    """Represents a Janus instance in enterprise deployment"""
    instance_id: str
    name: str
    status: str  # "running", "stopped", "error", "maintenance"
    model_size: str
    capabilities: List[str]
    created_at: datetime
    last_active: datetime
    resource_usage: Dict[str, float]  # cpu, memory, gpu
    performance_metrics: Dict[str, float]  # requests_per_second, avg_response_time
    deployment_config: Dict[str, Any]
    owner_id: str

@dataclass
class EnterpriseUser:
    """Enterprise user with role-based access"""
    user_id: str
    email: str
    role: str  # "admin", "developer", "analyst", "viewer"
    permissions: List[str]
    instances: List[str]  # Instance IDs this user can access
    created_at: datetime
    last_login: datetime
    api_usage: Dict[str, Any]

@dataclass
class DeploymentConfig:
    """Configuration for deploying Janus instances"""
    instance_name: str
    model_size: str
    capabilities: List[str]
    resource_limits: Dict[str, Any]
    auto_scaling: bool = False
    backup_enabled: bool = True
    monitoring_enabled: bool = True
    security_config: Dict[str, Any] = field(default_factory=dict)

class EnterpriseDatabase:
    """Database for enterprise management"""
    
    def __init__(self, db_path: str = "enterprise.db"):
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
                role TEXT NOT NULL,
                permissions TEXT,
                instances TEXT,
                created_at TEXT,
                last_login TEXT,
                api_usage TEXT
            )
        """)
        
        # Instances table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS instances (
                instance_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                model_size TEXT NOT NULL,
                capabilities TEXT,
                created_at TEXT,
                last_active TEXT,
                resource_usage TEXT,
                performance_metrics TEXT,
                deployment_config TEXT,
                owner_id TEXT
            )
        """)
        
        # Usage logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                log_id TEXT PRIMARY KEY,
                timestamp TEXT,
                user_id TEXT,
                instance_id TEXT,
                operation TEXT,
                resource_cost REAL,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_user(self, user: EnterpriseUser):
        """Add user to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO users (user_id, email, role, permissions, instances, created_at, last_login, api_usage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user.user_id,
            user.email,
            user.role,
            json.dumps(user.permissions),
            json.dumps(user.instances),
            user.created_at.isoformat(),
            user.last_login.isoformat(),
            json.dumps(user.api_usage)
        ))
        
        conn.commit()
        conn.close()
    
    def get_user(self, user_id: str) -> Optional[EnterpriseUser]:
        """Get user from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return EnterpriseUser(
                user_id=row[0],
                email=row[1],
                role=row[2],
                permissions=json.loads(row[3]),
                instances=json.loads(row[4]),
                created_at=datetime.fromisoformat(row[5]),
                last_login=datetime.fromisoformat(row[6]),
                api_usage=json.loads(row[7])
            )
        return None
    
    def add_instance(self, instance: EnterpriseInstance):
        """Add instance to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO instances (instance_id, name, status, model_size, capabilities, created_at, last_active,
                                   resource_usage, performance_metrics, deployment_config, owner_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            instance.instance_id,
            instance.name,
            instance.status,
            instance.model_size,
            json.dumps(instance.capabilities),
            instance.created_at.isoformat(),
            instance.last_active.isoformat(),
            json.dumps(instance.resource_usage),
            json.dumps(instance.performance_metrics),
            json.dumps(instance.deployment_config),
            instance.owner_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_instances_for_user(self, user_id: str) -> List[EnterpriseInstance]:
        """Get all instances for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM instances WHERE owner_id = ?", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        
        instances = []
        for row in rows:
            instances.append(EnterpriseInstance(
                instance_id=row[0],
                name=row[1],
                status=row[2],
                model_size=row[3],
                capabilities=json.loads(row[4]),
                created_at=datetime.fromisoformat(row[5]),
                last_active=datetime.fromisoformat(row[6]),
                resource_usage=json.loads(row[7]),
                performance_metrics=json.loads(row[8]),
                deployment_config=json.loads(row[9]),
                owner_id=row[10]
            ))
        
        return instances

class JanusEnterpriseDashboard:
    """Main enterprise dashboard service"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Janus Enterprise Dashboard",
            description="Enterprise management for Janus AI instances",
            version="1.0.0"
        )
        
        # Setup CORS and static files
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount static files for frontend
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # Security
        self.security = HTTPBearer()
        
        # Database
        self.db = EnterpriseDatabase()
        
        # Active instances (in production, these would be actual running processes)
        self.active_instances: Dict[str, Any] = {}
        
        # WebSocket connections for real-time monitoring
        self.websockets: Dict[str, WebSocket] = {}
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Janus Enterprise Dashboard initialized")
    
    def _setup_routes(self):
        """Setup all dashboard routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Serve dashboard homepage"""
            return self._get_dashboard_html()
        
        @self.app.post("/enterprise/auth/login")
        async def enterprise_login(email: str, password: str):
            """Enterprise login (simplified for demo)"""
            # In production, use proper authentication
            user_id = str(uuid.uuid4())
            
            user = EnterpriseUser(
                user_id=user_id,
                email=email,
                role="admin",
                permissions=["manage_instances", "view_analytics", "manage_users"],
                instances=[],
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow(),
                api_usage={"requests_today": 0, "tokens_used": 0}
            )
            
            self.db.add_user(user)
            
            return {
                "user_id": user_id,
                "role": user.role,
                "permissions": user.permissions,
                "api_key": f"enterprise_{uuid.uuid4().hex}"
            }
        
        @self.app.post("/enterprise/instances/deploy")
        async def deploy_instance(
            config: DeploymentConfig,
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Deploy new Janus instance"""
            # Extract user from token (simplified)
            user_id = "demo_user"  # In production, validate token properly
            
            # Create instance
            instance_id = str(uuid.uuid4())
            instance = EnterpriseInstance(
                instance_id=instance_id,
                name=config.instance_name,
                status="starting",
                model_size=config.model_size,
                capabilities=config.capabilities,
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow(),
                resource_usage={"cpu": 0.0, "memory": 0.0, "gpu": 0.0},
                performance_metrics={"requests_per_second": 0.0, "avg_response_time": 0.0},
                deployment_config=asdict(config),
                owner_id=user_id
            )
            
            self.db.add_instance(instance)
            
            # Start instance (simulated)
            await self._start_instance(instance)
            
            return {
                "instance_id": instance_id,
                "status": "deploying",
                "estimated_ready_time": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
            }
        
        @self.app.get("/enterprise/instances")
        async def list_instances(
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """List all instances for user"""
            user_id = "demo_user"  # Extract from token in production
            instances = self.db.get_instances_for_user(user_id)
            
            return {
                "instances": [asdict(instance) for instance in instances],
                "total_instances": len(instances),
                "running_instances": len([i for i in instances if i.status == "running"])
            }
        
        @self.app.get("/enterprise/instances/{instance_id}")
        async def get_instance_details(
            instance_id: str,
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Get detailed instance information"""
            instances = self.db.get_instances_for_user("demo_user")
            instance = next((i for i in instances if i.instance_id == instance_id), None)
            
            if not instance:
                raise HTTPException(status_code=404, detail="Instance not found")
            
            # Get real-time metrics
            real_time_metrics = self._get_instance_metrics(instance_id)
            
            return {
                "instance": asdict(instance),
                "real_time_metrics": real_time_metrics,
                "logs": self._get_instance_logs(instance_id, limit=50)
            }
        
        @self.app.post("/enterprise/instances/{instance_id}/action")
        async def instance_action(
            instance_id: str,
            action: str,  # "start", "stop", "restart", "scale"
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Perform action on instance"""
            instances = self.db.get_instances_for_user("demo_user")
            instance = next((i for i in instances if i.instance_id == instance_id), None)
            
            if not instance:
                raise HTTPException(status_code=404, detail="Instance not found")
            
            # Perform action
            if action == "start":
                await self._start_instance(instance)
            elif action == "stop":
                await self._stop_instance(instance)
            elif action == "restart":
                await self._restart_instance(instance)
            elif action == "scale":
                await self._scale_instance(instance)
            else:
                raise HTTPException(status_code=400, detail="Invalid action")
            
            return {
                "instance_id": instance_id,
                "action": action,
                "status": instance.status,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/enterprise/analytics/overview")
        async def get_analytics_overview(
            credentials: HTTPAuthorizationCredentials = Security(self.security)
        ):
            """Get enterprise analytics overview"""
            instances = self.db.get_instances_for_user("demo_user")
            
            # Calculate metrics
            total_requests = sum(i.performance_metrics.get("requests_per_second", 0) * 3600 for i in instances)
            avg_response_time = sum(i.performance_metrics.get("avg_response_time", 0) for i in instances) / max(1, len(instances))
            total_cost = self._calculate_monthly_cost(instances)
            
            return {
                "overview": {
                    "total_instances": len(instances),
                    "running_instances": len([i for i in instances if i.status == "running"]),
                    "total_requests_per_hour": total_requests,
                    "average_response_time": avg_response_time,
                    "estimated_monthly_cost": total_cost,
                    "uptime_percentage": 99.9  # Calculate from logs
                },
                "instance_breakdown": [
                    {
                        "instance_id": i.instance_id,
                        "name": i.name,
                        "status": i.status,
                        "model_size": i.model_size,
                        "requests_per_second": i.performance_metrics.get("requests_per_second", 0),
                        "cost_per_month": self._calculate_instance_cost(i)
                    }
                    for i in instances
                ]
            }
        
        @self.app.websocket("/enterprise/monitoring/{instance_id}")
        async def websocket_monitoring(websocket: WebSocket, instance_id: str):
            """WebSocket for real-time monitoring"""
            await websocket.accept()
            self.websockets[instance_id] = websocket
            
            try:
                while True:
                    # Send real-time metrics
                    metrics = self._get_instance_metrics(instance_id)
                    await websocket.send_json({
                        "type": "metrics",
                        "instance_id": instance_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": metrics
                    })
                    
                    await asyncio.sleep(5)  # Update every 5 seconds
            except WebSocketDisconnect:
                if instance_id in self.websockets:
                    del self.websockets[instance_id]
    
    async def _start_instance(self, instance: EnterpriseInstance):
        """Start a Janus instance"""
        instance.status = "starting"
        
        # Simulate startup time
        await asyncio.sleep(2)
        
        instance.status = "running"
        instance.last_active = datetime.utcnow()
        
        # Initialize metrics
        instance.resource_usage = {"cpu": 15.0, "memory": 25.0, "gpu": 0.0}
        instance.performance_metrics = {"requests_per_second": 10.5, "avg_response_time": 0.15}
        
        # Store in active instances
        self.active_instances[instance.instance_id] = instance
        
        logger.info(f"Started instance {instance.instance_id}")
    
    async def _stop_instance(self, instance: EnterpriseInstance):
        """Stop a Janus instance"""
        instance.status = "stopping"
        await asyncio.sleep(1)
        instance.status = "stopped"
        
        if instance.instance_id in self.active_instances:
            del self.active_instances[instance.instance_id]
        
        logger.info(f"Stopped instance {instance.instance_id}")
    
    async def _restart_instance(self, instance: EnterpriseInstance):
        """Restart a Janus instance"""
        await self._stop_instance(instance)
        await asyncio.sleep(2)
        await self._start_instance(instance)
        
        logger.info(f"Restarted instance {instance.instance_id}")
    
    async def _scale_instance(self, instance: EnterpriseInstance):
        """Scale a Janus instance"""
        # Simulate scaling
        instance.performance_metrics["requests_per_second"] *= 2
        instance.resource_usage["cpu"] *= 1.5
        instance.resource_usage["memory"] *= 1.5
        
        logger.info(f"Scaled instance {instance.instance_id}")
    
    def _get_instance_metrics(self, instance_id: str) -> Dict[str, Any]:
        """Get real-time metrics for instance"""
        if instance_id in self.active_instances:
            instance = self.active_instances[instance_id]
            return {
                "resource_usage": instance.resource_usage,
                "performance_metrics": instance.performance_metrics,
                "uptime": (datetime.utcnow() - instance.created_at).total_seconds(),
                "last_request": instance.last_active.isoformat()
            }
        return {}
    
    def _get_instance_logs(self, instance_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get instance logs"""
        # Simulate logs
        logs = [
            {
                "timestamp": (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
                "level": "INFO" if i % 5 != 0 else "WARNING",
                "message": f"Instance {instance_id} - Log entry {i}",
                "source": "janus-core"
            }
            for i in range(limit)
        ]
        return logs
    
    def _calculate_monthly_cost(self, instances: List[EnterpriseInstance]) -> float:
        """Calculate total monthly cost for instances"""
        total = 0.0
        for instance in instances:
            total += self._calculate_instance_cost(instance)
        return total
    
    def _calculate_instance_cost(self, instance: EnterpriseInstance) -> float:
        """Calculate monthly cost for single instance"""
        # Pricing model
        base_costs = {
            "1b": 100,
            "3b": 250,
            "7b": 500,
            "13b": 1000,
            "34b": 2500,
            "70b": 5000
        }
        
        base_cost = base_costs.get(instance.model_size, 100)
        
        # Add capability costs
        capability_costs = {
            "video_analysis": 200,
            "speech_synthesis": 150,
            "holographic_memory": 100,
            "autonomous_execution": 300
        }
        
        for capability in instance.capabilities:
            base_cost += capability_costs.get(capability, 0)
        
        return base_cost
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Janus Enterprise Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 1rem; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .card { background: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; }
        .metric { text-align: center; padding: 1rem; background: #ecf0f1; border-radius: 8px; }
        .metric h3 { margin: 0; color: #2c3e50; }
        .metric .value { font-size: 2rem; font-weight: bold; color: #3498db; }
        .btn { background: #3498db; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #2980b9; }
        .instance-list { margin-top: 1rem; }
        .instance-item { display: flex; justify-content: space-between; align-items: center; padding: 1rem; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 0.5rem; }
        .status-running { color: #27ae60; }
        .status-stopped { color: #e74c3c; }
        .status-starting { color: #f39c12; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Janus Enterprise Dashboard</h1>
        <p>Manage and monitor your autonomous AI instances</p>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>Overview</h2>
            <div class="metrics">
                <div class="metric">
                    <h3>Total Instances</h3>
                    <div class="value" id="total-instances">0</div>
                </div>
                <div class="metric">
                    <h3>Running</h3>
                    <div class="value" id="running-instances">0</div>
                </div>
                <div class="metric">
                    <h3>Requests/Hour</h3>
                    <div class="value" id="requests-hour">0</div>
                </div>
                <div class="metric">
                    <h3>Monthly Cost</h3>
                    <div class="value" id="monthly-cost">$0</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Instances</h2>
            <button class="btn" onclick="deployInstance()">Deploy New Instance</button>
            <div class="instance-list" id="instance-list">
                <!-- Instances will be loaded here -->
            </div>
        </div>
        
        <div class="card">
            <h2>Performance</h2>
            <canvas id="performance-chart"></canvas>
        </div>
    </div>
    
    <script>
        // Load dashboard data
        async function loadDashboard() {
            try {
                const response = await fetch('/enterprise/analytics/overview');
                const data = await response.json();
                
                // Update metrics
                document.getElementById('total-instances').textContent = data.overview.total_instances;
                document.getElementById('running-instances').textContent = data.overview.running_instances;
                document.getElementById('requests-hour').textContent = Math.round(data.overview.total_requests_per_hour);
                document.getElementById('monthly-cost').textContent = `$${data.overview.estimated_monthly_cost}`;
                
                // Load instances
                loadInstances();
                
            } catch (error) {
                console.error('Error loading dashboard:', error);
            }
        }
        
        async function loadInstances() {
            try {
                const response = await fetch('/enterprise/instances');
                const data = await response.json();
                
                const instanceList = document.getElementById('instance-list');
                instanceList.innerHTML = '';
                
                data.instances.forEach(instance => {
                    const item = document.createElement('div');
                    item.className = 'instance-item';
                    item.innerHTML = `
                        <div>
                            <strong>${instance.name}</strong>
                            <span class="status-${instance.status}">${instance.status}</span>
                            <br>
                            <small>${instance.model_size} - ${instance.capabilities.join(', ')}</small>
                        </div>
                        <div>
                            <button class="btn" onclick="instanceAction('${instance.instance_id}', 'start')">Start</button>
                            <button class="btn" onclick="instanceAction('${instance.instance_id}', 'stop')">Stop</button>
                            <button class="btn" onclick="instanceAction('${instance.instance_id}', 'restart')">Restart</button>
                        </div>
                    `;
                    instanceList.appendChild(item);
                });
                
            } catch (error) {
                console.error('Error loading instances:', error);
            }
        }
        
        async function instanceAction(instanceId, action) {
            try {
                const response = await fetch(`/enterprise/instances/${instanceId}/action`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action: action })
                });
                
                if (response.ok) {
                    loadInstances(); // Reload instances
                }
            } catch (error) {
                console.error('Error performing action:', error);
            }
        }
        
        function deployInstance() {
            // Simple deployment dialog
            const name = prompt('Instance name:');
            const modelSize = prompt('Model size (1b, 3b, 7b, 13b, 34b, 70b):');
            
            if (name && modelSize) {
                fetch('/enterprise/instances/deploy', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        instance_name: name,
                        model_size: modelSize,
                        capabilities: ['text_generation', 'holographic_memory'],
                        resource_limits: {},
                        auto_scaling: false,
                        backup_enabled: true,
                        monitoring_enabled: true
                    })
                }).then(() => loadInstances());
            }
        }
        
        // Initialize dashboard
        loadDashboard();
        setInterval(loadDashboard, 30000); // Refresh every 30 seconds
    </script>
</body>
</html>
        """
    
    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the enterprise dashboard"""
        uvicorn.run(self.app, host=host, port=port, log_level="info")

# Create global dashboard instance
dashboard = JanusEnterpriseDashboard()

if __name__ == "__main__":
    dashboard.run(host="0.0.0.0", port=8001)
