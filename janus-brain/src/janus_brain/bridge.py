import sys
import json
import asyncio
import ray
import uuid
from datetime import datetime

from .core import AutonomousCore
from .nexus_client import NexusClient
# Assuming janus_core types are represented as dictionaries for now
# In a full implementation, a proper FFI or serialization layer would be needed
# from janus_core import Event, Task, IdentityContract 

# Initialize Ray if not already initialized
if not ray.is_initialized():
    ray.init()

@ray.remote
class DistributedBrainBridge:
    def __init__(self, nexus_host: str = "localhost", nexus_port: int = 50051):
        self.core = AutonomousCore.remote() # Instantiate AutonomousCore as a Ray actor
        self.nexus_client = NexusClient(nexus_host, nexus_port)

    async def handle_request(self, request_json: str) -> str:
        try:
            request = json.loads(request_json)
            cmd = request.get("cmd")
            
            if cmd == "perceive":
                text = request["text"]
                await self.core.perceive.remote(text)
                
                # Also send an event to the Nexus Core
                event_payload = {
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "event_type": "perception",
                    "data": {"stimulus": text}
                }
                await self.nexus_client.execute_command("add_event", event_payload)

                return json.dumps({"status": "ok"})
            
            elif cmd == "generate":
                prompt = request["prompt"]
                response = await self.core.generate_response.remote(prompt)
                
                # Also send an event to the Nexus Core
                event_payload = {
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "event_type": "generation",
                    "data": {"prompt": prompt, "response": response}
                }
                await self.nexus_client.execute_command("add_event", event_payload)

                return json.dumps({"status": "ok", "response": response})
            
            elif cmd == "reflect":
                reflection = await self.core.reflect.remote()
                
                # Also send an event to the Nexus Core
                event_payload = {
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "event_type": "reflection",
                    "data": {"reflection_summary": reflection}
                }
                await self.nexus_client.execute_command("add_event", event_payload)

                return json.dumps({"status": "ok", "reflection": reflection})
            
            elif cmd == "update_identity":
                identity_payload = request["identity"]
                await self.nexus_client.execute_command("update_identity", identity_payload)
                return json.dumps({"status": "ok"})

            else:
                return json.dumps({"status": "error", "message": f"Unknown command: {cmd}"})
        
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

# This part would typically be run as a separate process or service
# For demonstration, we will keep the actor alive.
async def main_bridge_server():
    bridge_actor = DistributedBrainBridge.remote()
    print("DistributedBrainBridge Ray actor started.")
    # Keep the actor alive indefinitely
    while True:
        await asyncio.sleep(3600) 

if __name__ == "__main__":
    asyncio.run(main_bridge_server())
