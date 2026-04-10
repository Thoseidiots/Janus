import grpc
import json
import asyncio
from typing import AsyncIterator

from janus_brain import nexus_pb2
from janus_brain import nexus_pb2_grpc

class NexusClient:
    def __init__(self, host: str = "localhost", port: int = 50051):
        self.channel = grpc.aio.insecure_channel(f"{host}:{port}")
        self.stub = nexus_pb2_grpc.NexusServiceStub(self.channel)

    async def execute_command(self, command_type: str, payload: dict) -> dict:
        request = nexus_pb2.NexusCommand(
            command_type=command_type,
            payload=json.dumps(payload)
        )
        response = await self.stub.ExecuteCommand(request)
        if response.success:
            return json.loads(response.result_payload)
        else:
            raise Exception(f"Nexus command failed: {response.message}")

    async def stream_state_updates(self) -> AsyncIterator[dict]:
        async for update in self.stub.StreamStateUpdates(nexus_pb2.Empty()):
            yield {
                "update_type": update.update_type,
                "state_payload": json.loads(update.state_payload)
            }

    async def close(self):
        await self.channel.close()

if __name__ == "__main__":
    async def main():
        client = NexusClient()
        try:
            # Example: Add an event
            event_payload = {
                "id": "test-event-1",
                "timestamp": "2026-02-25T10:00:00Z",
                "event_type": "test_event",
                "data": {"message": "Hello from Python client"}
            }
            response = await client.execute_command("add_event", event_payload)
            print(f"Execute command response: {response}")

            # Example: Stream state updates
            print("\nStreaming state updates...")
            async for update in client.stream_state_updates():
                print(f"Received state update: {update}")
                # For demonstration, break after a few updates
                if update["update_type"] == "command_applied":
                    break

        except Exception as e:
            print(f"Error: {e}")
        finally:
            await client.close()

    asyncio.run(main())
