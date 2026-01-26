import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://127.0.0.1:8001/ws/data-stream"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            print("Waiting for data...")
            
            for i in range(3):
                message = await websocket.recv()
                data = json.loads(message)
                print(f"\nReceived message {i+1}:")
                print(f"  ID: {data.get('id')}")
                print(f"  Timestamp: {data.get('timestamp')}")
                print(f"  Data keys: {len(data.get('data', {}))} features")
                print(f"  Sample features: {list(data.get('data', {}).keys())[:5]}")
    except websockets.exceptions.ConnectionRefused:
        print(f"ERROR: Could not connect to {uri}")
        print("Make sure the server is running with: uvicorn model_api:app --host 127.0.0.1 --port 8001 --reload")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
