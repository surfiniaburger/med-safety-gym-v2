import asyncio
import httpx
import json

from med_safety_eval.observability_hub import app
from med_safety_gym.crypto import verify_signature
from cryptography.hazmat.primitives import serialization

async def main():
    async with httpx.AsyncClient() as client:
        hub_url = "http://localhost:8000"
        
        # pubkey
        print("Fetching pubkey...")
        pub_resp = await client.get(f"{hub_url}/manifest/pubkey", timeout=10.0)
        pub_pem = pub_resp.json().get("pubkey")
        
        # manifest
        print("Fetching manifest...")
        man_resp = await client.get(f"{hub_url}/manifest", timeout=10.0)
        data = man_resp.json()
        
        manifest_dict = data.get("manifest")
        signature_hex = data.get("signature")
        
        manifest_json = json.dumps(manifest_dict, sort_keys=True)
        pub_key = serialization.load_pem_public_key(pub_pem.encode())
        
        is_valid = verify_signature(manifest_json.encode(), bytes.fromhex(signature_hex), pub_key)
        print("Is signature valid?", is_valid)
        if not is_valid:
            print("Client generated JSON:", repr(manifest_json))

if __name__ == "__main__":
    asyncio.run(main())
