import pytest
import respx
import httpx
from med_safety_gym.claw_agent import SafeClawAgent
from med_safety_gym.identity.secret_store import InMemorySecretStore

@pytest.mark.asyncio
async def test_agent_saves_secrets_to_store():
    store = InMemorySecretStore()
    agent = SafeClawAgent(secret_store=store)
    
    hub_url = "http://localhost:8000"
    
    async with respx.mock:
        # 1. Health check
        respx.get(f"{hub_url}/health").respond(200)
        
        # 2. Auth delegate
        respx.post(f"{hub_url}/auth/delegate").respond(200, json={"token": "mock-delegation-token"})
        
        # 3. Public key
        respx.get(f"{hub_url}/manifest/pubkey").respond(200, json={"pubkey": "mock-public-key"})
        
        # 4. Scoped manifest (minimal valid structure)
        # Assuming SkillManifest.from_dict is used and it needs specific keys
        mock_manifest_data = {
            "manifest": {"name": "TestPolicy", "version": "1.0", "tools": {}},
            "signature": "de" * 32 # Dummy signature
        }
        respx.get(f"{hub_url}/manifest/scoped").respond(200, json=mock_manifest_data)

        # Mock verify_signature to always return True in SafeClawAgent
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("med_safety_gym.claw_agent.SafeClawAgent._verify_and_parse_manifest", 
                       lambda self, data, pub: None) # Bypass parsing for this test
            
            await agent._init_scoped_session()
            
            # Verify secrets were saved to the store
            assert store.get_secret("auth_token") == "mock-delegation-token"
            assert store.get_secret("hub_pub_key") == "mock-public-key"

@pytest.mark.asyncio
async def test_agent_loads_public_key_from_store_during_boot():
    store = InMemorySecretStore()
    store.set_secret("hub_pub_key", "existing-pub-key")
    
    agent = SafeClawAgent(secret_store=store)
    
    # Check if agent loads it
    await agent._load_secrets_from_store()
    assert agent.hub_pub_key == "existing-pub-key"

@pytest.mark.asyncio
async def test_agent_resumes_session_without_rehandshake():
    store = InMemorySecretStore()
    store.set_secret("auth_token", "valid-token")
    store.set_secret("hub_pub_key", "valid-pub-key")
    
    agent = SafeClawAgent(secret_store=store)
    hub_url = "http://localhost:8000"
    
    async with respx.mock:
        # 1. Health check
        respx.get(f"{hub_url}/health").respond(200)
        
        # 2. Scoped manifest (minimal valid structure)
        mock_manifest_data = {
            "manifest": {"name": "TestPolicy", "version": "1.0", "tools": {}},
            "signature": "de" * 32
        }
        # If we resume, we only call /manifest/scoped once and NO /auth/delegate or /manifest/pubkey
        route = respx.get(f"{hub_url}/manifest/scoped").respond(200, json=mock_manifest_data)
        
        # We should NOT see these calls if resumption works
        auth_route = respx.post(f"{hub_url}/auth/delegate")
        pubkey_route = respx.get(f"{hub_url}/manifest/pubkey")

        with pytest.MonkeyPatch().context() as mp:
            from med_safety_gym.skill_manifest import SkillManifest, PermissionSet, ToolTiers
            mp.setattr("med_safety_gym.claw_agent.SafeClawAgent._verify_and_parse_manifest", 
                       lambda self, data, pub: SkillManifest(
                           name="TestPolicy", 
                           version="1.0", 
                           permissions=PermissionSet(tools=ToolTiers())
                       ))
            
            await agent._ensure_governor_interceptor()
            
            assert route.called
            assert not auth_route.called
            assert not pubkey_route.called
            assert agent.interceptor is not None
