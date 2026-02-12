import pytest
from unittest.mock import patch, MagicMock
from med_safety_eval.github_client import GithubClient

@pytest.fixture
def client():
    return GithubClient(repo="surfiniaburger/med-safety-gym", token="mock-token")

@pytest.mark.anyio
async def test_fetch_results_list(client):
    """Verify we can list files in the results directory."""
    mock_response = [
        {"name": "result_1.json", "path": "results/result_1.json", "type": "file"},
        {"name": "readme.md", "path": "results/readme.md", "type": "file"}
    ]
    
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: mock_response)
        
        files = await client.fetch_directory_contents("results")
        
        assert len(files) == 1
        assert files[0] == "results/result_1.json"
        mock_get.assert_called_once()

@pytest.mark.anyio
async def test_download_and_parse_artifact(client):
    """Verify we can download and parse a specific JSON artifact."""
    mock_json = {
        "results": [
            {
                "summary": {
                    "detailed_results": [
                        {"index": 0, "reward": 10.0, "response": "Safe answer", "metrics": {"safe": True}}
                    ]
                }
            }
        ]
    }
    
    with patch("httpx.AsyncClient.get") as mock_get:
        # Mocking the metadata fetch
        mock_metadata = {"download_url": "http://mock-download/result.json"}
        # Mocking the actual content fetch
        mock_content = mock_json
        
        mock_get.side_effect = [
            MagicMock(status_code=200, json=lambda: mock_metadata),
            MagicMock(status_code=200, json=lambda: mock_content)
        ]
    
        snapshots = await client.download_and_parse_artifact("results/result_1.json")
    
        assert len(snapshots) == 1
        assert snapshots[0]["scores"]["root"] == 10.0
        assert snapshots[0]["metadata"]["source"] == "results/result_1.json"
