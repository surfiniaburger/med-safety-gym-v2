import os
import httpx
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("github_client")

class GithubClient:
    """
    Client for fetching historical evaluation artifacts from GitHub.
    """
    def __init__(self, repo: str = "surfiniaburger/med-safety-gym", token: Optional[str] = None):
        self.repo = repo
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.base_url = f"https://api.github.com/repos/{repo}/contents"
        
    def _get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers

    async def fetch_directory_contents(self, path: str) -> List[str]:
        """
        Fetches the list of files in a specific directory on GitHub.
        """
        url = f"{self.base_url}/{path}"
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, headers=self._get_headers())
                resp.raise_for_status()
                items = resp.json()
                # Return only paths to JSON files
                return [item["path"] for item in items if item["type"] == "file" and item["name"].endswith(".json")]
            except Exception as e:
                logger.error(f"Failed to fetch directory {path}: {e}")
                return []

    async def download_and_parse_artifact(self, path: str) -> List[Dict[str, Any]]:
        """
        Downloads a JSON artifact from GitHub and parses it into Hub snapshots.
        """
        url = f"{self.base_url}/{path}"
        async with httpx.AsyncClient() as client:
            try:
                # 1. Get file metadata (including download URL)
                resp = await client.get(url, headers=self._get_headers())
                resp.raise_for_status()
                data = resp.json()
                
                # 2. Extract contents (GitHub might return 'content' as base64 or a download_url)
                if "download_url" in data:
                    content_resp = await client.get(data["download_url"])
                    content_resp.raise_for_status()
                    artifact_data = content_resp.json()
                else:
                    return []

                # 3. Parse into Snapshots (Logic from DataAgent.sync_github_results)
                snapshots = []
                results_list = []
                if "results" in artifact_data:
                    if isinstance(artifact_data["results"], list):
                        for outer_res in artifact_data["results"]:
                            if "results" in outer_res:
                                results_list.extend(outer_res["results"])
                            else:
                                results_list.append(outer_res)

                for res in results_list:
                    if "summary" in res and "detailed_results" in res["summary"]:
                        filename = os.path.basename(path)
                        for detail in res["summary"]["detailed_results"]:
                            snapshot = {
                                "session_id": f"archived-{filename}",
                                "step": detail["index"],
                                "scores": detail.get("metrics", {}),
                                "metadata": {
                                    "response": detail.get("response", ""),
                                    "reward": detail.get("reward", 0.0),
                                    "ground_truth": detail.get("ground_truth", {}),
                                    "source": path
                                }
                            }
                            snapshot["scores"]["root"] = detail.get("reward", 0.0)
                            snapshots.append(snapshot)
                
                return snapshots

            except Exception as e:
                logger.error(f"Failed to download/parse artifact {path}: {e}")
                return []
