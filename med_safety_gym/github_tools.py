"""
GitHub Integration for SafeClaw using FastMCP 3.0.
Implements repo listing, issue management, and session-based configuration.
"""

import os
import logging
from typing import Optional, List
from github import Github, GithubException
from fastmcp import FastMCP, Context
from fastmcp.server.auth import require_scopes

# Initialize FastMCP Server
mcp = FastMCP("GitHub SafeClaw")
logger = logging.getLogger(__name__)

# --- GitHub Client Helper ---
def get_github_client():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN environment variable is required")
    return Github(token)

# --- MCP Tools ---

@mcp.tool()
async def configure_repo(repo_name: str, ctx: Context) -> str:
    """
    Configure the target repository for this session.
    Example: 'surfiniaburger/med-safety-gym-v2'
    """
    # Use FastMCP 3.0 Context for session state
    await ctx.set_state("repo_name", repo_name)
    return f"Target repository set to: {repo_name}"

@mcp.tool()
async def list_issues(ctx: Context, state: str = "open") -> str:
    """List issues in the configured repository."""
    repo_name = await ctx.get_state("repo_name")
    if not repo_name:
         return "Error: Repository not configured. Call configure_repo first."
    
    try:
        g = get_github_client()
        repo = g.get_repo(repo_name)
        issues = repo.get_issues(state=state)
        
        if issues.totalCount == 0:
            return f"No {state} issues found in {repo_name}."
            
        result = [f"Issues in {repo_name}:"]
        for issue in issues[:10]:  # Limit to 10
            result.append(f"- #{issue.number}: {issue.title}")
        return "\n".join(result)
    except Exception as e:
        return f"Error listing issues: {str(e)}"

@mcp.tool(auth=require_scopes("write"))
async def create_issue(title: str, body: str, ctx: Context) -> str:
    """
    Create a new issue in the configured repository.
    Requires 'write' scope.
    """
    repo_name = await ctx.get_state("repo_name")
    if not repo_name:
         return "Error: Repository not configured. Call configure_repo first."
         
    try:
        g = get_github_client()
        repo = g.get_repo(repo_name)
        issue = repo.create_issue(title=title, body=body)
        return f"Successfully created issue #{issue.number}: {issue.html_url}"
    except Exception as e:
        return f"Error creating issue: {str(e)}"

# --- Advanced / Admin Tools (Hidden by default) ---

@mcp.tool(tags=["admin"])
async def delete_issue_comment(issue_number: int, comment_id: int, ctx: Context) -> str:
    """Delete a comment from an issue (Admin only)."""
    repo_name = await ctx.get_state("repo_name")
    try:
        g = get_github_client()
        repo = g.get_repo(repo_name)
        issue = repo.get_issue(number=issue_number)
        comment = issue.get_comment(comment_id)
        comment.delete()
        return f"Deleted comment {comment_id} on issue #{issue_number}"
    except Exception as e:
        return f"Error deleting comment: {str(e)}"

@mcp.tool()
async def unlock_admin_tools(ctx: Context) -> str:
    """Enable administrative tools for this session."""
    # FastMCP 3.0 Visibility system
    await ctx.enable_components(tags={"admin"})
    return "Administrative GitHub tools are now available for this session."

@mcp.tool()
async def list_pull_requests(ctx: Context, state: str = "open") -> str:
    """List pull requests in the configured repository."""
    repo_name = await ctx.get_state("repo_name")
    if not repo_name:
         return "Error: Repository not configured. Call configure_repo first."
    
    try:
        g = get_github_client()
        repo = g.get_repo(repo_name)
        pulls = repo.get_pulls(state=state)
        
        if pulls.totalCount == 0:
            return f"No {state} pull requests found in {repo_name}."
            
        result = [f"Pull Requests in {repo_name}:"]
        for pr in pulls[:10]:
            result.append(f"- #{pr.number}: {pr.title} ({pr.user.login})")
        return "\n".join(result)
    except Exception as e:
        return f"Error listing PRs: {str(e)}"

# Initialize visibility: Hide admin tools by default
mcp.disable(tags={"admin"})

if __name__ == "__main__":
    mcp.run()
