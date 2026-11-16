"""Publish a GitHub release using existing noninteractive Git credentials."""

import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY = "jermmy19998/Tri-MIL"


def github_token():
    """Read an existing token in memory, without logging it or prompting for login."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    environment = dict(os.environ, GCM_INTERACTIVE="never", GIT_TERMINAL_PROMPT="0")
    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    if result.returncode:
        return None
    credentials = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    return credentials.get("password")


def request(token, path, payload=None):
    """Send a GitHub JSON request and return the decoded response without secrets."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        "https://api.github.com/" + path,
        data=data,
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "Tri-MIL-release",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def main():
    """Check release access, or publish the already-pushed version tag on request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--tag", default="v0.4.0")
    args = parser.parse_args()
    token = github_token()
    if not token:
        print("No noninteractive GitHub API credential is available.")
        raise SystemExit(2)
    repository = request(token, f"repos/{REPOSITORY}")
    print(
        f"Repository: {repository['full_name']}; push access: {repository.get('permissions', {}).get('push', False)}"
    )
    if not args.publish:
        return
    try:
        existing = request(token, f"repos/{REPOSITORY}/releases/tags/{args.tag}")
        print(f"Release already exists: {existing['html_url']}")
        return
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    root = Path(__file__).resolve().parents[1]
    result = request(
        token,
        f"repos/{REPOSITORY}/releases",
        {
            "tag_name": args.tag,
            "target_commitish": "release",
            "name": "Tri-MIL 0.4.0: upstream updates and survival prediction",
            "body": (root / "CHANGELOG.md").read_text(encoding="utf-8"),
            "draft": False,
            "prerelease": False,
        },
    )
    print(f"Published release: {result['html_url']}")


if __name__ == "__main__":
    main()
