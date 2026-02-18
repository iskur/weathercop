"""Trigger a GitHub Actions workflow and wait for completion."""
import requests
import time
import sys
import os


def main():
    """Trigger GitHub workflow and poll for completion."""
    GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
    GITHUB_REPO = "iskur/weathercop"
    WORKFLOW_ID = "build-wheels.yml"
    # Use CI_COMMIT_REF (always available for branches/tags) instead of CI_COMMIT_TAG (only for tags)
    REF = os.environ['CI_COMMIT_REF']

    # === DEBUG: Component 1 - Environment ===
    print(f"[ENV] GITHUB_TOKEN present: {bool(GITHUB_TOKEN)}")
    print(f"[ENV] GITHUB_TOKEN length: {len(GITHUB_TOKEN)}")
    print(f"[ENV] GITHUB_TOKEN type: {type(GITHUB_TOKEN).__name__}")
    # Check if token looks like a literal variable reference (not expanded)
    if GITHUB_TOKEN.startswith('$') or GITHUB_TOKEN == 'GITHUB_TOKEN':
        print(f"[ENV] WARNING: Token appears to be unexpanded: {GITHUB_TOKEN}")
    else:
        print(f"[ENV] Token appears to be expanded (does not start with $)")
    # Check token format (classic tokens start with 'ghp_', fine-grained start with 'github_pat_')
    if GITHUB_TOKEN.startswith(('ghp_', 'ghu_', 'ghs_', 'ghr_', 'github_pat_')):
        print(f"[ENV] Token format looks valid")
    else:
        print(f"[ENV] WARNING: Token format does not match known GitHub token prefixes")
    print(f"[ENV] CI_COMMIT_REF: {REF}")

    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }

    # === DEBUG: Component 2 - Headers before request ===
    print(f"[HEADERS] Authorization starts with: {headers['Authorization'][:30]}...")

    # === DEBUG: Component 3 - Test basic auth first ===
    print("\n[TEST] Verifying token with GitHub API...")
    test_url = "https://api.github.com/user"
    test_response = requests.get(test_url, headers=headers)
    print(f"[TEST] GitHub /user response: {test_response.status_code}")
    if test_response.status_code == 200:
        user_data = test_response.json()
        print(f"[TEST] Authenticated as: {user_data.get('login')}")
    else:
        print(f"[TEST] Auth failed! Response: {test_response.text}")
        sys.exit(1)

    # 1. Trigger workflow
    print(f"\n[WORKFLOW] Triggering GitHub workflow for ref: {REF}")
    trigger_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_ID}/dispatches"
    trigger_payload = {
        'ref': REF,
        'inputs': {'ref': REF}
    }
    print(f"[WORKFLOW] URL: {trigger_url}")
    print(f"[WORKFLOW] Payload: {trigger_payload}")
    response = requests.post(trigger_url, json=trigger_payload, headers=headers)

    if response.status_code != 204:
        print(f"Failed to trigger workflow: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response body: {response.text}")
        sys.exit(1)

    print("✓ Workflow triggered successfully")

    # 2. Wait for workflow to start (give it 30 seconds)
    time.sleep(30)

    # 3. Find the workflow run
    runs_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW_ID}/runs"
    params = {'branch': REF, 'per_page': 1}
    response = requests.get(runs_url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"Failed to list workflow runs: {response.status_code}")
        sys.exit(1)

    runs = response.json()['workflow_runs']
    if not runs:
        print("No workflow runs found")
        sys.exit(1)

    run_id = runs[0]['id']
    run_url = runs[0]['html_url']
    print(f"Found workflow run: {run_url}")

    # 4. Poll for completion (timeout after 60 minutes)
    max_wait = 60 * 60  # 60 minutes
    poll_interval = 30  # Check every 30 seconds
    elapsed = 0

    while elapsed < max_wait:
        run_status_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{run_id}"
        response = requests.get(run_status_url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to get run status: {response.status_code}")
            sys.exit(1)

        run_data = response.json()
        status = run_data['status']
        conclusion = run_data.get('conclusion')

        print(f"[{elapsed}s] Status: {status}, Conclusion: {conclusion}")

        if status == 'completed':
            if conclusion == 'success':
                print("✓ GitHub workflow completed successfully")
                sys.exit(0)
            else:
                print(f"✗ GitHub workflow failed with conclusion: {conclusion}")
                print(f"View details: {run_url}")
                sys.exit(1)

        time.sleep(poll_interval)
        elapsed += poll_interval

    print(f"✗ Timeout: GitHub workflow did not complete within {max_wait}s")
    print(f"View status: {run_url}")
    sys.exit(1)


if __name__ == '__main__':
    main()