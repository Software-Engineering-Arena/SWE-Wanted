import json
import os
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv
import backoff
import requests
import requests.exceptions
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
import gradio as gr
from gradio_leaderboard import Leaderboard, ColumnFilter
import pandas as pd
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/bot_data"
LEADERBOARD_REPO = "SWE-Arena/leaderboard_data"
LEADERBOARD_TIME_FRAME_DAYS = 180
MAX_RETRIES = 5

# GitHub organizations and repositories to track
# Focus on Apache-related repositories
TRACKED_ORGS = [
    "apache",
]

# Labels that indicate "patch wanted" status
PATCH_WANTED_LABELS = [
    "patch wanted",
    "help wanted",
    "good first issue",
    "contributions welcome",
]

# Minimum days an issue must be open to be considered "long-standing"
MIN_ISSUE_AGE_DAYS = 30

LEADERBOARD_COLUMNS = [
    ("Agent Name", "string"),
    ("Website", "string"),
    ("Total Points", "number"),
    ("Resolved Issues", "number"),
    ("Month Points", "number"),
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_github_token():
    """Get GitHub token from environment variables."""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("Warning: GITHUB_TOKEN not found in environment variables")
    return token


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


def normalize_date_format(date_string):
    """Convert date strings or datetime objects to standardized ISO 8601 format with Z suffix."""
    if not date_string or date_string == 'N/A':
        return 'N/A'

    try:
        import re

        if isinstance(date_string, datetime):
            return date_string.strftime('%Y-%m-%dT%H:%M:%SZ')

        date_string = re.sub(r'\\s+', ' ', date_string.strip())
        date_string = date_string.replace(' ', 'T')

        if len(date_string) >= 3:
            if date_string[-3:-2] in ('+', '-') and ':' not in date_string[-3:]:
                date_string = date_string + ':00'

        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(f"Warning: Could not parse date '{date_string}': {e}")
        return date_string


# =============================================================================
# HUGGINGFACE API WRAPPERS WITH BACKOFF
# =============================================================================

def is_retryable_error(e):
    """Check if exception is retryable (rate limit or timeout error)."""
    if isinstance(e, HfHubHTTPError):
        if e.response.status_code == 429:
            return True

    if isinstance(e, (requests.exceptions.Timeout,
                     requests.exceptions.ReadTimeout,
                     requests.exceptions.ConnectTimeout)):
        return True

    if isinstance(e, Exception):
        error_str = str(e).lower()
        if 'timeout' in error_str or 'timed out' in error_str:
            return True

    return False


@backoff.on_exception(
    backoff.expo,
    (HfHubHTTPError, requests.exceptions.Timeout, requests.exceptions.RequestException, Exception),
    max_tries=MAX_RETRIES,
    base=300,
    max_value=3600,
    giveup=lambda e: not is_retryable_error(e),
    on_backoff=lambda details: print(
        f"   {details['exception']} error. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/5..."
    )
)
def list_repo_files_with_backoff(api, **kwargs):
    """Wrapper for api.list_repo_files() with exponential backoff."""
    return api.list_repo_files(**kwargs)


@backoff.on_exception(
    backoff.expo,
    (HfHubHTTPError, requests.exceptions.Timeout, requests.exceptions.RequestException, Exception),
    max_tries=MAX_RETRIES,
    base=300,
    max_value=3600,
    giveup=lambda e: not is_retryable_error(e),
    on_backoff=lambda details: print(
        f"   {details['exception']} error. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/5..."
    )
)
def hf_hub_download_with_backoff(**kwargs):
    """Wrapper for hf_hub_download() with exponential backoff."""
    return hf_hub_download(**kwargs)


@backoff.on_exception(
    backoff.expo,
    (HfHubHTTPError, requests.exceptions.Timeout, requests.exceptions.RequestException, Exception),
    max_tries=MAX_RETRIES,
    base=300,
    max_value=3600,
    giveup=lambda e: not is_retryable_error(e),
    on_backoff=lambda details: print(
        f"   {details['exception']} error. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/5..."
    )
)
def upload_file_with_backoff(api, **kwargs):
    """Wrapper for api.upload_file() with exponential backoff."""
    return api.upload_file(**kwargs)


# =============================================================================
# GITHUB API FUNCTIONS
# =============================================================================

def github_api_request(url, params=None):
    """Make a GitHub API request with authentication and rate limit handling."""
    headers = {
        'Accept': 'application/vnd.github.v3+json',
    }

    token = get_github_token()
    if token:
        headers['Authorization'] = f'token {token}'

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)

        # Check rate limit
        remaining = response.headers.get('X-RateLimit-Remaining')
        if remaining and int(remaining) < 10:
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            wait_time = max(reset_time - time.time(), 0)
            if wait_time > 0:
                print(f"   ⚠ Rate limit low ({remaining} remaining), waiting {wait_time:.0f}s...")
                time.sleep(wait_time)

        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            print(f"   ⚠ Rate limit exceeded or forbidden: {e}")
            return None
        raise
    except Exception as e:
        print(f"   ⚠ GitHub API error: {e}")
        return None


def fetch_org_repositories(org):
    """Fetch all repositories for an organization."""
    repos = []
    page = 1
    per_page = 100

    while True:
        url = f"https://api.github.com/orgs/{org}/repos"
        params = {'page': page, 'per_page': per_page, 'type': 'public'}

        data = github_api_request(url, params)
        if not data:
            break

        repos.extend(data)

        if len(data) < per_page:
            break

        page += 1
        time.sleep(0.5)  # Rate limit protection

    return repos


def has_patch_wanted_label(labels):
    """Check if issue has any patch wanted label."""
    label_names = [label.get('name', '').lower() for label in labels]
    return any(wanted_label in label_names for wanted_label in PATCH_WANTED_LABELS)


def is_long_standing_issue(created_at, min_days=MIN_ISSUE_AGE_DAYS):
    """Check if issue has been open for minimum number of days."""
    try:
        created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        age_days = (datetime.now(timezone.utc) - created).days
        return age_days >= min_days
    except:
        return False


def fetch_patch_wanted_issues(org, repo_name, state='open'):
    """Fetch issues with patch wanted labels from a repository."""
    issues = []
    page = 1
    per_page = 100

    while True:
        url = f"https://api.github.com/repos/{org}/{repo_name}/issues"
        params = {
            'state': state,
            'page': page,
            'per_page': per_page,
            'sort': 'created',
            'direction': 'asc'
        }

        data = github_api_request(url, params)
        if not data:
            break

        for issue in data:
            # Skip pull requests (they have 'pull_request' key)
            if 'pull_request' in issue:
                continue

            # Check for patch wanted labels
            if not has_patch_wanted_label(issue.get('labels', [])):
                continue

            # For open issues, check if long-standing
            if state == 'open' and not is_long_standing_issue(issue.get('created_at', '')):
                continue

            issues.append({
                'url': issue.get('html_url'),
                'title': issue.get('title'),
                'number': issue.get('number'),
                'repo': f"{org}/{repo_name}",
                'created_at': normalize_date_format(issue.get('created_at')),
                'closed_at': normalize_date_format(issue.get('closed_at')),
                'state': issue.get('state'),
                'labels': [label.get('name') for label in issue.get('labels', [])],
                'closed_by': None  # Will be filled if closed
            })

        if len(data) < per_page:
            break

        page += 1
        time.sleep(0.5)  # Rate limit protection

    return issues


def get_issue_closing_info(org, repo_name, issue_number):
    """Get information about who closed an issue and if it was via merged PR."""
    url = f"https://api.github.com/repos/{org}/{repo_name}/issues/{issue_number}/events"
    data = github_api_request(url)

    if not data:
        return None

    for event in reversed(data):  # Check from most recent
        if event.get('event') == 'closed':
            # Check if closed by a commit (merged PR)
            commit_id = event.get('commit_id')
            actor = event.get('actor', {}).get('login')

            if commit_id and actor:
                return {
                    'closed_by': actor,
                    'commit_id': commit_id,
                    'closed_via_pr': True
                }

    return None


def fetch_all_patch_wanted_issues(orgs=TRACKED_ORGS):
    """Fetch all patch wanted issues from tracked organizations."""
    all_issues = []

    for org in orgs:
        print(f"   Fetching repositories for {org}...")
        repos = fetch_org_repositories(org)
        print(f"   Found {len(repos)} repositories")

        for repo in repos[:10]:  # Limit to first 10 repos for demo (remove limit in production)
            repo_name = repo.get('name')
            print(f"   Scanning {org}/{repo_name}...")

            # Fetch open issues
            open_issues = fetch_patch_wanted_issues(org, repo_name, state='open')
            all_issues.extend(open_issues)

            time.sleep(0.5)  # Rate limit protection

    return all_issues


# =============================================================================
# AGENT LOADING
# =============================================================================

def sync_agents_repo():
    """
    Sync local bot_data repository with remote using git pull.
    This ensures we have the latest bot data.
    """
    agents_repo_local = os.path.expanduser("~/bot_data")

    if not os.path.exists(agents_repo_local):
        # Try to clone if doesn't exist
        try:
            import subprocess
            print(f"   Cloning bot_data repository...")
            result = subprocess.run(
                ['git', 'clone', f'https://huggingface.co/datasets/{AGENTS_REPO}', agents_repo_local],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print(f"   ✓ Repository cloned successfully")
                return True
            else:
                print(f"   ✗ Clone failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"   ⚠ Could not clone repository: {e}")
            return False

    try:
        import subprocess
        result = subprocess.run(
            ['git', 'pull'],
            cwd=agents_repo_local,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print(f"   ✓ Repository synced")
            return True
        else:
            print(f"   ⚠ Git pull had issues: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ⚠ Could not sync repository: {e}")
        return False


def load_agents_from_hf():
    """Load all agent metadata JSON files from HuggingFace dataset."""
    try:
        api = HfApi()
        agents = []

        # List all files in the repository
        files = list_repo_files_with_backoff(api=api, repo_id=AGENTS_REPO, repo_type="dataset")

        # Filter for JSON files only
        json_files = [f for f in files if f.endswith('.json')]

        # Download and parse each JSON file
        for json_file in json_files:
            try:
                file_path = hf_hub_download_with_backoff(
                    repo_id=AGENTS_REPO,
                    filename=json_file,
                    repo_type="dataset"
                )

                with open(file_path, 'r') as f:
                    agent_data = json.load(f)

                    # Only process agents with status == "public"
                    if agent_data.get('status') != 'public':
                        continue

                    # Extract github_identifier from filename
                    filename_identifier = json_file.replace('.json', '')
                    agent_data['github_identifier'] = filename_identifier

                    agents.append(agent_data)

            except Exception as e:
                print(f"Warning: Could not load {json_file}: {str(e)}")
                continue

        print(f"   ✓ Loaded {len(agents)} public agents from HuggingFace")
        return agents

    except Exception as e:
        print(f"Could not load agents from HuggingFace: {str(e)}")
        return []


# =============================================================================
# ISSUE TRACKING AND POINTS CALCULATION
# =============================================================================

def calculate_agent_points(issues, agents):
    """
    Calculate points for each agent based on resolved issues.
    1 point per resolved issue.
    """
    agent_identifiers = {agent.get('github_identifier') for agent in agents}
    agent_points = defaultdict(lambda: {
        'total_points': 0,
        'resolved_issues': [],
        'monthly_points': defaultdict(int)
    })

    for issue in issues:
        # Only count closed issues
        if issue.get('state') != 'closed':
            continue

        # Check who closed it
        closed_by = issue.get('closed_by')
        if not closed_by or closed_by not in agent_identifiers:
            continue

        # Award 1 point
        agent_points[closed_by]['total_points'] += 1
        agent_points[closed_by]['resolved_issues'].append(issue)

        # Track monthly points
        closed_at = issue.get('closed_at')
        if closed_at and closed_at != 'N/A':
            try:
                dt = datetime.fromisoformat(closed_at.replace('Z', '+00:00'))
                month_key = f"{dt.year}-{dt.month:02d}"
                agent_points[closed_by]['monthly_points'][month_key] += 1
            except:
                pass

    return dict(agent_points)


def calculate_monthly_metrics_by_agent(agent_points, agents):
    """Calculate monthly metrics for all agents for visualization."""
    identifier_to_name = {agent.get('github_identifier'): agent.get('name') for agent in agents}

    if not agent_points:
        return {'agents': [], 'months': [], 'data': {}}

    # Get all months
    all_months = set()
    for data in agent_points.values():
        all_months.update(data['monthly_points'].keys())
    months = sorted(list(all_months))

    result_data = {}
    for agent_id, data in agent_points.items():
        agent_name = identifier_to_name.get(agent_id, agent_id)

        monthly_points_list = []
        cumulative_points_list = []
        cumulative = 0

        for month in months:
            points = data['monthly_points'].get(month, 0)
            monthly_points_list.append(points)
            cumulative += points
            cumulative_points_list.append(cumulative)

        result_data[agent_name] = {
            'monthly_points': monthly_points_list,
            'cumulative_points': cumulative_points_list
        }

    agents_list = sorted(list(result_data.keys()))

    return {
        'agents': agents_list,
        'months': months,
        'data': result_data
    }


# =============================================================================
# DATA STORAGE
# =============================================================================

def save_leaderboard_data_to_hf(leaderboard_dict, monthly_metrics, wanted_issues):
    """Save leaderboard data, monthly metrics, and wanted issues to HuggingFace dataset."""
    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi(token=token)
        filename = "swe-wanted.json"

        combined_data = {
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'leaderboard': leaderboard_dict,
            'monthly_metrics': monthly_metrics,
            'wanted_issues': wanted_issues,
            'metadata': {
                'leaderboard_time_frame_days': LEADERBOARD_TIME_FRAME_DAYS,
                'tracked_orgs': TRACKED_ORGS,
                'patch_wanted_labels': PATCH_WANTED_LABELS
            }
        }

        with open(filename, 'w') as f:
            json.dump(combined_data, f, indent=2)

        try:
            upload_file_with_backoff(
                api=api,
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=LEADERBOARD_REPO,
                repo_type="dataset"
            )
            print(f"   ✓ Saved leaderboard data to HuggingFace")
            return True
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    except Exception as e:
        print(f"Error saving leaderboard data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def load_leaderboard_data_from_hf():
    """Load leaderboard data from HuggingFace dataset."""
    try:
        token = get_hf_token()
        filename = "swe-wanted.json"

        file_path = hf_hub_download_with_backoff(
            repo_id=LEADERBOARD_REPO,
            filename=filename,
            repo_type="dataset",
            token=token
        )

        with open(file_path, 'r') as f:
            data = json.load(f)

        last_updated = data.get('last_updated', 'Unknown')
        print(f"   ✓ Loaded leaderboard data (last updated: {last_updated})")

        return data

    except Exception as e:
        print(f"Could not load leaderboard data from HuggingFace: {str(e)}")
        return None


# =============================================================================
# LEADERBOARD CONSTRUCTION
# =============================================================================

def construct_leaderboard(agent_points, agents):
    """Construct leaderboard from agent points data."""
    if not agents:
        print("Error: No agents found")
        return {}

    leaderboard = {}

    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('name', 'Unknown')

        points_data = agent_points.get(identifier, {})
        total_points = points_data.get('total_points', 0)
        resolved_issues = len(points_data.get('resolved_issues', []))

        # Calculate current month points
        current_month = datetime.now(timezone.utc).strftime('%Y-%m')
        month_points = points_data.get('monthly_points', {}).get(current_month, 0)

        leaderboard[identifier] = {
            'name': agent_name,
            'website': agent.get('website', 'N/A'),
            'github_identifier': identifier,
            'total_points': total_points,
            'resolved_issues': resolved_issues,
            'month_points': month_points
        }

    return leaderboard


# =============================================================================
# UI FUNCTIONS
# =============================================================================

def create_monthly_metrics_plot(top_n=5):
    """Create a Plotly figure showing monthly and cumulative points for top N agents."""
    saved_data = load_leaderboard_data_from_hf()

    if not saved_data or 'monthly_metrics' not in saved_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=None, xaxis_title=None, height=500)
        return fig

    metrics = saved_data['monthly_metrics']

    # Apply top_n filter
    if top_n and top_n > 0 and metrics.get('agents'):
        agent_totals = []
        for agent_name in metrics['agents']:
            agent_data = metrics['data'].get(agent_name, {})
            total_points = max(agent_data.get('cumulative_points', [0]))
            agent_totals.append((agent_name, total_points))

        agent_totals.sort(key=lambda x: x[1], reverse=True)
        top_agents = [agent_name for agent_name, _ in agent_totals[:top_n]]

        metrics = {
            'agents': top_agents,
            'months': metrics['months'],
            'data': {agent: metrics['data'][agent] for agent in top_agents if agent in metrics['data']}
        }

    if not metrics['agents'] or not metrics['months']:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=None, xaxis_title=None, height=500)
        return fig

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    def generate_color(index, total):
        hue = (index * 360 / total) % 360
        saturation = 70 + (index % 3) * 10
        lightness = 45 + (index % 2) * 10
        return f'hsl({hue}, {saturation}%, {lightness}%)'

    agents = metrics['agents']
    months = metrics['months']
    data = metrics['data']

    agent_colors = {agent: generate_color(idx, len(agents)) for idx, agent in enumerate(agents)}

    for agent_name in agents:
        color = agent_colors[agent_name]
        agent_data = data[agent_name]

        # Add cumulative points line (left y-axis)
        fig.add_trace(
            go.Scatter(
                x=months,
                y=agent_data['cumulative_points'],
                name=agent_name,
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=8),
                legendgroup=agent_name,
                showlegend=(top_n is not None and top_n <= 10),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Month: %{x}<br>' +
                             'Cumulative Points: %{y}<br>' +
                             '<extra></extra>'
            ),
            secondary_y=False
        )

        # Add monthly points bars (right y-axis)
        x_bars = []
        y_bars = []
        for month, points in zip(months, agent_data['monthly_points']):
            if points > 0:
                x_bars.append(month)
                y_bars.append(points)

        if x_bars and y_bars:
            fig.add_trace(
                go.Bar(
                    x=x_bars,
                    y=y_bars,
                    name=agent_name,
                    marker=dict(color=color, opacity=0.6),
                    legendgroup=agent_name,
                    showlegend=False,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Monthly Points: %{y}<br>' +
                                 '<extra></extra>',
                    offsetgroup=agent_name
                ),
                secondary_y=True
            )

    fig.update_xaxes(title_text=None)
    fig.update_yaxes(
        title_text="<b>Cumulative Points</b>",
        secondary_y=False,
        showgrid=True
    )
    fig.update_yaxes(title_text="<b>Monthly Points</b>", secondary_y=True)

    show_legend = (top_n is not None and top_n <= 10)
    fig.update_layout(
        title=None,
        hovermode='closest',
        barmode='group',
        height=600,
        showlegend=show_legend,
        margin=dict(l=50, r=150 if show_legend else 50, t=50, b=50)
    )

    return fig


def get_leaderboard_dataframe():
    """Load leaderboard from saved dataset and convert to pandas DataFrame."""
    saved_data = load_leaderboard_data_from_hf()

    if not saved_data or 'leaderboard' not in saved_data:
        print(f"No leaderboard data available")
        column_names = [col[0] for col in LEADERBOARD_COLUMNS]
        return pd.DataFrame(columns=column_names)

    leaderboard = saved_data['leaderboard']
    last_updated = saved_data.get('last_updated', 'Unknown')
    print(f"Loaded leaderboard (last updated: {last_updated})")

    if not leaderboard:
        column_names = [col[0] for col in LEADERBOARD_COLUMNS]
        return pd.DataFrame(columns=column_names)

    rows = []
    for identifier, data in leaderboard.items():
        total_points = data.get('total_points', 0)

        # Filter out agents with zero points
        if total_points == 0:
            continue

        rows.append([
            data.get('name', 'Unknown'),
            data.get('website', 'N/A'),
            total_points,
            data.get('resolved_issues', 0),
            data.get('month_points', 0),
        ])

    column_names = [col[0] for col in LEADERBOARD_COLUMNS]
    df = pd.DataFrame(rows, columns=column_names)

    # Ensure numeric types
    numeric_cols = ["Total Points", "Resolved Issues", "Month Points"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Sort by Total Points descending
    if "Total Points" in df.columns and not df.empty:
        df = df.sort_values(by="Total Points", ascending=False).reset_index(drop=True)

    print(f"Leaderboard showing {len(rows)} agents with points")

    return df


def get_wanted_issues_dataframe():
    """Load wanted issues and convert to pandas DataFrame."""
    saved_data = load_leaderboard_data_from_hf()

    if not saved_data or 'wanted_issues' not in saved_data:
        print(f"No wanted issues data available")
        return pd.DataFrame(columns=["Repository", "Issue", "Title", "Age (days)", "Labels"])

    wanted_issues = saved_data['wanted_issues']
    print(f"Loaded {len(wanted_issues)} wanted issues")

    if not wanted_issues:
        return pd.DataFrame(columns=["Repository", "Issue", "Title", "Age (days)", "Labels"])

    rows = []
    for issue in wanted_issues:
        # Calculate age
        created_at = issue.get('created_at')
        age_days = 0
        if created_at and created_at != 'N/A':
            try:
                created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                age_days = (datetime.now(timezone.utc) - created).days
            except:
                pass

        # Create clickable link
        url = issue.get('url', '')
        issue_number = issue.get('number', '')
        issue_link = f'<a href="{url}" target="_blank">#{issue_number}</a>'

        rows.append([
            issue.get('repo', ''),
            issue_link,
            issue.get('title', ''),
            age_days,
            ', '.join(issue.get('labels', []))
        ])

    df = pd.DataFrame(rows, columns=["Repository", "Issue", "Title", "Age (days)", "Labels"])

    # Sort by age descending
    if "Age (days)" in df.columns and not df.empty:
        df = df.sort_values(by="Age (days)", ascending=False).reset_index(drop=True)

    return df


def validate_github_username(identifier):
    """Verify that a GitHub identifier exists."""
    try:
        response = requests.get(f'https://api.github.com/users/{identifier}', timeout=10)
        if response.status_code == 200:
            return True, "Username is valid"
        elif response.status_code == 404:
            return False, "GitHub identifier not found"
        else:
            return False, f"Validation error: HTTP {response.status_code}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def save_agent_to_hf(data):
    """Save a new agent to HuggingFace dataset."""
    try:
        api = HfApi()
        token = get_hf_token()

        if not token:
            raise Exception("No HuggingFace token found. Please set HF_TOKEN in your Space settings.")

        identifier = data['github_identifier']
        filename = f"{identifier}.json"

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        try:
            delay = 2.0
            for attempt in range(MAX_RETRIES):
                try:
                    api.upload_file(
                        path_or_fileobj=filename,
                        path_in_repo=filename,
                        repo_id=AGENTS_REPO,
                        repo_type="dataset",
                        token=token
                    )
                    print(f"Saved agent to HuggingFace: {filename}")
                    return True
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = delay + random.uniform(0, 1.0)
                        print(f"   Upload failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
                        print(f"   Retrying in {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                        delay = min(delay * 2, 60.0)
                    else:
                        raise
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    except Exception as e:
        print(f"Error saving agent: {str(e)}")
        return False


def submit_agent(identifier, agent_name, organization, website):
    """Submit a new agent to the leaderboard."""
    # Validate required fields
    if not identifier or not identifier.strip():
        return "ERROR: GitHub identifier is required", gr.update()
    if not agent_name or not agent_name.strip():
        return "ERROR: Agent name is required", gr.update()
    if not organization or not organization.strip():
        return "ERROR: Organization name is required", gr.update()
    if not website or not website.strip():
        return "ERROR: Website URL is required", gr.update()

    # Clean inputs
    identifier = identifier.strip()
    agent_name = agent_name.strip()
    organization = organization.strip()
    website = website.strip()

    # Validate GitHub identifier
    is_valid, message = validate_github_username(identifier)
    if not is_valid:
        return f"ERROR: {message}", gr.update()

    # Check for duplicates
    agents = load_agents_from_hf()
    if agents:
        existing_names = {agent['github_identifier'] for agent in agents}
        if identifier in existing_names:
            return f"WARNING: Agent with identifier '{identifier}' already exists", gr.update()

    # Create submission
    submission = {
        'name': agent_name,
        'organization': organization,
        'github_identifier': identifier,
        'website': website,
        'status': 'public'
    }

    # Save to HuggingFace
    if not save_agent_to_hf(submission):
        return "ERROR: Failed to save submission", gr.update()

    return f"SUCCESS: Successfully submitted {agent_name}! Points will be calculated during the next backend update.", gr.update()


# =============================================================================
# DATA MINING FUNCTION
# =============================================================================

def mine_all_data():
    """Mine all patch wanted issues and calculate agent points."""
    print(f"\n{'='*80}")
    print(f"[1/3] Loading agents...")
    print(f"{'='*80}\n")

    agents = load_agents_from_hf()
    if not agents:
        print("Error: No agents found")
        return

    print(f"\n{'='*80}")
    print(f"[2/3] Fetching patch wanted issues...")
    print(f"{'='*80}\n")

    issues = fetch_all_patch_wanted_issues()
    print(f"   ✓ Found {len(issues)} patch wanted issues")

    # Separate open and closed issues
    open_issues = [issue for issue in issues if issue.get('state') == 'open']
    closed_issues = [issue for issue in issues if issue.get('state') == 'closed']

    print(f"   ✓ Open: {len(open_issues)}, Closed: {len(closed_issues)}")

    # For closed issues, get closing info
    print(f"   Fetching closing information for closed issues...")
    for issue in closed_issues:
        repo_parts = issue['repo'].split('/')
        if len(repo_parts) == 2:
            org, repo_name = repo_parts
            closing_info = get_issue_closing_info(org, repo_name, issue['number'])
            if closing_info:
                issue['closed_by'] = closing_info.get('closed_by')

    print(f"\n{'='*80}")
    print(f"[3/3] Calculating points and saving...")
    print(f"{'='*80}\n")

    agent_points = calculate_agent_points(closed_issues, agents)
    leaderboard = construct_leaderboard(agent_points, agents)
    monthly_metrics = calculate_monthly_metrics_by_agent(agent_points, agents)

    # Save to HuggingFace
    save_leaderboard_data_to_hf(leaderboard, monthly_metrics, open_issues)

    print(f"   ✓ Mining complete")


# =============================================================================
# SCHEDULER FUNCTION
# =============================================================================

def reload_leaderboard_data():
    """Reload leaderboard data from HuggingFace."""
    print(f"\n{'='*80}")
    print(f"Reloading leaderboard data from HuggingFace...")
    print(f"{'='*80}\n")

    try:
        data = load_leaderboard_data_from_hf()
        if data:
            print(f"Successfully reloaded leaderboard data")
            print(f"   Last updated: {data.get('last_updated', 'Unknown')}")
            print(f"   Agents: {len(data.get('leaderboard', {}))}")
            print(f"   Wanted Issues: {len(data.get('wanted_issues', []))}")
        else:
            print(f"No data available")
    except Exception as e:
        print(f"Error reloading leaderboard data: {str(e)}")

    print(f"{'='*80}\n")


# =============================================================================
# GRADIO APPLICATION
# =============================================================================

print(f"\nStarting SWE-Wanted Leaderboard")
print(f"   Data source: {LEADERBOARD_REPO}")
print(f"   Reload frequency: Daily at 12:00 AM UTC\n")

# Start scheduler
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    reload_leaderboard_data,
    trigger=CronTrigger(hour=0, minute=0),
    id='daily_data_reload',
    name='Daily Data Reload',
    replace_existing=True
)
scheduler.start()
print(f"Scheduler initialized successfully\n")

# Create Gradio interface
with gr.Blocks(title="SWE-Wanted Leaderboard", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🏆 SWE-Wanted Leaderboard")
    gr.Markdown("Track coding agents resolving long-standing patch-wanted bugs in major open-source projects")

    with gr.Tabs():

        # Leaderboard Tab
        with gr.Tab("🏆 Leaderboard"):
            gr.Markdown("*Rankings based on resolved patch-wanted issues (1 point per resolved issue)*")
            leaderboard_table = Leaderboard(
                value=pd.DataFrame(columns=[col[0] for col in LEADERBOARD_COLUMNS]),
                datatype=LEADERBOARD_COLUMNS,
                search_columns=["Agent Name", "Website"],
                filter_columns=[
                    ColumnFilter(
                        "Total Points",
                        min=0,
                        max=1000,
                        default=[0, 1000],
                        type="slider",
                        label="Total Points"
                    )
                ]
            )

            app.load(
                fn=get_leaderboard_dataframe,
                inputs=[],
                outputs=[leaderboard_table]
            )

            gr.Markdown("---")
            gr.Markdown("### 📈 Monthly Performance - Top 5 Agents")
            gr.Markdown("*Shows cumulative and monthly points for the top performing agents*")

            monthly_metrics_plot = gr.Plot(label="Monthly Metrics")

            app.load(
                fn=lambda: create_monthly_metrics_plot(),
                inputs=[],
                outputs=[monthly_metrics_plot]
            )

        # Wanted Tab
        with gr.Tab("🔍 Wanted"):
            gr.Markdown("### Long-Standing Patch-Wanted Issues")
            gr.Markdown(f"*Issues open for {MIN_ISSUE_AGE_DAYS}+ days with patch-wanted labels from tracked repositories*")

            wanted_table = gr.Dataframe(
                value=pd.DataFrame(columns=["Repository", "Issue", "Title", "Age (days)", "Labels"]),
                datatype=["str", "html", "str", "number", "str"],
                interactive=False,
                wrap=True
            )

            app.load(
                fn=get_wanted_issues_dataframe,
                inputs=[],
                outputs=[wanted_table]
            )

        # Submit Tab
        with gr.Tab("➕ Submit Agent"):
            gr.Markdown("### Submit Your Coding Agent")
            gr.Markdown("Register your agent to start tracking bug resolution points")

            with gr.Row():
                with gr.Column():
                    github_input = gr.Textbox(
                        label="GitHub Identifier*",
                        placeholder="e.g., my-agent[bot]"
                    )
                    name_input = gr.Textbox(
                        label="Agent Name*",
                        placeholder="Your agent's display name"
                    )

                with gr.Column():
                    organization_input = gr.Textbox(
                        label="Organization*",
                        placeholder="Your organization or team name"
                    )
                    website_input = gr.Textbox(
                        label="Website*",
                        placeholder="https://your-agent-website.com"
                    )

            submit_button = gr.Button("Submit Agent", variant="primary")
            submission_status = gr.Textbox(label="Submission Status", interactive=False)

            submit_button.click(
                fn=submit_agent,
                inputs=[github_input, name_input, organization_input, website_input],
                outputs=[submission_status, leaderboard_table]
            )

# Launch application
if __name__ == "__main__":
    app.launch()
