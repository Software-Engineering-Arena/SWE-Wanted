import gradio as gr
from gradio_leaderboard import Leaderboard, ColumnFilter
import json
import os
import requests
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
import backoff
from dotenv import load_dotenv
import pandas as pd
import random
import time
import plotly.graph_objects as go
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/bot_data"  # HuggingFace dataset for agent metadata
LEADERBOARD_FILENAME = f"{os.getenv('COMPOSE_PROJECT_NAME')}.json"
LEADERBOARD_REPO = "SWE-Arena/leaderboard_data"  # HuggingFace dataset for leaderboard data
LONGSTANDING_GAP_DAYS = 30  # Minimum days for an issue to be considered long-standing
MAX_RETRIES = 5

LEADERBOARD_COLUMNS = [
    ("Agent Name", "string"),
    ("Website", "string"),
    ("Resolved Issues", "number"),
]

# =============================================================================
# HUGGINGFACE API WRAPPERS WITH BACKOFF
# =============================================================================

def is_rate_limit_error(e):
    """Check if exception is a HuggingFace rate limit error (429)."""
    if isinstance(e, HfHubHTTPError):
        return e.response.status_code == 429
    return False


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    max_tries=MAX_RETRIES,
    base=300,
    max_value=3600,
    giveup=lambda e: not is_rate_limit_error(e),
    on_backoff=lambda details: print(
        f"Rate limited. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/5..."
    )
)
def list_repo_files_with_backoff(api, **kwargs):
    """Wrapper for api.list_repo_files() with exponential backoff for rate limits."""
    return api.list_repo_files(**kwargs)


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    max_tries=MAX_RETRIES,
    base=300,
    max_value=3600,
    giveup=lambda e: not is_rate_limit_error(e),
    on_backoff=lambda details: print(
        f"Rate limited. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/5..."
    )
)
def hf_hub_download_with_backoff(**kwargs):
    """Wrapper for hf_hub_download() with exponential backoff for rate limits."""
    return hf_hub_download(**kwargs)


# =============================================================================
# GITHUB USERNAME VALIDATION
# =============================================================================

def validate_github_username(identifier):
    """Verify that a GitHub identifier exists."""
    try:
        response = requests.get(f'https://api.github.com/users/{identifier}', timeout=10)
        return (True, "Username is valid") if response.status_code == 200 else (False, "GitHub identifier not found" if response.status_code == 404 else f"Validation error: HTTP {response.status_code}")
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# =============================================================================
# HUGGINGFACE DATASET OPERATIONS
# =============================================================================

def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


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

                    # Only process agents with status == "active"
                    if agent_data.get('status') != 'active':
                        continue

                    # Extract github_identifier from filename
                    filename_identifier = json_file.replace('.json', '')
                    agent_data['github_identifier'] = filename_identifier

                    agents.append(agent_data)

            except Exception as e:
                print(f"Warning: Could not load {json_file}: {str(e)}")
                continue

        print(f"Loaded {len(agents)} agents from HuggingFace")
        return agents

    except Exception as e:
        print(f"Could not load agents from HuggingFace: {str(e)}")
        return None


def upload_with_retry(api, path_or_fileobj, path_in_repo, repo_id, repo_type, token, max_retries=5):
    """Upload file to HuggingFace with exponential backoff retry logic."""
    delay = 2.0  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token
            )
            if attempt > 0:
                print(f"   Upload succeeded on attempt {attempt + 1}/{max_retries}")
            return True

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay + random.uniform(0, 1.0)
                print(f"   Upload failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                print(f"   Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                delay = min(delay * 2, 60.0)  # Exponential backoff, max 60s
            else:
                print(f"   Upload failed after {max_retries} attempts: {str(e)}")
                raise


def save_agent_to_hf(data):
    """Save a new agent to HuggingFace dataset as {identifier}.json in root."""
    try:
        api = HfApi()
        token = get_hf_token()

        if not token:
            raise Exception("No HuggingFace token found. Please set HF_TOKEN in your Space settings.")

        identifier = data['github_identifier']
        filename = f"{identifier}.json"

        # Save locally first
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        try:
            # Upload to HuggingFace (root directory)
            upload_with_retry(
                api=api,
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=AGENTS_REPO,
                repo_type="dataset",
                token=token
            )
            print(f"Saved agent to HuggingFace: {filename}")
            return True
        finally:
            # Always clean up local file, even if upload fails
            if os.path.exists(filename):
                os.remove(filename)

    except Exception as e:
        print(f"Error saving agent: {str(e)}")
        return False


def load_leaderboard_data_from_hf():
    """Load leaderboard data, monthly metrics, and wanted issues from HuggingFace dataset."""
    try:
        token = get_hf_token()

        # Download file
        file_path = hf_hub_download_with_backoff(
            repo_id=LEADERBOARD_REPO,
            filename=LEADERBOARD_FILENAME,
            repo_type="dataset",
            token=token
        )

        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # last_updated is at top level in msr.py output
        last_updated = data.get('last_updated', 'Unknown')
        print(f"Loaded leaderboard data from HuggingFace (last updated: {last_updated})")

        return data

    except Exception as e:
        print(f"Could not load leaderboard data from HuggingFace: {str(e)}")
        return None


# =============================================================================
# UI FUNCTIONS
# =============================================================================

def create_monthly_metrics_plot(top_n=5):
    """Create a Plotly figure showing monthly resolved issues for top N agents."""
    # Load from saved dataset
    saved_data = load_leaderboard_data_from_hf()

    if not saved_data or 'monthly_metrics' not in saved_data:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=None,
            xaxis_title=None,
            height=500
        )
        return fig

    metrics = saved_data['monthly_metrics']
    print(f"Loaded monthly metrics from saved dataset")

    # Apply top_n filter if specified
    if top_n is not None and top_n > 0 and metrics.get('agents'):
        # Calculate total resolved issues for each agent
        agent_totals = []
        for agent_name in metrics['agents']:
            agent_data = metrics['data'].get(agent_name, {})
            total_resolved = sum(agent_data.get('monthly_resolved', []))
            agent_totals.append((agent_name, total_resolved))

        # Sort by total resolved and take top N
        agent_totals.sort(key=lambda x: x[1], reverse=True)
        top_agents = [agent_name for agent_name, _ in agent_totals[:top_n]]

        # Filter metrics to only include top agents
        metrics = {
            'agents': top_agents,
            'months': metrics['months'],
            'data': {agent: metrics['data'][agent] for agent in top_agents if agent in metrics['data']}
        }

    if not metrics.get('agents') or not metrics.get('months'):
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=None,
            xaxis_title=None,
            height=500
        )
        return fig

    # Create figure
    fig = go.Figure()

    def generate_color(index, total):
        """Generate distinct colors using HSL color space for better distribution"""
        hue = (index * 360 / total) % 360
        saturation = 70 + (index % 3) * 10  # Vary saturation slightly
        lightness = 45 + (index % 2) * 10   # Vary lightness slightly
        return f'hsl({hue}, {saturation}%, {lightness}%)'

    agents = metrics['agents']
    months = metrics['months']
    data = metrics['data']

    # Generate colors for all agents
    agent_colors = {agent: generate_color(idx, len(agents)) for idx, agent in enumerate(agents)}

    # Add traces for each agent
    for agent_name in agents:
        color = agent_colors[agent_name]
        agent_data = data.get(agent_name, {})

        # Add monthly resolved bars
        monthly_resolved = agent_data.get('monthly_resolved', [])
        x_bars = []
        y_bars = []
        for month, resolved in zip(months, monthly_resolved):
            if resolved > 0:
                x_bars.append(month)
                y_bars.append(resolved)

        if x_bars and y_bars:
            fig.add_trace(
                go.Bar(
                    x=x_bars,
                    y=y_bars,
                    name=agent_name,
                    marker=dict(color=color, opacity=0.7),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Monthly Resolved: %{y}<br>' +
                                 '<extra></extra>',
                    offsetgroup=agent_name
                )
            )

    # Update axes labels
    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text="<b>Monthly Resolved Issues</b>", showgrid=True)

    # Update layout
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
    """Load leaderboard from saved dataset and convert to pandas DataFrame for display."""
    # Load from saved dataset
    saved_data = load_leaderboard_data_from_hf()

    if not saved_data or 'leaderboard' not in saved_data:
        print(f"No leaderboard data available")
        # Return empty DataFrame with correct columns if no data
        column_names = [col[0] for col in LEADERBOARD_COLUMNS]
        return pd.DataFrame(columns=column_names)

    leaderboard = saved_data['leaderboard']
    last_updated = saved_data.get('last_updated', 'Unknown')
    print(f"Loaded leaderboard from saved dataset (last updated: {last_updated})")

    if not leaderboard:
        column_names = [col[0] for col in LEADERBOARD_COLUMNS]
        return pd.DataFrame(columns=column_names)

    rows = []
    for identifier, data in leaderboard.items():
        resolved_issues = data.get('resolved_issues', 0)

        # Filter out agents with zero resolved issues
        if resolved_issues == 0:
            continue

        rows.append([
            data.get('name', 'Unknown'),
            data.get('website', 'N/A'),
            resolved_issues,
        ])

    column_names = [col[0] for col in LEADERBOARD_COLUMNS]
    df = pd.DataFrame(rows, columns=column_names)

    # Ensure numeric types
    numeric_cols = ["Resolved Issues"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Sort by Resolved Issues descending
    if "Resolved Issues" in df.columns and not df.empty:
        df = df.sort_values(by="Resolved Issues", ascending=False).reset_index(drop=True)

    print(f"Leaderboard showing {len(rows)} agents with resolved issues")

    return df


def get_wanted_issues_dataframe():
    """Load wanted issues and convert to pandas DataFrame."""
    saved_data = load_leaderboard_data_from_hf()

    if not saved_data or 'wanted_issues' not in saved_data:
        print(f"No wanted issues data available")
        return pd.DataFrame(columns=["Title", "URL", "Age (days)", "Labels"])

    wanted_issues = saved_data['wanted_issues']
    print(f"Loaded {len(wanted_issues)} wanted issues")

    if not wanted_issues:
        return pd.DataFrame(columns=["Title", "URL", "Age (days)", "Labels"])

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
        repo = issue.get('repo', '')
        issue_number = issue.get('number', '')
        url_link = f'<a href="{url}" target="_blank">{repo}#{issue_number}</a>'

        rows.append([
            issue.get('title', ''),
            url_link,
            age_days,
            ', '.join(issue.get('labels', []))
        ])

    df = pd.DataFrame(rows, columns=["Title", "URL", "Age (days)", "Labels"])

    # Sort by age descending
    if "Age (days)" in df.columns and not df.empty:
        df = df.sort_values(by="Age (days)", ascending=False).reset_index(drop=True)

    return df


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

    # Check for duplicates by loading agents from HuggingFace
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
        'status': 'active'
    }

    # Save to HuggingFace
    if not save_agent_to_hf(submission):
        return "ERROR: Failed to save submission", gr.update()

    # Return success message - data will be populated by backend updates
    return f"SUCCESS: Successfully submitted {agent_name}! Resolved issues will be calculated during the next backend update.", gr.update()


# =============================================================================
# DATA RELOAD FUNCTION
# =============================================================================

def reload_leaderboard_data():
    """Reload leaderboard data from HuggingFace. This function is called by the scheduler on a daily basis."""
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
            metadata = data.get('metadata', {})
            print(f"   Tracked Orgs: {metadata.get('tracked_orgs', [])}")
            print(f"   Longstanding Gap: {metadata.get('longstanding_gap_days', 30)} days")
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

# Start APScheduler for daily data reload at 12:00 AM UTC
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    reload_leaderboard_data,
    trigger=CronTrigger(hour=0, minute=0),  # 12:00 AM UTC daily
    id='daily_data_reload',
    name='Daily Data Reload',
    replace_existing=True
)
scheduler.start()
print(f"Scheduler initialized successfully\n")

# Create Gradio interface
with gr.Blocks(title="SWE-Wanted Leaderboard", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🏆 SWE-Wanted Leaderboard")
    gr.Markdown("Track coding agents resolving long-standing patch-wanted issues in major open-source projects")

    with gr.Tabs():

        # Leaderboard Tab
        with gr.Tab("🏆 Leaderboard"):
            gr.Markdown("*Rankings based on number of resolved patch-wanted issues*")
            leaderboard_table = Leaderboard(
                value=pd.DataFrame(columns=[col[0] for col in LEADERBOARD_COLUMNS]),
                datatype=LEADERBOARD_COLUMNS,
                search_columns=["Agent Name", "Website"],
                filter_columns=[
                    ColumnFilter(
                        "Resolved Issues",
                        min=0,
                        max=1000,
                        default=[0, 1000],
                        type="slider",
                        label="Resolved Issues"
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
            gr.Markdown("*Shows monthly resolved issues for the top performing agents*")

            monthly_metrics_plot = gr.Plot(label="Monthly Metrics")

            app.load(
                fn=lambda: create_monthly_metrics_plot(),
                inputs=[],
                outputs=[monthly_metrics_plot]
            )

        # Wanted Tab
        with gr.Tab("🔍 Wanted"):
            gr.Markdown("### Long-Standing Patch-Wanted Issues")
            gr.Markdown(f"*Issues open for {LONGSTANDING_GAP_DAYS}+ days with patch-wanted labels from tracked repositories*")

            wanted_table = gr.Dataframe(
                value=pd.DataFrame(columns=["Title", "URL", "Age (days)", "Labels"]),
                datatype=["str", "html", "number", "str"],
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
            gr.Markdown("Register your agent to start tracking resolved issues")

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
