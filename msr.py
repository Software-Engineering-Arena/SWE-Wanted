import json
import os
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv
import duckdb
import backoff
import requests
import requests.exceptions
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
import traceback
import subprocess
import re

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/bot_data"
AGENTS_REPO_LOCAL_PATH = os.path.expanduser("~/bot_data")  # Local git clone path
DUCKDB_CACHE_FILE = "cache.duckdb"
GHARCHIVE_DATA_LOCAL_PATH = os.path.expanduser("~/gharchive/data")
LEADERBOARD_FILENAME = f"{os.getenv('COMPOSE_PROJECT_NAME')}.json"
LEADERBOARD_REPO = "SWE-Arena/leaderboard_data"
LEADERBOARD_TIME_FRAME_DAYS = 180
LONGSTANDING_GAP_DAYS = 30  # Minimum days for an issue to be considered long-standing

# Git sync configuration (mandatory to get latest bot data)
GIT_SYNC_TIMEOUT = 300  # 5 minutes timeout for git pull

# OPTIMIZED DUCKDB CONFIGURATION
DUCKDB_THREADS = 8
DUCKDB_MEMORY_LIMIT = "64GB"

# Streaming batch configuration
BATCH_SIZE_DAYS = 7  # Process 1 week at a time (~168 hourly files)

# Download configuration
DOWNLOAD_WORKERS = 4
DOWNLOAD_RETRY_DELAY = 2
MAX_RETRIES = 5

# Upload configuration
UPLOAD_DELAY_SECONDS = 5
UPLOAD_MAX_BACKOFF = 3600

# Scheduler configuration
SCHEDULE_ENABLED = True
SCHEDULE_DAY_OF_WEEK = 'sun'  # Sunday
SCHEDULE_HOUR = 0
SCHEDULE_MINUTE = 0
SCHEDULE_TIMEZONE = 'UTC'

# GitHub organizations and repositories to track
TRACKED_ORGS = [
    "apache",
    "github",
    "huggingface",
]

# Labels that indicate "patch wanted" status
PATCH_WANTED_LABELS = [
    "bug",
    "enhancement",
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_date_format(date_string):
    """Convert date strings or datetime objects to standardized ISO 8601 format with Z suffix."""
    if not date_string or date_string == 'N/A':
        return 'N/A'

    try:
        if isinstance(date_string, datetime):
            return date_string.strftime('%Y-%m-%dT%H:%M:%SZ')

        date_string = re.sub(r'\s+', ' ', date_string.strip())
        date_string = date_string.replace(' ', 'T')

        if len(date_string) >= 3:
            if date_string[-3:-2] in ('+', '-') and ':' not in date_string[-3:]:
                date_string = date_string + ':00'

        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(f"Warning: Could not parse date '{date_string}': {e}")
        return date_string


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


# =============================================================================
# GHARCHIVE DOWNLOAD FUNCTIONS
# =============================================================================

def download_file(url):
    """Download a GHArchive file with retry logic."""
    filename = url.split("/")[-1]
    filepath = os.path.join(GHARCHIVE_DATA_LOCAL_PATH, filename)

    if os.path.exists(filepath):
        return True

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(response.content)
            return True

        except requests.exceptions.HTTPError as e:
            # 404 means the file doesn't exist in GHArchive - skip without retry
            if e.response.status_code == 404:
                if attempt == 0:  # Only log once, not for each retry
                    print(f"   Warning {filename}: Not available (404) - skipping")
                return False

            # Other HTTP errors (5xx, etc.) should be retried
            wait_time = DOWNLOAD_RETRY_DELAY * (2 ** attempt)
            print(f"   Warning {filename}: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)

        except Exception as e:
            # Network errors, timeouts, etc. should be retried
            wait_time = DOWNLOAD_RETRY_DELAY * (2 ** attempt)
            print(f"   Warning {filename}: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)

    return False


def download_all_gharchive_data():
    """Download all GHArchive data files for the last LEADERBOARD_TIME_FRAME_DAYS."""
    os.makedirs(GHARCHIVE_DATA_LOCAL_PATH, exist_ok=True)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    urls = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        for hour in range(24):
            url = f"https://data.gharchive.org/{date_str}-{hour}.json.gz"
            urls.append(url)
        current_date += timedelta(days=1)

    downloads_processed = 0

    try:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            futures = [executor.submit(download_file, url) for url in urls]
            for future in as_completed(futures):
                downloads_processed += 1

        print(f"   Download complete: {downloads_processed} files")
        return True

    except Exception as e:
        print(f"Error during download: {str(e)}")
        traceback.print_exc()
        return False


# =============================================================================
# HUGGINGFACE API WRAPPERS
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
def upload_file_with_backoff(api, **kwargs):
    """Wrapper for api.upload_file() with exponential backoff."""
    return api.upload_file(**kwargs)


def get_duckdb_connection():
    """
    Initialize DuckDB connection with OPTIMIZED memory settings.
    Uses persistent database and reduced memory footprint.
    Automatically removes cache file if lock conflict is detected.
    """
    try:
        conn = duckdb.connect(DUCKDB_CACHE_FILE)
    except Exception as e:
        # Check if it's a locking error
        error_msg = str(e)
        if "lock" in error_msg.lower() or "conflicting" in error_msg.lower():
            print(f"   Warning Lock conflict detected, removing {DUCKDB_CACHE_FILE}...")
            if os.path.exists(DUCKDB_CACHE_FILE):
                os.remove(DUCKDB_CACHE_FILE)
                print(f"   Success Cache file removed, retrying connection...")
            # Retry connection after removing cache
            conn = duckdb.connect(DUCKDB_CACHE_FILE)
        else:
            # Re-raise if it's not a locking error
            raise

    # OPTIMIZED SETTINGS
    conn.execute(f"SET threads TO {DUCKDB_THREADS};")
    conn.execute("SET preserve_insertion_order = false;")
    conn.execute("SET enable_object_cache = true;")
    conn.execute("SET temp_directory = '/tmp/duckdb_temp';")
    conn.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}';")  # Per-query limit
    conn.execute(f"SET max_memory = '{DUCKDB_MEMORY_LIMIT}';")  # Hard cap

    return conn


def generate_file_path_patterns(start_date, end_date, data_dir=GHARCHIVE_DATA_LOCAL_PATH):
    """Generate file path patterns for GHArchive data in date range (only existing files)."""
    file_patterns = []
    missing_dates = set()

    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

    while current_date <= end_day:
        date_has_files = False
        for hour in range(24):
            pattern = os.path.join(data_dir, f"{current_date.strftime('%Y-%m-%d')}-{hour}.json.gz")
            if os.path.exists(pattern):
                file_patterns.append(pattern)
                date_has_files = True

        if not date_has_files:
            missing_dates.add(current_date.strftime('%Y-%m-%d'))

        current_date += timedelta(days=1)

    if missing_dates:
        print(f"   Warning Skipping {len(missing_dates)} date(s) with no data")

    return file_patterns


# =============================================================================
# AGENT LOADING
# =============================================================================

def sync_agents_repo():
    """
    Sync local bot_data repository with remote using git pull.
    This is MANDATORY to ensure we have the latest bot data.
    Raises exception if sync fails.
    """
    if not os.path.exists(AGENTS_REPO_LOCAL_PATH):
        error_msg = f"Local repository not found at {AGENTS_REPO_LOCAL_PATH}"
        print(f"   Error {error_msg}")
        print(f"   Please clone it first: git clone https://huggingface.co/datasets/{AGENTS_REPO}")
        raise FileNotFoundError(error_msg)

    if not os.path.exists(os.path.join(AGENTS_REPO_LOCAL_PATH, '.git')):
        error_msg = f"{AGENTS_REPO_LOCAL_PATH} exists but is not a git repository"
        print(f"   Error {error_msg}")
        raise ValueError(error_msg)

    try:
        # Run git pull with extended timeout due to large repository
        result = subprocess.run(
            ['git', 'pull'],
            cwd=AGENTS_REPO_LOCAL_PATH,
            capture_output=True,
            text=True,
            timeout=GIT_SYNC_TIMEOUT
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            if "Already up to date" in output or "Already up-to-date" in output:
                print(f"   Success Repository is up to date")
            else:
                print(f"   Success Repository synced successfully")
                if output:
                    # Print first few lines of output
                    lines = output.split('\n')[:5]
                    for line in lines:
                        print(f"     {line}")
            return True
        else:
            error_msg = f"Git pull failed: {result.stderr.strip()}"
            print(f"   Error {error_msg}")
            raise RuntimeError(error_msg)

    except subprocess.TimeoutExpired:
        error_msg = f"Git pull timed out after {GIT_SYNC_TIMEOUT} seconds"
        print(f"   Error {error_msg}")
        raise TimeoutError(error_msg)
    except (FileNotFoundError, ValueError, RuntimeError, TimeoutError):
        raise  # Re-raise expected exceptions
    except Exception as e:
        error_msg = f"Error syncing repository: {str(e)}"
        print(f"   Error {error_msg}")
        raise RuntimeError(error_msg) from e


def load_agents_from_hf():
    """
    Load all agent metadata JSON files from local git repository.
    ALWAYS syncs with remote first to ensure we have the latest bot data.
    """
    # MANDATORY: Sync with remote first to get latest bot data
    print(f"   Syncing bot_data repository to get latest agents...")
    sync_agents_repo()  # Will raise exception if sync fails

    agents = []

    # Scan local directory for JSON files
    if not os.path.exists(AGENTS_REPO_LOCAL_PATH):
        raise FileNotFoundError(f"Local repository not found at {AGENTS_REPO_LOCAL_PATH}")

    # Walk through the directory to find all JSON files
    files_processed = 0
    print(f"   Loading agent metadata from {AGENTS_REPO_LOCAL_PATH}...")

    for root, dirs, files in os.walk(AGENTS_REPO_LOCAL_PATH):
        # Skip .git directory
        if '.git' in root:
            continue

        for filename in files:
            if not filename.endswith('.json'):
                continue

            files_processed += 1
            file_path = os.path.join(root, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    agent_data = json.load(f)

                # Only include active agents
                if agent_data.get('status') != 'active':
                    continue

                # Extract github_identifier from filename
                github_identifier = filename.replace('.json', '')
                agent_data['github_identifier'] = github_identifier

                agents.append(agent_data)

            except Exception as e:
                print(f"   Warning Error loading {filename}: {str(e)}")
                continue

    print(f"   Success Loaded {len(agents)} active agents (from {files_processed} total files)")
    return agents


# =============================================================================
# STREAMING BATCH PROCESSING FOR ISSUES
# =============================================================================

def fetch_issue_metadata_streaming(conn, identifiers, start_date, end_date):
    """
    OPTIMIZED: Fetch issue metadata using streaming batch processing.

    Tracks issues from TRACKED_ORGS that:
    1. Have labels in PATCH_WANTED_LABELS
    2. Are linked to merged pull requests
    3. Have PRs created by agent identifiers

    For open issues: only include those open > LONGSTANDING_GAP_DAYS
    For closed issues: track closed_at and agent who resolved it

    Args:
        conn: DuckDB connection instance
        identifiers: List of GitHub usernames/bot identifiers
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)

    Returns:
        Dictionary with:
        - 'open_issues': List of long-standing open issues (wanted issues)
        - 'agent_resolved': Dict mapping agent identifier to list of resolved issue metadata
    """
    # Calculate total batches
    total_days = (end_date - start_date).days
    total_batches = (total_days // BATCH_SIZE_DAYS) + 1

    # Storage for results
    all_issues = {}  # issue_url -> issue_metadata
    issue_to_prs = defaultdict(set)  # issue_url -> set of PR URLs
    pr_creators = {}  # pr_url -> creator login
    pr_merged_at = {}  # pr_url -> merged_at timestamp

    # Process in batches
    current_date = start_date
    batch_num = 0

    print(f"   Streaming {total_batches} batches of {BATCH_SIZE_DAYS}-day intervals...")

    while current_date <= end_date:
        batch_num += 1
        batch_end = min(current_date + timedelta(days=BATCH_SIZE_DAYS - 1), end_date)

        # Get file patterns for THIS BATCH ONLY
        file_patterns = generate_file_path_patterns(current_date, batch_end)

        if not file_patterns:
            print(f"   Batch {batch_num}/{total_batches}: {current_date.date()} to {batch_end.date()} - NO DATA")
            current_date = batch_end + timedelta(days=1)
            continue

        # Progress indicator
        print(f"   Batch {batch_num}/{total_batches}: {current_date.date()} to {batch_end.date()} ({len(file_patterns)} files)... ", end="", flush=True)

        # Build file patterns SQL for THIS BATCH
        file_patterns_sql = '[' + ', '.join([f"'{fp}'" for fp in file_patterns]) + ']'

        try:
            # OPTIMIZED: Create temp view from file read (done ONCE per batch)
            # This avoids reading/decompressing the same files twice
            conn.execute(f"""
                CREATE OR REPLACE TEMP VIEW batch_data AS
                SELECT *
                FROM read_json({file_patterns_sql}, union_by_name=true, filename=true, compression='gzip', format='newline_delimited', ignore_errors=true, maximum_object_size=2147483648)
            """)

            # Query 1: Fetch all issues (NOT PRs) from IssuesEvent and IssueCommentEvent
            # Note: In GHArchive, if payload.issue.pull_request exists, it's a PR, not an issue
            # Note: We don't group by state to avoid duplicates - state is determined by closed_at
            issue_query = """
            SELECT
                json_extract_string(payload, '$.issue.html_url') as issue_url,
                json_extract_string(repo, '$.name') as repo_name,
                json_extract_string(payload, '$.issue.title') as title,
                json_extract_string(payload, '$.issue.number') as issue_number,
                MIN(json_extract_string(payload, '$.issue.created_at')) as created_at,
                MAX(json_extract_string(payload, '$.issue.closed_at')) as closed_at,
                json_extract(payload, '$.issue.labels') as labels
            FROM batch_data
            WHERE
                type IN ('IssuesEvent', 'IssueCommentEvent')
                AND json_extract_string(payload, '$.issue.pull_request') IS NULL
                AND json_extract_string(payload, '$.issue.html_url') IS NOT NULL
            GROUP BY issue_url, repo_name, title, issue_number, labels
            """

            issue_results = conn.execute(issue_query).fetchall()

            # Filter issues by org and labels
            for row in issue_results:
                issue_url = row[0]
                repo_name = row[1]  # Format: "apache/kafka"
                title = row[2]
                issue_number = row[3]
                created_at = row[4]
                closed_at = row[5]
                labels_json = row[6]

                if not issue_url or not repo_name:
                    continue

                # Extract org from repo_name
                parts = repo_name.split('/')
                if len(parts) != 2:
                    continue
                org = parts[0]

                # Filter by tracked orgs
                if org not in TRACKED_ORGS:
                    continue

                # Parse labels and store them (filtering will happen in post-processing)
                try:
                    if isinstance(labels_json, str):
                        labels_data = json.loads(labels_json)
                    else:
                        labels_data = labels_json

                    if not isinstance(labels_data, list):
                        label_names = []
                    else:
                        label_names = [label.get('name', '').lower() for label in labels_data if isinstance(label, dict)]

                except (json.JSONDecodeError, TypeError):
                    label_names = []

                # Determine state based on closed_at (if closed_at is set, issue is closed)
                normalized_closed_at = normalize_date_format(closed_at) if closed_at else None
                state = 'closed' if (normalized_closed_at and normalized_closed_at != 'N/A') else 'open'

                # Store issue metadata (no label filtering at this stage)
                all_issues[issue_url] = {
                    'url': issue_url,
                    'repo': repo_name,
                    'title': title,
                    'number': issue_number,
                    'state': state,
                    'created_at': normalize_date_format(created_at),
                    'closed_at': normalized_closed_at,
                    'labels': label_names
                }

            # Query 2: Find PRs from both IssueCommentEvent (with PR metadata) and PullRequestEvent
            # OPTIMIZED: Single scan using COALESCE to handle both event types efficiently
            pr_query = """
            SELECT DISTINCT
                COALESCE(
                    json_extract_string(payload, '$.issue.html_url'),
                    json_extract_string(payload, '$.pull_request.html_url')
                ) as pr_url,
                COALESCE(
                    json_extract_string(payload, '$.issue.user.login'),
                    json_extract_string(payload, '$.pull_request.user.login')
                ) as pr_creator,
                COALESCE(
                    json_extract_string(payload, '$.issue.pull_request.merged_at'),
                    json_extract_string(payload, '$.pull_request.merged_at')
                ) as merged_at,
                COALESCE(
                    json_extract_string(payload, '$.issue.body'),
                    json_extract_string(payload, '$.pull_request.body')
                ) as pr_body
            FROM batch_data
            WHERE
                (type = 'IssueCommentEvent' AND json_extract_string(payload, '$.issue.pull_request') IS NOT NULL)
                OR type = 'PullRequestEvent'
            """

            pr_results = conn.execute(pr_query).fetchall()

            for row in pr_results:
                pr_url = row[0]
                pr_creator = row[1]
                merged_at = row[2]
                pr_body = row[3]

                if not pr_url or not pr_creator:
                    continue

                pr_creators[pr_url] = pr_creator
                pr_merged_at[pr_url] = merged_at

                # Extract linked issues from PR body (common patterns: #123, fixes #123, closes #123, etc.)
                if pr_body:
                    # Match issue URLs or #number references
                    issue_refs = re.findall(r'(?:https?://github\.com/[\w-]+/[\w-]+/issues/\d+)|(?:#\d+)', pr_body, re.IGNORECASE)

                    for ref in issue_refs:
                        # Convert #number to full URL if needed
                        if ref.startswith('#'):
                            # Extract org/repo from PR URL
                            # Format: https://github.com/apache/kafka/pull/123
                            pr_parts = pr_url.split('/')
                            if len(pr_parts) >= 5:
                                org = pr_parts[-4]
                                repo = pr_parts[-3]
                                issue_num = ref[1:]
                                issue_url = f"https://github.com/{org}/{repo}/issues/{issue_num}"
                                issue_to_prs[issue_url].add(pr_url)
                        else:
                            issue_to_prs[ref].add(pr_url)

            print(f"Success {len(issue_results)} issues, {len(pr_results)} PRs")

            # Clean up temp view after batch processing
            conn.execute("DROP VIEW IF EXISTS batch_data")

        except Exception as e:
            print(f"\n   Error Batch {batch_num} error: {str(e)}")
            traceback.print_exc()
            # Clean up temp view even on error
            try:
                conn.execute("DROP VIEW IF EXISTS batch_data")
            except:
                pass

        # Move to next batch
        current_date = batch_end + timedelta(days=1)

    # Post-processing: Filter issues and assign to agents
    print(f"\n   Post-processing {len(all_issues)} issues...")

    open_issues = []
    agent_resolved = defaultdict(list)
    current_time = datetime.now(timezone.utc)

    for issue_url, issue_meta in all_issues.items():
        # Check if issue has linked PRs
        linked_prs = issue_to_prs.get(issue_url, set())
        if not linked_prs:
            continue

        # Check if any linked PR was merged AND created by an agent
        resolved_by = None
        for pr_url in linked_prs:
            merged_at = pr_merged_at.get(pr_url)
            if merged_at:  # PR was merged
                pr_creator = pr_creators.get(pr_url)
                if pr_creator in identifiers:
                    resolved_by = pr_creator
                    break

        if not resolved_by:
            continue

        # Process based on issue state
        if issue_meta['state'] == 'open':
            # For open issues: check if labels match PATCH_WANTED_LABELS using sub-word matching
            issue_labels = issue_meta.get('labels', [])
            has_patch_label = False
            for issue_label in issue_labels:
                for wanted_label in PATCH_WANTED_LABELS:
                    if wanted_label.lower() in issue_label:
                        has_patch_label = True
                        break
                if has_patch_label:
                    break

            if not has_patch_label:
                continue

            # Check if long-standing
            created_at_str = issue_meta.get('created_at')
            if created_at_str and created_at_str != 'N/A':
                try:
                    created_dt = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    days_open = (current_time - created_dt).days
                    if days_open >= LONGSTANDING_GAP_DAYS:
                        open_issues.append(issue_meta)
                except:
                    pass

        elif issue_meta['state'] == 'closed':
            # For closed issues with merged PRs: must be closed within time frame
            # AND must have been open for at least LONGSTANDING_GAP_DAYS before being closed
            closed_at_str = issue_meta.get('closed_at')
            created_at_str = issue_meta.get('created_at')

            if closed_at_str and closed_at_str != 'N/A' and created_at_str and created_at_str != 'N/A':
                try:
                    closed_dt = datetime.fromisoformat(closed_at_str.replace('Z', '+00:00'))
                    created_dt = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))

                    # Calculate how long the issue was open
                    days_open = (closed_dt - created_dt).days

                    # Only include if:
                    # 1. Closed within the LEADERBOARD_TIME_FRAME_DAYS
                    # 2. Was open for at least LONGSTANDING_GAP_DAYS (long-standing issue)
                    if start_date <= closed_dt <= end_date and days_open >= LONGSTANDING_GAP_DAYS:
                        agent_resolved[resolved_by].append(issue_meta)
                except:
                    pass

    print(f"   Success Found {len(open_issues)} long-standing open issues")
    print(f"   Success Found {sum(len(issues) for issues in agent_resolved.values())} resolved issues across {len(agent_resolved)} agents")

    return {
        'open_issues': open_issues,
        'agent_resolved': dict(agent_resolved)
    }


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_monthly_metrics_by_agent(agent_resolved, agents):
    """Calculate monthly metrics for all agents for visualization."""
    identifier_to_name = {agent.get('github_identifier'): agent.get('name') for agent in agents if agent.get('github_identifier')}

    if not agent_resolved:
        return {'agents': [], 'months': [], 'data': {}}

    agent_month_data = defaultdict(lambda: defaultdict(int))

    for agent_identifier, issue_list in agent_resolved.items():
        for issue_meta in issue_list:
            closed_at = issue_meta.get('closed_at')

            if not closed_at or closed_at == 'N/A':
                continue

            agent_name = identifier_to_name.get(agent_identifier, agent_identifier)

            try:
                dt = datetime.fromisoformat(closed_at.replace('Z', '+00:00'))
                month_key = f"{dt.year}-{dt.month:02d}"
                agent_month_data[agent_name][month_key] += 1
            except Exception as e:
                print(f"Warning: Could not parse date '{closed_at}': {e}")
                continue

    all_months = set()
    for agent_data in agent_month_data.values():
        all_months.update(agent_data.keys())
    months = sorted(list(all_months))

    result_data = {}
    for agent_name, month_dict in agent_month_data.items():
        monthly_resolved = [month_dict.get(month, 0) for month in months]

        result_data[agent_name] = {
            'monthly_resolved': monthly_resolved
        }

    agents_list = sorted(list(agent_month_data.keys()))

    return {
        'agents': agents_list,
        'months': months,
        'data': result_data
    }


def construct_leaderboard(agent_resolved, agents):
    """Construct leaderboard from agent resolved data."""
    if not agents:
        print("Error: No agents found")
        return {}

    leaderboard = {}

    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('name', 'Unknown')

        resolved_issues = agent_resolved.get(identifier, [])
        resolved_count = len(resolved_issues)

        leaderboard[identifier] = {
            'name': agent_name,
            'website': agent.get('website', 'N/A'),
            'github_identifier': identifier,
            'resolved_issues': resolved_count
        }

    return leaderboard


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

        combined_data = {
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'leaderboard': leaderboard_dict,
            'monthly_metrics': monthly_metrics,
            'wanted_issues': wanted_issues,
            'metadata': {
                'leaderboard_time_frame_days': LEADERBOARD_TIME_FRAME_DAYS,
                'longstanding_gap_days': LONGSTANDING_GAP_DAYS,
                'tracked_orgs': TRACKED_ORGS,
                'patch_wanted_labels': PATCH_WANTED_LABELS
            }
        }

        with open(LEADERBOARD_FILENAME, 'w') as f:
            json.dump(combined_data, f, indent=2)

        try:
            upload_file_with_backoff(
                api=api,
                path_or_fileobj=LEADERBOARD_FILENAME,
                path_in_repo=LEADERBOARD_FILENAME,
                repo_id=LEADERBOARD_REPO,
                repo_type="dataset"
            )
            return True
        finally:
            if os.path.exists(LEADERBOARD_FILENAME):
                os.remove(LEADERBOARD_FILENAME)

    except Exception as e:
        print(f"Error saving leaderboard data: {str(e)}")
        traceback.print_exc()
        return False


# =============================================================================
# MINING FUNCTION
# =============================================================================

def mine_all_data():
    """
    Mine issue metadata for all agents using STREAMING batch processing.
    Downloads GHArchive data, then uses BATCH-based DuckDB queries.
    """
    print(f"\n[1/4] Downloading GHArchive data...")

    if not download_all_gharchive_data():
        print("Warning: Download had errors, continuing with available data...")

    print(f"\n[2/4] Loading agent metadata...")

    agents = load_agents_from_hf()
    if not agents:
        print("Error: No agents found")
        return

    identifiers = [agent['github_identifier'] for agent in agents if agent.get('github_identifier')]
    if not identifiers:
        print("Error: No valid agent identifiers found")
        return

    print(f"\n[3/4] Mining issue metadata ({len(identifiers)} agents, {LEADERBOARD_TIME_FRAME_DAYS} days)...")

    try:
        conn = get_duckdb_connection()
    except Exception as e:
        print(f"Failed to initialize DuckDB connection: {str(e)}")
        return

    current_time = datetime.now(timezone.utc)
    end_date = current_time  # Include all of today up to now
    start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    try:
        # USE STREAMING FUNCTION FOR ISSUES
        results = fetch_issue_metadata_streaming(
            conn, identifiers, start_date, end_date
        )

        open_issues = results['open_issues']
        agent_resolved = results['agent_resolved']

    except Exception as e:
        print(f"Error during DuckDB fetch: {str(e)}")
        traceback.print_exc()
        return
    finally:
        conn.close()

    print(f"\n[4/4] Saving leaderboard...")

    try:
        leaderboard_dict = construct_leaderboard(agent_resolved, agents)
        monthly_metrics = calculate_monthly_metrics_by_agent(agent_resolved, agents)
        save_leaderboard_data_to_hf(leaderboard_dict, monthly_metrics, open_issues)

    except Exception as e:
        print(f"Error saving leaderboard: {str(e)}")
        traceback.print_exc()


# =============================================================================
# SCHEDULER SETUP
# =============================================================================

def setup_scheduler():
    """Set up APScheduler to run mining jobs periodically."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logging.getLogger('httpx').setLevel(logging.WARNING)

    scheduler = BlockingScheduler(timezone=SCHEDULE_TIMEZONE)

    trigger = CronTrigger(
        day_of_week=SCHEDULE_DAY_OF_WEEK,
        hour=SCHEDULE_HOUR,
        minute=SCHEDULE_MINUTE,
        timezone=SCHEDULE_TIMEZONE
    )

    scheduler.add_job(
        mine_all_data,
        trigger=trigger,
        id='mine_all_data',
        name='Mine patch wanted issues from GHArchive',
        replace_existing=True
    )

    next_run = trigger.get_next_fire_time(None, datetime.now(trigger.timezone))
    print(f"Scheduler: Weekly on {SCHEDULE_DAY_OF_WEEK} at {SCHEDULE_HOUR:02d}:{SCHEDULE_MINUTE:02d} {SCHEDULE_TIMEZONE}")
    print(f"Next run: {next_run}\n")

    print(f"\nScheduler started")
    scheduler.start()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if SCHEDULE_ENABLED:
        setup_scheduler()
    else:
        mine_all_data()
