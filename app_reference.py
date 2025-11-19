import json
import os
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv
import duckdb
import backoff
import requests
import requests.exceptions
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/bot_data"
AGENTS_REPO_LOCAL_PATH = os.path.expanduser("~/bot_data")  # Local git clone path
DUCKDB_CACHE_FILE = "cache.duckdb"
GHARCHIVE_DATA_LOCAL_PATH = os.path.expanduser("~/gharchive/data")
LEADERBOARD_REPO = "SWE-Arena/leaderboard_data"
LEADERBOARD_TIME_FRAME_DAYS = 180

# Git sync configuration (mandatory to get latest bot data)
GIT_SYNC_TIMEOUT = 300  # 5 minutes timeout for git pull

# OPTIMIZED DUCKDB CONFIGURATION
DUCKDB_THREADS = 8
DUCKDB_MEMORY_LIMIT = "64GB"

# Streaming batch configuration
BATCH_SIZE_DAYS = 7  # Process 1 week at a time (~168 hourly files)
# At this size: ~7 days × 24 files × ~100MB per file = ~16GB uncompressed per batch

# Download configuration
DOWNLOAD_WORKERS = 4
DOWNLOAD_RETRY_DELAY = 2
MAX_RETRIES = 5

# Upload configuration
UPLOAD_DELAY_SECONDS = 5
UPLOAD_MAX_BACKOFF = 3600

# Scheduler configuration
SCHEDULE_ENABLED = True
SCHEDULE_DAY_OF_WEEK = 'sat'  # Saturday
SCHEDULE_HOUR = 0
SCHEDULE_MINUTE = 0
SCHEDULE_TIMEZONE = 'UTC'

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_jsonl(filename):
    """Load JSONL file and return list of dictionaries."""
    if not os.path.exists(filename):
        return []

    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data


def save_jsonl(filename, data):
    """Save list of dictionaries to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


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
                    print(f"   ⚠ {filename}: Not available (404) - skipping")
                return False

            # Other HTTP errors (5xx, etc.) should be retried
            wait_time = DOWNLOAD_RETRY_DELAY * (2 ** attempt)
            print(f"   ⚠ {filename}: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)

        except Exception as e:
            # Network errors, timeouts, etc. should be retried
            wait_time = DOWNLOAD_RETRY_DELAY * (2 ** attempt)
            print(f"   ⚠ {filename}: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)

    return False


def download_all_gharchive_data():
    """Download all GHArchive data files for the last LEADERBOARD_TIME_FRAME_DAYS."""
    os.makedirs(GHARCHIVE_DATA_LOCAL_PATH, exist_ok=True)

    end_date = datetime.now()
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
        import traceback
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
def upload_folder_with_backoff(api, **kwargs):
    """Wrapper for api.upload_folder() with exponential backoff."""
    return api.upload_folder(**kwargs)


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
            print(f"   ⚠ Lock conflict detected, removing {DUCKDB_CACHE_FILE}...")
            if os.path.exists(DUCKDB_CACHE_FILE):
                os.remove(DUCKDB_CACHE_FILE)
                print(f"   ✓ Cache file removed, retrying connection...")
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
        print(f"   ⚠ Skipping {len(missing_dates)} date(s) with no data")

    return file_patterns


# =============================================================================
# STREAMING BATCH PROCESSING FOR DISCUSSIONS
# =============================================================================

def fetch_all_discussion_metadata_streaming(conn, identifiers, start_date, end_date):
    """
    OPTIMIZED: Fetch discussion metadata using streaming batch processing.

    Only tracks discussions assigned to the agents.

    Processes GHArchive files in BATCH_SIZE_DAYS chunks to limit memory usage.
    Instead of loading 180 days (4,344 files) at once, processes 7 days at a time.

    This prevents OOM errors by:
    1. Only keeping ~168 hourly files in memory per batch (vs 4,344)
    2. Incrementally building the results dictionary
    3. Allowing DuckDB to garbage collect after each batch

    Args:
        conn: DuckDB connection instance
        identifiers: List of GitHub usernames/bot identifiers (~1500)
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)

    Returns:
        Dictionary mapping agent identifier to list of discussion metadata
    """
    identifier_list = ', '.join([f"'{id}'" for id in identifiers])
    metadata_by_agent = defaultdict(list)

    # Calculate total batches
    total_days = (end_date - start_date).days
    total_batches = (total_days // BATCH_SIZE_DAYS) + 1

    # Process in configurable batches
    current_date = start_date
    batch_num = 0
    total_discussions = 0

    print(f"   Streaming {total_batches} batches of {BATCH_SIZE_DAYS}-day intervals...")

    while current_date <= end_date:
        batch_num += 1
        batch_end = min(current_date + timedelta(days=BATCH_SIZE_DAYS - 1), end_date)

        # Get file patterns for THIS BATCH ONLY (not all 180 days)
        file_patterns = generate_file_path_patterns(current_date, batch_end)

        if not file_patterns:
            print(f"   Batch {batch_num}/{total_batches}: {current_date.date()} to {batch_end.date()} - NO DATA")
            current_date = batch_end + timedelta(days=1)
            continue

        # Progress indicator
        print(f"   Batch {batch_num}/{total_batches}: {current_date.date()} to {batch_end.date()} ({len(file_patterns)} files)... ", end="", flush=True)

        # Build file patterns SQL for THIS BATCH
        file_patterns_sql = '[' + ', '.join([f"'{fp}'" for fp in file_patterns]) + ']'

        # Query for this batch
        # Note: Only DiscussionEvent is tracked in GHArchive (DiscussionCommentEvent is not available)
        # IMPORTANT: We only include discussions that were CREATED within the timeframe
        # Note: Discussions don't have assignees like issues do, so we only check the creator
        query = f"""
        SELECT
            json_extract_string(payload, '$.discussion.html_url') as url,
            json_extract_string(payload, '$.discussion.user.login') as agent_identifier,
            MIN(json_extract_string(payload, '$.discussion.created_at')) as created_at,
            MAX(json_extract_string(payload, '$.discussion.answer_chosen_at')) as closed_at,
            MAX(json_extract_string(payload, '$.discussion.state_reason')) as state_reason
        FROM read_json({file_patterns_sql}, union_by_name=true, filename=true, compression='gzip', format='newline_delimited', ignore_errors=true, maximum_object_size=2147483648)
        WHERE
            type = 'DiscussionEvent'
            AND json_extract_string(payload, '$.action') = 'created'
            AND json_extract(payload, '$.discussion.number') IS NOT NULL
            AND json_extract_string(payload, '$.discussion.user.login') IN ({identifier_list})
            AND json_extract_string(payload, '$.discussion.created_at') >= '{start_date.isoformat()}'
        GROUP BY url, agent_identifier
        HAVING agent_identifier IS NOT NULL AND created_at IS NOT NULL
        """

        try:
            results = conn.execute(query).fetchall()
            batch_discussions = 0

            # Add results to accumulating dictionary
            for row in results:
                url = row[0]
                agent_identifier = row[1]
                created_at = normalize_date_format(row[2]) if row[2] else None
                closed_at = normalize_date_format(row[3]) if row[3] else None
                state_reason = row[4]

                if not url or not agent_identifier:
                    continue

                discussion_metadata = {
                    'url': url,
                    'created_at': created_at,
                    'closed_at': closed_at,
                    'state_reason': state_reason,
                }

                metadata_by_agent[agent_identifier].append(discussion_metadata)
                batch_discussions += 1
                total_discussions += 1

            print(f"✓ {batch_discussions} discussions found")

        except Exception as e:
            print(f"\n   ✗ Batch {batch_num} error: {str(e)}")
            import traceback
            traceback.print_exc()

        # Move to next batch
        current_date = batch_end + timedelta(days=1)

    # Final summary
    agents_with_data = sum(1 for discussions in metadata_by_agent.values() if discussions)
    print(f"\n   ✓ Complete: {total_discussions} discussions found for {agents_with_data}/{len(identifiers)} agents")

    return dict(metadata_by_agent)


def sync_agents_repo():
    """
    Sync local bot_data repository with remote using git pull.
    This is MANDATORY to ensure we have the latest bot data.
    Raises exception if sync fails.
    """
    if not os.path.exists(AGENTS_REPO_LOCAL_PATH):
        error_msg = f"Local repository not found at {AGENTS_REPO_LOCAL_PATH}"
        print(f"   ✗ {error_msg}")
        print(f"   Please clone it first: git clone https://huggingface.co/datasets/{AGENTS_REPO}")
        raise FileNotFoundError(error_msg)

    if not os.path.exists(os.path.join(AGENTS_REPO_LOCAL_PATH, '.git')):
        error_msg = f"{AGENTS_REPO_LOCAL_PATH} exists but is not a git repository"
        print(f"   ✗ {error_msg}")
        raise ValueError(error_msg)

    try:
        import subprocess

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
                print(f"   ✓ Repository is up to date")
            else:
                print(f"   ✓ Repository synced successfully")
                if output:
                    # Print first few lines of output
                    lines = output.split('\n')[:5]
                    for line in lines:
                        print(f"     {line}")
            return True
        else:
            error_msg = f"Git pull failed: {result.stderr.strip()}"
            print(f"   ✗ {error_msg}")
            raise RuntimeError(error_msg)

    except subprocess.TimeoutExpired:
        error_msg = f"Git pull timed out after {GIT_SYNC_TIMEOUT} seconds"
        print(f"   ✗ {error_msg}")
        raise TimeoutError(error_msg)
    except (FileNotFoundError, ValueError, RuntimeError, TimeoutError):
        raise  # Re-raise expected exceptions
    except Exception as e:
        error_msg = f"Error syncing repository: {str(e)}"
        print(f"   ✗ {error_msg}")
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

                # Only include public agents
                if agent_data.get('status') != 'public':
                    continue

                # Extract github_identifier from filename
                github_identifier = filename.replace('.json', '')
                agent_data['github_identifier'] = github_identifier

                agents.append(agent_data)

            except Exception as e:
                print(f"   ⚠ Error loading {filename}: {str(e)}")
                continue

    print(f"   ✓ Loaded {len(agents)} public agents (from {files_processed} total files)")
    return agents


def calculate_discussion_stats_from_metadata(metadata_list):
    """Calculate statistics from a list of discussion metadata."""
    total_discussions = len(metadata_list)
    closed = sum(1 for discussion_meta in metadata_list if discussion_meta.get('closed_at'))
    resolved = sum(1 for discussion_meta in metadata_list
                   if discussion_meta.get('state_reason') == 'completed')

    # Resolved rate = resolved / closed (not resolved / total)
    resolved_rate = (resolved / closed * 100) if closed > 0 else 0

    return {
        'total_discussions': total_discussions,
        'closed_discussions': closed,
        'resolved_discussions': resolved,
        'resolved_rate': round(resolved_rate, 2),
    }


def calculate_monthly_metrics_by_agent(all_metadata_dict, agents):
    """Calculate monthly metrics for all agents for visualization."""
    identifier_to_name = {agent.get('github_identifier'): agent.get('name') for agent in agents if agent.get('github_identifier')}

    if not all_metadata_dict:
        return {'agents': [], 'months': [], 'data': {}}

    agent_month_data = defaultdict(lambda: defaultdict(list))

    for agent_identifier, metadata_list in all_metadata_dict.items():
        for discussion_meta in metadata_list:
            created_at = discussion_meta.get('created_at')

            if not created_at:
                continue

            agent_name = identifier_to_name.get(agent_identifier, agent_identifier)

            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                month_key = f"{dt.year}-{dt.month:02d}"
                agent_month_data[agent_name][month_key].append(discussion_meta)
            except Exception as e:
                print(f"Warning: Could not parse date '{created_at}': {e}")
                continue

    all_months = set()
    for agent_data in agent_month_data.values():
        all_months.update(agent_data.keys())
    months = sorted(list(all_months))

    result_data = {}
    for agent_name, month_dict in agent_month_data.items():
        resolved_rates = []
        total_discussions_list = []
        resolved_discussions_list = []
        closed_discussions_list = []

        for month in months:
            discussions_in_month = month_dict.get(month, [])

            resolved_count = sum(1 for discussion in discussions_in_month if discussion.get('state_reason') == 'completed')
            closed_count = sum(1 for discussion in discussions_in_month if discussion.get('closed_at'))
            total_count = len(discussions_in_month)

            # Resolved rate = resolved / closed (not resolved / total)
            resolved_rate = (resolved_count / closed_count * 100) if closed_count > 0 else None

            resolved_rates.append(resolved_rate)
            total_discussions_list.append(total_count)
            resolved_discussions_list.append(resolved_count)
            closed_discussions_list.append(closed_count)

        result_data[agent_name] = {
            'resolved_rates': resolved_rates,
            'total_discussions': total_discussions_list,
            'resolved_discussions': resolved_discussions_list,
            'closed_discussions': closed_discussions_list
        }

    agents_list = sorted(list(agent_month_data.keys()))

    return {
        'agents': agents_list,
        'months': months,
        'data': result_data
    }


def construct_leaderboard_from_metadata(all_metadata_dict, agents):
    """Construct leaderboard from in-memory discussion metadata."""
    if not agents:
        print("Error: No agents found")
        return {}

    cache_dict = {}

    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('name', 'Unknown')

        bot_metadata = all_metadata_dict.get(identifier, [])
        stats = calculate_discussion_stats_from_metadata(bot_metadata)

        cache_dict[identifier] = {
            'name': agent_name,
            'website': agent.get('website', 'N/A'),
            'github_identifier': identifier,
            **stats
        }

    return cache_dict


def save_leaderboard_data_to_hf(leaderboard_dict, monthly_metrics):
    """Save leaderboard data and monthly metrics to HuggingFace dataset."""
    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi(token=token)
        filename = "swe-discussion.json"

        combined_data = {
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'leaderboard': leaderboard_dict,
            'monthly_metrics': monthly_metrics,
            'metadata': {
                'leaderboard_time_frame_days': LEADERBOARD_TIME_FRAME_DAYS
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
            return True
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    except Exception as e:
        print(f"Error saving leaderboard data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# MINING FUNCTION
# =============================================================================

def mine_all_agents():
    """
    Mine discussion metadata for all agents using STREAMING batch processing.
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

    print(f"\n[3/4] Mining discussion metadata ({len(identifiers)} agents, {LEADERBOARD_TIME_FRAME_DAYS} days)...")

    try:
        conn = get_duckdb_connection()
    except Exception as e:
        print(f"Failed to initialize DuckDB connection: {str(e)}")
        return

    current_time = datetime.now(timezone.utc)
    end_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    try:
        # USE STREAMING FUNCTION FOR DISCUSSIONS
        all_metadata = fetch_all_discussion_metadata_streaming(
            conn, identifiers, start_date, end_date
        )

    except Exception as e:
        print(f"Error during DuckDB fetch: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    finally:
        conn.close()

    print(f"\n[4/4] Saving leaderboard...")

    try:
        leaderboard_dict = construct_leaderboard_from_metadata(all_metadata, agents)
        monthly_metrics = calculate_monthly_metrics_by_agent(all_metadata, agents)
        save_leaderboard_data_to_hf(leaderboard_dict, monthly_metrics)

    except Exception as e:
        print(f"Error saving leaderboard: {str(e)}")
        import traceback
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
        mine_all_agents,
        trigger=trigger,
        id='mine_all_agents',
        name='Mine GHArchive data for all agents',
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
        mine_all_agents()

import gradio as gr
from gradio_leaderboard import Leaderboard, ColumnFilter
import json
import os
import time
import requests
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
import backoff
from dotenv import load_dotenv
import pandas as pd
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/bot_metadata"  # HuggingFace dataset for agent metadata
LEADERBOARD_REPO = "SWE-Arena/leaderboard_metadata"  # HuggingFace dataset for leaderboard data
MAX_RETRIES = 5

LEADERBOARD_COLUMNS = [
    ("Agent Name", "string"),
    ("Website", "string"),
    ("Total Discussions", "number"),
    ("Resolved Discussions", "number"),
    ("Resolved Rate (%)", "number"),
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

                    # Extract github_identifier from filename (e.g., "agent[bot].json" -> "agent[bot]")
                    filename_identifier = json_file.replace('.json', '')

                    # Add or override github_identifier to match filename
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


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


def upload_with_retry(api, path_or_fileobj, path_in_repo, repo_id, repo_type, token, max_retries=5):
    """
    Upload file to HuggingFace with exponential backoff retry logic.

    Args:
        api: HfApi instance
        path_or_fileobj: Local file path to upload
        path_in_repo: Target path in the repository
        repo_id: Repository ID
        repo_type: Type of repository (e.g., "dataset")
        token: HuggingFace token
        max_retries: Maximum number of retry attempts

    Returns:
        True if upload succeeded, raises exception if all retries failed
    """
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
    """
    Load leaderboard data and monthly metrics from HuggingFace dataset.

    Returns:
        dict: Dictionary with 'leaderboard', 'monthly_metrics', and 'metadata' keys
              Returns None if file doesn't exist or error occurs
    """
    try:
        token = get_hf_token()
        filename = "swe-discussion.json"

        # Download file
        file_path = hf_hub_download_with_backoff(
            repo_id=LEADERBOARD_REPO,
            filename=filename,
            repo_type="dataset",
            token=token
        )

        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        last_updated = data.get('metadata', {}).get('last_updated', 'Unknown')
        print(f"Loaded leaderboard data from HuggingFace (last updated: {last_updated})")

        return data

    except Exception as e:
        print(f"Could not load leaderboard data from HuggingFace: {str(e)}")
        return None


# =============================================================================
# UI FUNCTIONS
# =============================================================================

def create_monthly_metrics_plot(top_n=5):
    """
    Create a Plotly figure with dual y-axes showing:
    - Left y-axis: Resolved Rate (%) as line curves
    - Right y-axis: Total Discussions created as bar charts

    Each agent gets a unique color for both their line and bars.

    Args:
        top_n: Number of top agents to show (default: 5)
    """
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
        # Calculate total discussions for each agent
        agent_totals = []
        for agent_name in metrics['agents']:
            agent_data = metrics['data'].get(agent_name, {})
            total_discussions = sum(agent_data.get('total_discussions', []))
            agent_totals.append((agent_name, total_discussions))

        # Sort by total discussions and take top N
        agent_totals.sort(key=lambda x: x[1], reverse=True)
        top_agents = [agent_name for agent_name, _ in agent_totals[:top_n]]

        # Filter metrics to only include top agents
        metrics = {
            'agents': top_agents,
            'months': metrics['months'],
            'data': {agent: metrics['data'][agent] for agent in top_agents if agent in metrics['data']}
        }

    if not metrics['agents'] or not metrics['months']:
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

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Generate unique colors for many agents using HSL color space
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
    for idx, agent_name in enumerate(agents):
        color = agent_colors[agent_name]
        agent_data = data[agent_name]

        # Add line trace for resolved rate (left y-axis)
        resolved_rates = agent_data['resolved_rates']
        # Filter out None values for plotting
        x_resolved = [month for month, rate in zip(months, resolved_rates) if rate is not None]
        y_resolved = [rate for rate in resolved_rates if rate is not None]

        if x_resolved and y_resolved:  # Only add trace if there's data
            fig.add_trace(
                go.Scatter(
                    x=x_resolved,
                    y=y_resolved,
                    name=agent_name,
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    legendgroup=agent_name,
                    showlegend=(top_n is not None and top_n <= 10),  # Show legend for top N agents
                    hovertemplate='<b>Agent: %{fullData.name}</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Resolved Rate: %{y:.2f}%<br>' +
                                 '<extra></extra>'
                ),
                secondary_y=False
            )

        # Add bar trace for total discussions (right y-axis)
        # Only show bars for months where agent has discussions
        x_bars = []
        y_bars = []
        for month, count in zip(months, agent_data['total_discussions']):
            if count > 0:  # Only include months with discussions
                x_bars.append(month)
                y_bars.append(count)

        if x_bars and y_bars:  # Only add trace if there's data
            fig.add_trace(
                go.Bar(
                    x=x_bars,
                    y=y_bars,
                    name=agent_name,
                    marker=dict(color=color, opacity=0.6),
                    legendgroup=agent_name,
                    showlegend=False,  # Hide duplicate legend entry (already shown in Scatter)
                    hovertemplate='<b>Agent: %{fullData.name}</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Total Discussions: %{y}<br>' +
                                 '<extra></extra>',
                    offsetgroup=agent_name  # Group bars by agent for proper spacing
                ),
                secondary_y=True
            )

    # Update axes labels
    fig.update_xaxes(title_text=None)
    fig.update_yaxes(
        title_text="<b>Resolved Rate (%)</b>",
        range=[0, 100],
        secondary_y=False,
        showticklabels=True,
        tickmode='linear',
        dtick=10,
        showgrid=True
    )
    fig.update_yaxes(title_text="<b>Total Discussions</b>", secondary_y=True)

    # Update layout
    show_legend = (top_n is not None and top_n <= 10)
    fig.update_layout(
        title=None,
        hovermode='closest',  # Show individual agent info on hover
        barmode='group',
        height=600,
        showlegend=show_legend,
        margin=dict(l=50, r=150 if show_legend else 50, t=50, b=50)  # More right margin when legend is shown
    )

    return fig


def get_leaderboard_dataframe():
    """
    Load leaderboard from saved dataset and convert to pandas DataFrame for display.
    Returns formatted DataFrame sorted by total discussions.
    """
    # Load from saved dataset
    saved_data = load_leaderboard_data_from_hf()

    if not saved_data or 'leaderboard' not in saved_data:
        print(f"No leaderboard data available")
        # Return empty DataFrame with correct columns if no data
        column_names = [col[0] for col in LEADERBOARD_COLUMNS]
        return pd.DataFrame(columns=column_names)

    cache_dict = saved_data['leaderboard']
    last_updated = saved_data.get('metadata', {}).get('last_updated', 'Unknown')
    print(f"Loaded leaderboard from saved dataset (last updated: {last_updated})")
    print(f"Cache dict size: {len(cache_dict)}")

    if not cache_dict:
        print("WARNING: cache_dict is empty!")
        # Return empty DataFrame with correct columns if no data
        column_names = [col[0] for col in LEADERBOARD_COLUMNS]
        return pd.DataFrame(columns=column_names)

    rows = []
    filtered_count = 0
    for identifier, data in cache_dict.items():
        total_discussions = data.get('total_discussions', 0)
        print(f"   Agent '{identifier}': {total_discussions} discussions")

        # Filter out agents with zero total discussions
        if total_discussions == 0:
            filtered_count += 1
            continue

        # Only include display-relevant fields
        rows.append([
            data.get('name', 'Unknown'),
            data.get('website', 'N/A'),
            total_discussions,
            data.get('resolved_discussions', 0),
            data.get('resolved_rate', 0.0),
        ])

    print(f"Filtered out {filtered_count} agents with 0 discussions")
    print(f"Leaderboard will show {len(rows)} agents")

    # Create DataFrame
    column_names = [col[0] for col in LEADERBOARD_COLUMNS]
    df = pd.DataFrame(rows, columns=column_names)

    # Ensure numeric types
    numeric_cols = ["Total Discussions", "Resolved Discussions", "Resolved Rate (%)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Sort by Total Discussions descending
    if "Total Discussions" in df.columns and not df.empty:
        df = df.sort_values(by="Total Discussions", ascending=False).reset_index(drop=True)

    print(f"Final DataFrame shape: {df.shape}")
    print("="*60 + "\n")

    return df


def submit_agent(identifier, agent_name, organization, website):
    """
    Submit a new agent to the leaderboard.
    Validates input and saves submission.
    """
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
        'status': 'public'
    }

    # Save to HuggingFace
    if not save_agent_to_hf(submission):
        return "ERROR: Failed to save submission", gr.update()

    # Return success message - data will be populated by backend updates
    return f"SUCCESS: Successfully submitted {agent_name}! Discussion data will be automatically populated by the backend system via the maintainers.", gr.update()


# =============================================================================
# DATA RELOAD FUNCTION
# =============================================================================

def reload_leaderboard_data():
    """
    Reload leaderboard data from HuggingFace.
    This function is called by the scheduler on a daily basis.
    """
    print(f"\n{'='*80}")
    print(f"Reloading leaderboard data from HuggingFace...")
    print(f"{'='*80}\n")

    try:
        data = load_leaderboard_data_from_hf()
        if data:
            print(f"Successfully reloaded leaderboard data")
            print(f"   Last updated: {data.get('metadata', {}).get('last_updated', 'Unknown')}")
            print(f"   Agents: {len(data.get('leaderboard', {}))}")
        else:
            print(f"No data available")
    except Exception as e:
        print(f"Error reloading leaderboard data: {str(e)}")

    print(f"{'='*80}\n")


# =============================================================================
# GRADIO APPLICATION
# =============================================================================

print(f"\nStarting SWE Agent Discussion Leaderboard")
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
print(f"\n{'='*80}")
print(f"Scheduler initialized successfully")
print(f"Reload schedule: Daily at 12:00 AM UTC")
print(f"On startup: Loads cached data from HuggingFace on demand")
print(f"{'='*80}\n")

# Create Gradio interface
with gr.Blocks(title="SWE Agent Discussion Leaderboard", theme=gr.themes.Soft()) as app:
    gr.Markdown("# SWE Agent Discussion Leaderboard")
    gr.Markdown(f"Track and compare GitHub discussion resolution statistics for SWE agents")

    with gr.Tabs():

        # Leaderboard Tab
        with gr.Tab("Leaderboard"):
            gr.Markdown("*Statistics are based on agent discussion resolution activity tracked by the system*")
            leaderboard_table = Leaderboard(
                value=pd.DataFrame(columns=[col[0] for col in LEADERBOARD_COLUMNS]),  # Empty initially
                datatype=LEADERBOARD_COLUMNS,
                search_columns=["Agent Name", "Website"],
                filter_columns=[
                    ColumnFilter(
                        "Resolved Rate (%)",
                        min=0,
                        max=100,
                        default=[0, 100],
                        type="slider",
                        label="Resolved Rate (%)"
                    )
                ]
            )

            # Load leaderboard data when app starts
            app.load(
                fn=get_leaderboard_dataframe,
                inputs=[],
                outputs=[leaderboard_table]
            )

            # Monthly Metrics Section
            gr.Markdown("---")  # Divider
            gr.Markdown("### Monthly Performance - Top 5 Agents")
            gr.Markdown("*Shows resolved rate trends and discussion volumes for the most active agents*")

            monthly_metrics_plot = gr.Plot(label="Monthly Metrics")

            # Load monthly metrics when app starts
            app.load(
                fn=lambda: create_monthly_metrics_plot(),
                inputs=[],
                outputs=[monthly_metrics_plot]
            )


        # Submit Agent Tab
        with gr.Tab("Submit Agent"):

            gr.Markdown("### Submit Your Agent")
            gr.Markdown("Fill in the details below to add your agent to the leaderboard.")

            with gr.Row():
                with gr.Column():
                    github_input = gr.Textbox(
                        label="GitHub Identifier*",
                        placeholder="Your agent username (e.g., my-agent[bot])"
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

            submit_button = gr.Button(
                "Submit Agent",
                variant="primary"
            )
            submission_status = gr.Textbox(
                label="Submission Status",
                interactive=False
            )

            # Event handler
            submit_button.click(
                fn=submit_agent,
                inputs=[github_input, name_input, organization_input, website_input],
                outputs=[submission_status, leaderboard_table]
            )


# Launch application
if __name__ == "__main__":
    app.launch()
