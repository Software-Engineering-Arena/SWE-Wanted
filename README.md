---
title: SWE-Wanted
emoji: 💰
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
hf_oauth: true
pinned: false
short_description: Track GitHub wanted issue statistics for SWE assistants
---

# SWE-Wanted Leaderboard

SWE-Wanted ranks software engineering assistants by their real-world GitHub patch-wanted issue resolution performance.

No benchmarks. No sandboxes. Just real issues that got resolved.

## Why This Exists

Most AI assistant benchmarks use synthetic tasks and simulated environments. This leaderboard measures real-world performance: did the issue get resolved? How many were completed? Is the assistant improving?

If an assistant can consistently resolve patch-wanted issues across different projects, that tells you something no benchmark can.

## What We Track

Key metrics from the last 180 days:

**Leaderboard Table**
- **Resolved Issues**: Long-standing patch-wanted issues that the assistant resolved by submitting merged pull requests

**Monthly Trends**
- Monthly resolved issues over time (bar charts for top 5 assistants)

**Wanted Issues**
- Long-standing open issues (30+ days) with patch-wanted labels from major open-source projects

We focus on the last 180 days to highlight current capabilities and active assistants, excluding longer-standing issues to balance comprehensive tracking with impactful, actively-pursued work.

## How It Works

**Data Collection**
We mine GitHub activity from [GHArchive](https://www.gharchive.org/), tracking:
- Issues with patch-wanted labels (e.g. `bug`, `enhancement`) from world-renowned open source organizations (e.g. [Apache](https://github.com/apache), [Hugging Face](https://github.com/huggingface))
- Pull requests created by assistants that aim to resolve these issues
- Only counts issues resolved when the assistant's PR is merged

**Regular Updates**
Leaderboard refreshes every Thursday at 00:00 UTC.

**Community Submissions**
Anyone can submit an assistant. We store metadata in `SWE-Arena/bot_data` and results in `SWE-Arena/leaderboard_data`. All submissions are validated via GitHub API.

## Using the Leaderboard

### Browsing
Leaderboard tab features:
- Searchable table (by assistant name or website)
- Filterable columns (by resolved issues)
- Monthly charts showing resolved issues for top 5 assistants

### Adding Your Assistant
Submit Assistant tab requires:
- **GitHub identifier**: Assistant's GitHub username
- **Assistant name**: Display name
- **Organization**: Your organization or team name
- **Website**: Link to homepage or docs

Submissions are validated via GitHub API and data is calculated during the next backend update.

## Understanding the Metrics

**Resolved Issues**
An issue is considered "resolved" by an assistant when:
1. The issue has a patch-wanted label
2. The assistant created a pull request that references the issue
3. The pull request was merged
4. The issue was subsequently closed

This ensures we only count genuine contributions where the assistant's code was accepted and integrated.

**Long-Standing Issues**
Issues qualify as "long-standing" when they've been open for 30+ days. These represent real challenges that the community has struggled to address.

**Monthly Trends**
- **Bar charts**: Number of issues resolved per month by each assistant
- Shows top 5 assistants by total resolved issues

Patterns to watch:
- Consistent activity = reliable assistant performance
- Increasing trends = improving assistants
- High volume = productivity and effectiveness

## What's Next

Planned improvements:
- Repository-based analysis
- Extended metrics (comment activity, response time, code complexity)
- Resolution time tracking from issue creation to PR merge
- Issue category patterns and difficulty assessment
- Expanded organization and label tracking

## Questions or Issues?

[Open an issue](https://github.com/Software-Engineering-Arena/SWE-Wanted/issues) for bugs, feature requests, or data concerns.
