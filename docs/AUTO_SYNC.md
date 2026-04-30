# Auto Sync After Commit (Windows)

This project includes scripts to periodically auto-sync committed code to GitHub.

## What it does

- Runs on a timer (for example, every 15 minutes).
- Only pushes when all conditions are safe:
  - Current branch matches target branch (default: `main`)
  - Working tree is clean (no uncommitted changes)
  - Local branch is ahead of remote
  - Local branch is not behind remote

## Scripts

- `scripts/git_auto_sync.ps1`: one sync cycle
- `scripts/register_git_auto_sync_task.ps1`: create/update Windows Scheduled Task

## Enable periodic auto-sync

From project root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\register_git_auto_sync_task.ps1 -IntervalMinutes 15 -Branch main -Remote origin
```

## Dry run check

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\git_auto_sync.ps1 -RepoPath . -Remote origin -Branch main -DryRun
```

## Run one real sync manually

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\git_auto_sync.ps1 -RepoPath . -Remote origin -Branch main
```

## Disable auto-sync

```powershell
schtasks /Delete /TN llm_eval_simulation-auto-sync /F
```

## Notes

- Auto-sync only pushes committed changes.
- If remote has new commits first, sync is skipped and you should pull/rebase manually.
- Keep your local git credential/login valid so scheduled pushes can authenticate.
