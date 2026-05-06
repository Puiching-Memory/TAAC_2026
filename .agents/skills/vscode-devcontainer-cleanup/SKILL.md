---
name: vscode-devcontainer-cleanup
description: Use when Dev Container attach is slow, VS Code Remote Containers leaves stale helper processes, /tmp/.X11-unix has many displays, /root/.vscode-server has old versions/logs, or extension caches need safe cleanup inside a dev container.
---

# VS Code Dev Container Cleanup

Use the bundled script as the implementation source of truth.

## Read First

- `.agents/skills/vscode-devcontainer-cleanup/scripts/cleanup-vscode-devcontainer.sh`

## Procedure

Inspect before cleanup:

```bash
./.agents/skills/vscode-devcontainer-cleanup/scripts/cleanup-vscode-devcontainer.sh inspect
```

Clean only after the inspection matches the user's intent:

```bash
./.agents/skills/vscode-devcontainer-cleanup/scripts/cleanup-vscode-devcontainer.sh cleanup
```

## Non-Obvious Context

- Run inside the dev container with permission to manage `/root/.vscode-server` and `/tmp/.X11-unix`.
- Preserve active VS Code sessions; trust the script's active-process checks and do not replace it with ad hoc `rm` commands.
