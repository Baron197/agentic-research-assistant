# Working on this repo

Notes for Claude Code (and anyone else) making changes here. The design is in
[ARCHITECTURE.md](ARCHITECTURE.md); real mode is in [REAL_MODE.md](REAL_MODE.md);
deployments are in [DEPLOYMENT.md](DEPLOYMENT.md).

## Local setup (Windows)

- Python comes from the repo's virtual environment: `.venv/Scripts/python.exe`. There is
  no system Python, and `make` isn't available; the Makefile shows each command.
- Set `PYTHONPATH=src` for everything: tests, the CLI, the API and the UI.
- Run commands from the repo root, in Git Bash or PowerShell.

## Commands

| What | Command (with `PYTHONPATH=src`) |
|---|---|
| Lint | `python -m ruff check .` |
| Tests | `python -m pytest` |
| Eval gate (CI runs it) | `python -m eval.run_eval --min-citation-coverage 0.8` |
| CLI run | `python -m agent.runner "What is reranking?"` |
| API | `python -m uvicorn agent.api:app --port 8000` |
| UI | `python -m streamlit run ui/streamlit_app.py` |

Everything above is keyless: no API keys, no network, $0. Real mode is opt-in through `.env`.

## Rules

- **Never print values from `.env`.** Check only that a key is set.
- **Ports:** don't start test servers on 8501, which is in use on this machine. Use 86xx.
- **Two test runs.** CI's `build` job installs only `requirements.txt`, so the tests that need
  optional extras (MCP, DSPy, PDF, markdown-it) skip there, and its `extras` job runs them all.
  The README states both totals (all tests, and "passed, skipped" on a plain install). Keep
  those numbers true whenever tests change.
- **Two Streamlit versions.** The UI must work on 1.58 and on 1.65, which Streamlit Cloud and
  the Docker image install; `requirements.txt` stays below 1.66 until a newer version is
  checked. The dark theme restyles Streamlit's own HTML, so UI changes need a look in a
  browser on both versions.
- **Untrusted text is never markup.** Page titles, snippets, model claims and people's
  questions go through `escape_markdown` in reports and `plain()` in the UI.
- **`$` in UI text** turns into LaTeX in Streamlit: pass such text through `md()`.
- **Line endings are LF** (`.gitattributes`). Write generated files with `\n`.
- **Nothing private in the repo:** no server addresses, account IDs or email addresses.

## Definition of done

1. The diff is reviewed for bugs, not just style.
2. `ruff check .` passes, and both test runs (all extras, plain install) pass with the totals
   the README states.
3. The eval gate passes.
4. UI changes are checked in a browser on Streamlit 1.58 and 1.65.
5. Each new test is shown to fail when the code it guards is broken.
6. README and ARCHITECTURE describe the change.
7. After the push, CI is green.

## Commits

- Commit and push only when asked. Make one commit per concern, and each commit should
  pass the tests on its own.
- Messages follow `git log`: a plain-English subject line of about 50-70 characters, then a
  body that says why, then `- file: what changed` bullets, then a line with the test totals.
