# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "aqora[pyarrow]==0.39.0",
#     "marimo",
#     "mqt-qcec>=3.0.0b3",
#     "polars==2.1.0",
#     "qiskit>=1.3",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", auto_download=["html", "markdown", "ipynb"])


@app.cell
def _():
    import ast
    import asyncio
    import datetime
    import hashlib
    import html
    import inspect
    import tempfile
    from pathlib import Path

    import aqora
    import marimo as mo
    from pushquantum_validator import ArchitectureError, NotEquivalentError, VerificationError, score_compiler
    from pushquantum_validator.verify import describe_violations, score_metrics

    return (
        ArchitectureError,
        NotEquivalentError,
        Path,
        VerificationError,
        aqora,
        ast,
        asyncio,
        datetime,
        describe_violations,
        hashlib,
        html,
        mo,
        score_compiler,
        tempfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # planqc compiler leaderboard

    Compile quantum circuits for a zoned neutral-atom architecture in your own workspace, then submit it here
    to be scored on the benchmark circuits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    leaderboard_refresh = mo.ui.refresh(options=["30s", "1m", "5m"], default_interval="1m")
    return (leaderboard_refresh,)


@app.cell(hide_code=True)
async def _(aqora, datetime, leaderboard_refresh, mo, submission_outcome):
    # Re-read on every refresh and after this session's own submission. A fresh KV each time:
    # the store gateway currently answers a long-lived KV's conditional re-fetch with a 500.
    _ = (leaderboard_refresh.value, submission_outcome)
    _kv = aqora.KV("planqc/leaderboard.json", aqora.Store())
    _entries = []
    for _user in await _kv.list_async():
        _entries.append((_user, await _kv.get_async(_user)))
    _entries.sort(key=lambda item: (-item[1]["score"], item[1]["submitted_at"]))
    if _entries:
        _body = mo.ui.table(
            [
                {
                    "rank": rank,
                    "user": user,
                    "score": entry["score"],
                    "workspace": entry["workspace"],
                    "submitted at": datetime.datetime.fromisoformat(entry["submitted_at"]).strftime(
                        "%Y-%m-%d %H:%M UTC"
                    ),
                }
                for rank, (user, entry) in enumerate(_entries, start=1)
            ],
            selection=None,
            format_mapping={"score": "{:+.2f}%"},
            show_column_summaries=False,
            show_data_types=False,
            page_size=25,
        )
    else:
        _body = mo.md("_No submissions yet._")
    mo.vstack(
        [mo.hstack([mo.md("## Leaderboard"), leaderboard_refresh], align="center"), _body]
    )
    return


@app.cell
def _(mo):
    mo.ui.button(label="open leaderboard")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rules

    **The problem.** Compile quantum circuits for a zoned neutral-atom architecture. Atoms start in a storage
    zone, where `rx` and `rz` run, and AOD `move` instructions carry them to an entangling zone, where `cz`
    acts on interaction pairs. Your `compile_circuit(circuit, architecture_spec)` returns the compiled circuit
    and an `Architecture` holding the initial loading of its qubits. The template notebook explains the
    architecture and the rules for moves.

    **Validity.** Every compilation must pass two checks: a replay on the architecture (legal moves, a valid
    initial loading, each gate in a zone that allows it, `cz` only on an interaction pair) and equivalence to
    the input circuit. If either fails on any benchmark, the submission is rejected.

    **The metric.** Your compiler runs on 36 benchmark circuits: hardware-efficient, IHVA and QAOA ansätze on
    12, 20 and 50 qubits. Over all of them we total the move distance (each move counts the farthest any of its
    atoms travels) and the depth of the circuits without their moves, and compare each total with that of a naive
    compiler, which carries the two atoms of every `cz` to the entangling zone and back. Your score is
    0.8 × the % reduction in move distance + 0.2 × the % reduction in depth. **Higher is better**: the naive
    compiler scores 0%.

    **How to submit.**

    1. Clone the template workspace and implement `compile_circuit` in its notebook, as a top-level function
       (`@app.function`).
    2. The scoring process imports your notebook, which runs its setup cell and defines its top-level
       functions and classes but runs none of its other cells. `compile_circuit` can only rely on those: for example, append
       `Move(...)` to your circuit instead of calling `circuit.move(...)`, which an ordinary cell registers.
       It can import marimo, qiskit, numpy and scipy, and it runs without network access.
    3. Publish a version of your workspace: this notebook scores the latest published one. Your workspace
       must be readable by `planqc`, since this notebook downloads it as `planqc`.
    4. Come back here, sign in with aqora below, enter your workspace as `username/workspace` and press
       **Submit**. Scoring takes a few minutes, and each compilation may take up to 5 minutes.

    **Only your best score counts.** Submit as often as you like: the leaderboard keeps your highest score,
    and ties go to whoever reached it first. The scoring code is at the bottom of this notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Submit

    Sign in so this notebook knows who you are, then enter your workspace. It downloads the latest published
    version of your workspace's notebook, keeps a copy of exactly what it scored, and runs your `compile_circuit` on
    every benchmark.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sign_in_button = mo.ui.run_button(label="Sign in with aqora")
    return (sign_in_button,)


@app.cell(hide_code=True)
async def _(aqora, html, mo, sign_in_button):
    viewer = None
    viewer_username = None
    _failure = None
    if sign_in_button.value:
        try:
            viewer = await aqora.viewer_login(["read:workspace"])
            viewer_username = (await viewer.send("query { viewer { username } }"))["viewer"]["username"]
        except aqora.ClientError as _error:
            viewer = None
            _failure = _error
    if viewer is not None:
        _output = mo.Html(f"Signed in as <b>{html.escape(viewer_username)}</b>.")
    elif _failure is not None:
        _output = mo.vstack(
            [
                mo.callout(
                    mo.vstack(
                        [mo.md("**Sign-in didn't complete.** Press the button to try again."), mo.plain_text(str(_failure))]
                    ),
                    kind="warn",
                ),
                sign_in_button,
            ]
        )
    else:
        _output = sign_in_button
    _output
    return viewer, viewer_username


@app.cell(hide_code=True)
def _(mo):
    workspace_input = mo.ui.text(placeholder="username/workspace", label="Your workspace")
    return (workspace_input,)


@app.cell(hide_code=True)
def _(mo, viewer):
    submit_button = mo.ui.run_button(
        label="Submit", kind="success", disabled=viewer is None, tooltip="Sign in first" if viewer is None else None
    )
    return (submit_button,)


@app.cell(hide_code=True)
async def _(
    aqora,
    judge_submission,
    mo,
    submit_button,
    viewer_username,
    workspace_input,
):
    if submit_button.value:
        with mo.status.spinner(title="Scoring your compiler", subtitle="Compiling and checking every benchmark"):
            try:
                submission_outcome = await judge_submission(viewer_username, workspace_input.value)
            except aqora.ClientError as _error:
                submission_outcome = mo.callout(
                    mo.vstack(
                        [
                            mo.md("**Something went wrong talking to aqora.** Your submission may not have been recorded; please try again."),
                            mo.plain_text(str(_error)),
                        ]
                    ),
                    kind="danger",
                )
    else:
        submission_outcome = None
    _controls = mo.hstack([workspace_input, submit_button], justify="start", align="end")
    mo.vstack([_controls, submission_outcome]) if submission_outcome is not None else _controls
    return (submission_outcome,)


@app.cell(hide_code=True)
def _(
    ArchitectureError,
    NotEquivalentError,
    Path,
    VerificationError,
    aqora,
    ast,
    asyncio,
    datetime,
    describe_violations,
    hashlib,
    mo,
    score_compiler,
    tempfile,
):
    async def judge_submission(username, workspace):
        """Download, archive and score the notebook of `workspace`, which must be `username`'s, and record it if it is their best."""

        def rejected(problems):
            return mo.callout(
                mo.vstack([mo.md("**Rejected.** Fix these problems and submit again:"), mo.plain_text(problems)]),
                kind="danger",
            )

        owner, _, slug = workspace.strip().partition("/")
        if owner.removeprefix("@") != username or not slug or "/" in slug:
            return rejected(f"enter one of your own workspaces, as {username}/<workspace>")
        workspace = f"{username}/{slug}"
        client = aqora.Client()
        if not client.authenticated:
            await client.authenticate()
        with tempfile.TemporaryDirectory() as directory:
            # The download half of `aqora.load`, which would then import the notebook here, as planqc and
            # with network. Its code only runs in the scoring process instead, without network.
            try:
                filename = await client._download_workspace_notebook(username, slug, directory, force=True)
            except aqora.ClientError as error:
                return rejected(f"couldn't download {workspace}: {error}")
            path = Path(directory, filename)
            raw = path.read_bytes()
            try:
                tree = ast.parse(raw)
            except (SyntaxError, ValueError) as error:
                return rejected(f"{workspace}'s notebook is not valid Python: {error}")
            # Defined at the top level or in the setup cell (`with app.setup:`), so importing the notebook defines it.
            setup = [node for block in tree.body if isinstance(block, ast.With) for node in block.body]
            if not any(isinstance(node, ast.FunctionDef) and node.name == "compile_circuit" for node in [*tree.body, *setup]):
                return rejected(f"{workspace} has no top-level compile_circuit function (@app.function)")
            digest = hashlib.sha256(raw).hexdigest()
            key = f"planqc/{username}/{digest}/{filename}"
            await aqora.Store().put_async(key, raw, content_type="text/x-python")
            try:
                # No network for the compiler, which could otherwise reach aqora as planqc through localhost.
                new = await asyncio.to_thread(score_compiler, f"{path}:compile_circuit", reduced=True)
            except VerificationError as error:
                # Like the validator's `--summary`: the failed benchmark and the kinds of errors, not the
                # messages, which could reveal the benchmark circuits.
                if isinstance(error, ArchitectureError):
                    reasons = describe_violations(error.violations or (error,))
                elif isinstance(error, NotEquivalentError):
                    reasons = ["not equivalent to the input circuit"]
                else:  # CompilerOutputError
                    mo.status.toast(title=str(type(error)), description=str(error))
                    reasons = ["compiler error (it raised, crashed, timed out or returned bad output)"]
                return rejected("\n".join([*getattr(error, "__notes__", []), f"failed: {'; '.join(reasons)}"]))
        kv = aqora.KV("planqc/leaderboard.json", aqora.Store())
        best = await kv.get_async(username)
        if best is not None and new <= best["score"]:
            return mo.callout(
                mo.md(f"Scored **{new:+.2f}%**, which doesn't beat your best of **{best['score']:+.2f}%**."),
                kind="info",
            )
        kv.set(
            username,
            {
                "score": new,
                "hash": digest,
                "key": key,
                "workspace": workspace,
                "submitted_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            },
        )
        await kv.flush_async()
        if best is None:
            return mo.callout(mo.md(f"**You're on the leaderboard with {new:+.2f}%.**"), kind="success")
        return mo.callout(
            mo.md(f"**New best: {new:+.2f}%**, up from {best['score']:+.2f}%."), kind="success"
        )

    return (judge_submission,)


if __name__ == "__main__":
    app.run()
