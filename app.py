#!/usr/bin/env python3
"""Gradio Space: enter a corporate name, get candidates, probabilities and a
colour-coded diff against existing Japanese G2P systems.

Claims supported: **unseen-entity performance** and **calibration**, in a form
the visitor can check themselves.

The point of this demo is not to show off cherry-picked successes; it is to get
people to **try their own company name**. Whatever a visitor types is almost
certainly an unseen entity for the model, and unseen entities are exactly the
regime Phonebook makes claims about. Confidence and rejection are shown as-is
for the same reason: the value is in knowing when the answer is wrong, so
nothing is hidden.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path

import gradio as gr

from phonebook.baselines import MeCabUniDicBaseline, PyOpenJTalkBaseline
from phonebook.runtime import load_reader, resolve_model_dir

MODEL_DIR = resolve_model_dir(os.environ.get("PHONEBOOK_MODEL"))

# Neutral, fictitious company names. Nothing here refers to a real company.
EXAMPLES = [
    "株式会社山田商店",
    "緑川食品株式会社",
    "合同会社あおぞらデザイン",
    "株式会社ABCシステム",
    "医療法人社団つばさ会",
    "有限会社みらい工務店",
    "一般社団法人地域交流推進機構",
]

CSS = """
.headline { font-size: 1.05rem; line-height: 1.7; }
.try-your-own { border: 2px solid #1a73e8; border-radius: 12px; padding: 14px 16px;
                background: linear-gradient(90deg,#eef4ff,#f7faff); margin-bottom: 8px; }
.try-your-own h3 { margin: 0 0 4px 0; color: #1a73e8; font-size: 1.15rem; }
.reading { font-size: 1.9rem; font-weight: 700; letter-spacing: .06em; }
.same { color: #1b5e20; }
.diff { background: #ffe0e0; color: #b71c1c; border-radius: 3px; padding: 0 2px; }
.ins  { background: #e0f2f1; color: #00695c; border-radius: 3px; padding: 0 2px; }
.muted { color: #666; font-size: .9rem; }
"""

_reader = None
_pyopenjtalk = PyOpenJTalkBaseline()
_mecab = MeCabUniDicBaseline()


def get_reader():
    global _reader
    if _reader is None:
        _reader = load_reader(MODEL_DIR)
    return _reader


def diff_html(reference: str, other: str) -> str:
    """Character-level diff rendered as coloured HTML.

    Phonebook's output is the reference: characters the other system agrees on
    are green, substitutions are red, insertions are teal. This makes the point
    where the two systems disagree visible at a glance.
    """
    if other is None:
        return "<span class='muted'>(unavailable)</span>"
    sm = difflib.SequenceMatcher(None, reference, other)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        seg = other[j1:j2]
        if tag == "equal":
            parts.append(f"<span class='same'>{seg}</span>")
        elif tag == "replace":
            parts.append(f"<span class='diff'>{seg}</span>")
        elif tag == "insert":
            parts.append(f"<span class='ins'>{seg}</span>")
        elif tag == "delete":
            parts.append(f"<span class='diff'>[{reference[i1:i2]}&rarr;]</span>")
    return "".join(parts)


def predict(name: str, nbest: int, threshold: float):
    if not name or not name.strip():
        return "<span class='muted'>Enter a corporate name.</span>", [], "", ""
    reader = get_reader()
    reader.threshold = threshold if threshold > 0 else None
    result = reader.read(name.strip(), nbest=int(nbest))

    top = result.display
    conf_color = (
        "#1b5e20" if result.confidence >= 0.8
        else ("#ef6c00" if result.confidence >= 0.5 else "#b71c1c")
    )
    main = [
        f"<div class='reading'>{top}</div>",
        f"<div class='muted'>confidence <b style='color:{conf_color}'>{result.confidence:.3f}</b>",
        f" &middot; path {result.source} &middot; {result.latency_ms:.1f} ms</div>",
    ]
    if result.rejected:
        main.append(
            "<div class='muted'>Below the threshold, so the answer is returned as "
            "<b>unknown</b>. The candidates below are shown for reference only.</div>"
        )
    st = result.structured
    main.append(
        f"<div class='muted'>Structure: prefix legal form = <b>{st.prefix_form or '&mdash;'}</b> / "
        f"trade-name core = <b>{st.core}</b> / suffix legal form = <b>{st.suffix_form or '&mdash;'}</b></div>"
    )

    table = [[i + 1, c.reading, round(c.prob, 4)] for i, c in enumerate(result.candidates)]

    reference = result.candidates[0].reading if result.candidates else ""
    oj = _pyopenjtalk.read(name) if _pyopenjtalk.available else None
    mc = _mecab.read(name) if _mecab.available else None
    compare = [
        "<table style='width:100%;border-collapse:collapse'>",
        "<tr><th style='text-align:left'>System</th>"
        "<th style='text-align:left'>Output (diff against Phonebook)</th></tr>",
        f"<tr><td><b>Phonebook</b></td>"
        f"<td class='reading' style='font-size:1.2rem'>{reference}</td></tr>",
        f"<tr><td>pyopenjtalk</td><td style='font-size:1.2rem'>{diff_html(reference, oj)}</td></tr>",
        f"<tr><td>MeCab+UniDic</td><td style='font-size:1.2rem'>{diff_html(reference, mc)}</td></tr>",
        "</table>",
        "<div class='muted'>green = agrees, red = differs, teal = extra. General-purpose G2P "
        "tends to diverge on proper nouns, especially unseen kanji sequences and Latin-script "
        "trade names.</div>",
    ]

    note = (
        "<div class='muted'>These readings are estimates, not official furigana. "
        "For the official furigana see the National Tax Agency Corporate Number "
        "Publication Site.</div>"
    )
    return "".join(main), table, "".join(compare), note


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Phonebook - readings for Japanese corporate names") as demo:
        # css= moved to launch() in Gradio 6; embed it so this works on any version.
        gr.HTML(f"<style>{CSS}</style>")
        gr.Markdown(
            "# Phonebook - Japanese corporate names to katakana\n"
            "<div class='headline'>A small model built for company names it has never seen. "
            "It reports calibrated confidence and answers <b>unknown</b> when it does not "
            "know.</div>",
        )
        with gr.Group(elem_classes="try-your-own"):
            gr.Markdown(
                "### Try your own company name\n"
                "Names absent from the training data are exactly what this approach targets. "
                "Use a real one."
            )
            name = gr.Textbox(
                label="Corporate name",
                placeholder="e.g. 株式会社◯◯ / ◯◯株式会社 / 一般社団法人◯◯",
                autofocus=True,
            )
            run = gr.Button("Predict the reading", variant="primary", size="lg")

        with gr.Row():
            nbest = gr.Slider(1, 5, value=3, step=1, label="Number of candidates (n-best)")
            threshold = gr.Slider(
                0.0, 1.0, value=0.0, step=0.05,
                label="Rejection threshold (answers below this become 'unknown'; 0 disables)",
            )

        out_main = gr.HTML(label="Prediction")
        out_table = gr.Dataframe(
            headers=["rank", "reading", "probability"],
            label="Candidates and probabilities",
            interactive=False,
            wrap=True,
        )
        gr.Markdown("### Comparison with general-purpose Japanese G2P")
        out_compare = gr.HTML()
        out_note = gr.HTML()

        gr.Examples(
            examples=[[e] for e in EXAMPLES],
            inputs=[name],
            label="Neutral examples (fictitious company names)",
        )

        gr.Markdown(
            "---\n"
            "Created by processing data from the National Tax Agency Corporate Number "
            "Publication Site (https://www.houjin-bangou.nta.go.jp/).\n"
            "Output is an estimate, not official furigana."
        )

        run.click(predict, [name, nbest, threshold], [out_main, out_table, out_compare, out_note])
        name.submit(predict, [name, nbest, threshold], [out_main, out_table, out_compare, out_note])
    return demo


if __name__ == "__main__":
    if not (Path(MODEL_DIR) / "model.pt").exists():
        raise SystemExit(
            f"No trained model at {MODEL_DIR}. Run scripts/train.py or set PHONEBOOK_MODEL."
        )
    build_app().launch()
