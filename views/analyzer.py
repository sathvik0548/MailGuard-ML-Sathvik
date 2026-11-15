"""Page 1 - Email Analyzer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.app_resources import get_service, theme_is_dark
from src.charts import contribution_chart, probability_bar
from src.features import FEATURE_DESCRIPTIONS
from src.inference import InferenceError, ModelUnavailableError, PredictionResult
from src.validation import ValidationError, validate_upload_filename

BODY_PREVIEW_CHARS = 4_000

service = get_service()
dark = theme_is_dark()

st.title("Email Analyzer")
st.caption(
    "Paste raw email source (headers + body), paste just the body text, or upload an `.eml` file. "
    "Content is analysed in memory only; logs record request metadata, never the email itself."
)

if not service.ready:
    st.error(f"The model could not be loaded, so analysis is disabled.\n\n{service.model_error}", icon=":material/error:")
    st.info("Generate the artifact locally with `python -m training.train`, then commit `models/`.")
    st.stop()

model = service.model
assert model is not None

if "threshold" not in st.session_state:
    st.session_state.threshold = float(model.threshold)


def _reset_threshold() -> None:
    st.session_state.threshold = float(model.threshold)


def _run(raw: str | bytes | None, source: str) -> None:
    try:
        st.session_state.last_result = service.analyze(raw, threshold=st.session_state.threshold, source=source)
        st.session_state.last_error = None
    except ValidationError as exc:
        st.session_state.last_error = ("warning", str(exc))
    except ModelUnavailableError as exc:
        st.session_state.last_error = ("error", f"Model unavailable: {exc}")
    except InferenceError as exc:
        st.session_state.last_error = ("error", f"Analysis failed unexpectedly (request id `{exc.request_id}`). It has been logged.")


with st.container(border=True):
    st.subheader("Decision threshold", icon=":material/tune:")
    threshold_col, reset_col = st.columns([3, 1], vertical_alignment="bottom")
    with threshold_col:
        st.slider(
            "Spam if probability ≥", min_value=0.0, max_value=1.0, step=0.01, key="threshold",
            help="Lower = catch more spam but risk more false positives. Moving the slider re-labels the current "
            "result instantly without sending a new request.",
        )
    with reset_col:
        st.button(
            f"Reset to tuned value ({model.threshold:.2f})",
            on_click=_reset_threshold,
            width="stretch",
        )
    st.caption("Tuned for precision-weighted F0.5 on the validation split.")

with st.container(border=True):
    st.subheader("Analyze an email", icon=":material/mail:")
    tab_paste, tab_upload, tab_sample = st.tabs(
        [":material/content_paste: Paste", ":material/upload_file: Upload .eml", ":material/science: Samples"]
    )
    with tab_paste:
        pasted = st.text_area(
            "Raw email source or body text", height=240, max_chars=config.MAX_EMAIL_BYTES,
            placeholder="From: sender@example.com\nTo: you@example.com\nSubject: Hello\n\nMessage body...",
        )
        if st.button("Analyze pasted email", type="primary", icon=":material/search:", width="stretch"):
            _run(pasted, "paste")
    with tab_upload:
        uploaded = st.file_uploader("Upload an email file", type=["eml", "txt"], accept_multiple_files=False)
        if st.button(
            "Analyze uploaded file",
            type="primary",
            icon=":material/search:",
            disabled=uploaded is None,
            width="stretch",
        ):
            try:
                validate_upload_filename(uploaded.name if uploaded else None)
                _run(uploaded.getvalue() if uploaded else None, "upload")
            except ValidationError as exc:
                st.session_state.last_error = ("warning", str(exc))
    with tab_sample:
        samples = sorted(p.name for p in config.SAMPLES_DIR.glob("*.eml"))
        choice = st.selectbox("Synthetic sample emails (fictional senders and domains)", samples)
        if choice:
            with st.expander("Preview sample source"):
                st.code((config.SAMPLES_DIR / choice).read_text(encoding="utf-8", errors="replace"), language=None)
        if st.button(
            "Analyze sample",
            type="primary",
            icon=":material/search:",
            disabled=not choice,
            width="stretch",
        ):
            _run((config.SAMPLES_DIR / choice).read_bytes(), "sample")

err = st.session_state.get("last_error")
if err:
    (st.warning if err[0] == "warning" else st.error)(err[1])

result: PredictionResult | None = st.session_state.get("last_result")
if result is None:
    st.info("Analyze an email to see the prediction, extracted features and explanation.", icon=":material/info:")
    st.stop()

st.subheader("Analysis results", icon=":material/analytics:")
threshold = float(st.session_state.threshold)
label = result.label_at(threshold)
is_spam = label == "spam"

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Verdict",
    "SPAM" if is_spam else "HAM",
    help="HAM = legitimate email",
    border=True,
    height="stretch",
)
c2.metric("Spam probability", f"{result.spam_probability:.1%}", border=True, height="stretch")
c3.metric("Threshold", f"{threshold:.2f}", border=True, height="stretch")
c4.metric("Latency", f"{result.latency_ms:.1f} ms", border=True, height="stretch")
if is_spam:
    st.error(f"Classified as **spam** (probability {result.spam_probability:.3f} ≥ {threshold:.2f}).", icon=":material/report:")
else:
    st.success(f"Classified as **ham** (probability {result.spam_probability:.3f} < {threshold:.2f}).", icon=":material/verified:")
with st.container(border=True):
    st.markdown("**Spam probability against the selected threshold**")
    st.plotly_chart(
        probability_bar(result.spam_probability, threshold, dark),
        width="stretch",
        config={"displayModeBar": False},
    )
st.caption(
    f"Request `{result.request_id}` · {result.model_name} v{result.model_version}"
    + (f" · analysed at threshold {result.threshold:.2f}; re-labelled locally at {threshold:.2f}" if abs(result.threshold - threshold) > 1e-9 else "")
)
if not result.parsed.has_headers:
    st.warning(
        "No email headers were detected, so header-based signals (Reply-To mismatch, missing Message-ID/Date) "
        "are unavailable. Paste the full raw source for a more complete analysis.", icon=":material/warning:"
    )

tab_why, tab_feat, tab_details = st.tabs(
    [":material/psychology: Why this prediction", ":material/tune: Extracted features", ":material/mail: Email details"]
)

with tab_why:
    exp = result.explanation
    st.markdown(
        f"**Method:** {exp['method']}. Bars show each signal's contribution to the model's spam score "
        f"(score {exp['score']:+.3f}, baseline {exp['base_value']:+.3f}). Positive pushes towards spam, negative towards ham."
    )
    if model.explainer == "tree":
        st.caption("TreeSHAP can credit text that is *absent* - e.g. no quoted-reply `>` lines, which are common "
                   "in legitimate replies. Such signals are marked “(absent)”.")
    if model.calibrator is not None:
        st.caption("This model's raw score is converted to a probability by Platt calibration fitted on the validation split.")
    if exp["contributions"]:
        st.plotly_chart(contribution_chart(exp["contributions"], dark), width="stretch")
        by_kind = exp.get("contribution_by_kind", {})
        if by_kind:
            k1, k2, k3 = st.columns(3)
            k1.metric("Word / bigram signals", f"{by_kind.get('word', 0.0):+.3f}")
            k2.metric("Character n-gram signals", f"{by_kind.get('char', 0.0):+.3f}")
            k3.metric("Structured signals", f"{by_kind.get('structured', 0.0):+.3f}")
        with st.expander("Contribution table"):
            st.dataframe(
                pd.DataFrame(exp["contributions"])[["feature", "kind", "contribution", "direction", "value", "description"]],
                hide_index=True, width="stretch",
            )
    else:
        st.info("No informative features were found in this email.")
    st.caption("Explanations describe how this model scored the email - not ground truth about the sender.")

with tab_feat:
    feat_df = pd.DataFrame(
        [{"feature": k, "value": v, "description": FEATURE_DESCRIPTIONS.get(k, "")} for k, v in result.features.items()]
    )
    st.dataframe(feat_df, hide_index=True, width="stretch", column_config={"value": st.column_config.NumberColumn(format="%.3f")})

with tab_details:
    p = result.parsed
    meta = pd.DataFrame(
        [
            ("Subject", p.subject),
            ("From", p.sender),
            ("Reply-To", p.reply_to or "-"),
            ("To", p.to or "-"),
            ("Date", p.date or "-"),
            ("Message-ID", p.message_id or "-"),
            ("MIME parts", ", ".join(p.content_types) or "-"),
            ("Attachments", ", ".join(p.attachments) or "none"),
            ("Received hops", str(p.received_count)),
        ],
        columns=["field", "value"],
    )
    st.dataframe(meta, hide_index=True, width="stretch")
    st.markdown(f"**URLs found ({len(p.urls)})**")
    if p.urls:
        st.dataframe(pd.DataFrame({"url": p.urls}), hide_index=True, width="stretch")
    else:
        st.caption("None")
    if p.link_pairs:
        with st.expander(f"HTML links: visible text vs. target ({len(p.link_pairs)})"):
            st.dataframe(pd.DataFrame(p.link_pairs, columns=["visible text", "href"]), hide_index=True, width="stretch")
    if p.headers:
        with st.expander(f"All headers ({len(p.headers)})"):
            st.dataframe(pd.DataFrame(p.headers, columns=["header", "value"]), hide_index=True, width="stretch")
    with st.expander("Body text preview" + (" (from HTML)" if p.html_only else "")):
        body = p.body
        st.text(body[:BODY_PREVIEW_CHARS] + ("\n... [truncated]" if len(body) > BODY_PREVIEW_CHARS else ""))
