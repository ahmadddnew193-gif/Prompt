import os
import time

import pandas as pd
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spookluke_ai_videos.csv")

# =========================================================
# Page setup
# =========================================================
st.set_page_config(page_title="Spookluke RL Coach", page_icon="⚪️", layout="wide")
st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 10px; }
    h1 { background: linear-gradient(90deg, #0AAFFF, #FF8C1A);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# systemprompt
PERSONA_AND_SCOPE_PROMPT = """You are an AI Rocket League coach built specifically on the YouTube coaching
content of Spookluke (spookyluke),Your creator is HMDSZN,spookluke a Rocket League creator and coach.

WHO YOU ARE:
- Your coaching voice mirrors Spookluke's: direct, energetic, and practical.
  You break mechanics down into clear, numbered, learnable steps rather than
  vague tips.
- You talk in terms of Rocket League ranks (Bronze through Supersonic Legend)
  and tailor advice to where the player says they're stuck.
- You favor fundamentals and consistency for lower ranks, and push technical
  mechanics (air dribbles, flip resets, ceiling shots, etc.) once someone has
  the basics locked in — the same progression his videos teach.
- You are an AI assistant modeled on his publicly published coaching videos,
  not the real person. If someone directly asks whether you ARE Spookluke,
  say plainly you're an AI coach built from his public YouTube content, not
  him.

STRICT SCOPE RULES — follow these no matter what:
- ONLY talk about Rocket League, and ONLY in a coaching/helping capacity.
  Every response should move the player toward improving at Rocket League.
- Ground your answers in the dataset provided below (video titles + transcript
  excerpts) whenever it's relevant. When you draw on a specific video, say so
  by name, e.g. "like Spookluke covers in '7 Mistakes All Low Rank Players
  Make'".
- If a question falls outside what's in the dataset, you can still answer
  using solid general Rocket League coaching knowledge in the same style —
  but never drift into topics unrelated to Rocket League coaching.
- If asked about anything outside Rocket League entirely (other games,
  unrelated topics, general chit-chat, etc.), politely decline and steer the
  conversation back to Rocket League.
- Don't recite promotional links, sponsor reads, or ad copy from the source
  videos unless the user specifically asks about his coaching/Discord/socials.
"""


# Data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(columns=[
        "video_id", "title", "url", "upload_date", "duration_seconds",
        "view_count", "like_count", "description", "transcript_status", "transcript",
    ])


def has_transcript(row):
    t = row.get("transcript")
    return isinstance(t, str) and t.strip() != ""


def chunk_text(text, chunk_size=1200, overlap=150):
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_chunk_index(df):
    records = []
    for _, row in df.iterrows():
        if not has_transcript(row):
            continue
        title = str(row["title"])
        for chunk in chunk_text(str(row["transcript"])):
            records.append({
                "video_id": row["video_id"],
                "title": title,
                "url": row.get("url", f"https://www.youtube.com/watch?v={row['video_id']}"),
                "text": chunk,
                # title repeated to weight it in retrieval matching, without
                # polluting what actually gets shown to the model as context
                "search_text": (title + " ") * 3 + chunk,
            })
    return records


def retrieve_relevant_chunks(query, chunk_records, top_k=6, max_per_video=2):
    if not chunk_records or not query.strip():
        return []
    texts = [r["search_text"] for r in chunk_records]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_matrix = vectorizer.fit_transform(texts + [query])
    sims = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    ranked = sims.argsort()[::-1]

    selected, per_video = [], {}
    for idx in ranked:
        if sims[idx] <= 0:
            break
        rec = chunk_records[idx]
        vid = rec["video_id"]
        if per_video.get(vid, 0) >= max_per_video:
            continue
        selected.append(rec)
        per_video[vid] = per_video.get(vid, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def build_catalog(df):
    lines = []
    for _, row in df.iterrows():
        mark = "✓" if has_transcript(row) else "✗"
        lines.append(f'- [{mark}] "{row["title"]}" — {row.get("url", "")}')
    return "\n".join(lines)


def build_system_prompt(df, chunk_records, user_query):
    relevant = retrieve_relevant_chunks(user_query, chunk_records)
    if relevant:
        context_block = "\n\n---\n\n".join(
            f'From "{r["title"]}" ({r["url"]}):\n{r["text"]}' for r in relevant
        )
    else:
        context_block = (
            "(No closely matching transcript excerpt for this question — answer "
            "using solid Rocket League coaching knowledge in Spookluke's style.)"
        )
    catalog = build_catalog(df)
    return (
        f"{PERSONA_AND_SCOPE_PROMPT}\n\n"
        f"VIDEO LIBRARY (✓ = transcript available):\n{catalog}\n\n"
        f"RELEVANT TRANSCRIPT EXCERPTS FOR THIS QUESTION:\n{context_block}\n"
    )


def fetch_missing_transcripts(df, progress_cb=None):
    from youtube_transcript_api import YouTubeTranscriptApi

    df = df.copy()
    missing_idx = [i for i, row in df.iterrows() if not has_transcript(row)]
    ytt_api = YouTubeTranscriptApi()

    for i, idx in enumerate(missing_idx):
        video_id = df.loc[idx, "video_id"]
        try:
            fetched = ytt_api.fetch(video_id)
            df.loc[idx, "transcript"] = " ".join(s.text for s in fetched)
            df.loc[idx, "transcript_status"] = "ok"
        except Exception as e:
            df.loc[idx, "transcript_status"] = f"failed: {type(e).__name__}"
        if progress_cb:
            progress_cb(i + 1, len(missing_idx), video_id)
        time.sleep(1.2)  
    return df



# sidebar
if "df" not in st.session_state:
    st.session_state.df = load_data()

with st.sidebar:
    st.header("Garage Settings")

    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=os.environ.get("OPENROUTER_API_KEY", ""),
        help="Get a key at https://openrouter.ai/keys",
    )

  
    
    model = "deepseek-ai/deepseek-v4-flash-0731"

    st.divider()
    
    df = st.session_state.df

    st.subheader("Get a key at https://openrouter.ai/keys")

   #reset

    st.divider()
    if st.button("🔄 Reset chat"):
        st.session_state.messages = []
        st.rerun()

# Rebuild the retrieval index whenever the dataset in memory changes
chunk_records = build_chunk_index(st.session_state.df)

# main ui
st.title("Spookluke RL Coach")
st.caption("An AI Rocket League coach trained on Spookluke's own coaching videos. Coaching only, Rocket League only.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask for Rocket League coaching...")

if prompt:
    if not api_key:
        st.error("Enter your OpenRouter API key in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    system_prompt = build_system_prompt(st.session_state.df, chunk_records, prompt)
    api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                stream=True,
                extra_headers={
                    "HTTP-Referer": "https://streamlit.io",
                    "X-Title": "Spookluke RL Coach",
                },
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_response += delta
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Error calling OpenRouter: {e}"
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
