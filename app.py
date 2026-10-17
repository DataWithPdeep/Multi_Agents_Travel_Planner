import os
import traceback
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

for k in ["GROQ_API_KEY", "TAVILY_API_KEY", "AVIATIONSTACK_API_KEY", "OPENWEATHER_API_KEY", "DATABASE_URL"]:
    if k in st.secrets:
        os.environ[k] = str(st.secrets[k]).strip()

from graph import app

st.set_page_config(
    page_title="Yatra AI",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design: aurora background + glass cards + gradient buttons
# (use together with .streamlit/config.toml dark theme)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
        background:
            radial-gradient(60rem 40rem at -10% -10%, rgba(99,102,241,.38), transparent 60%),
            radial-gradient(50rem 36rem at 110% 20%, rgba(236,72,153,.30), transparent 60%),
            radial-gradient(46rem 32rem at 40% 115%, rgba(245,158,11,.24), transparent 60%),
            #070a1c;
        background-attachment: fixed;
    }
    .block-container {max-width: 1080px; padding-top: 1.2rem; padding-bottom: 4rem;}

    .badge {
        display: inline-block; padding: 6px 14px; border-radius: 999px; font-size: .8rem;
        font-weight: 700; color: #a5b4fc; background: rgba(22,27,58,.6);
        border: 1px solid rgba(255,255,255,.12); margin-bottom: 14px;
    }
    .hero {text-align: center; padding: 18px 0 8px;}
    .hero h1 {
        font-size: clamp(2rem, 6vw, 3.6rem); line-height: 1.15; font-weight: 800;
        margin: 0 0 10px; letter-spacing: -.02em; color: #eef0ff;
    }
    .hero h1 em {
        font-style: normal; color: transparent; -webkit-background-clip: text; background-clip: text;
        background-image: linear-gradient(90deg, #6366f1, #ec4899, #f59e0b, #6366f1);
        background-size: 250% 100%; animation: shine 6s linear infinite;
    }
    .hero p {color: #9aa3c7; max-width: 620px; margin: 0 auto; font-size: 1.02rem;}
    @keyframes shine {to {background-position: 250% 0;}}

    /* glass cards: st.container(key="card_...") */
    div[class*="st-key-card"] {
        background: rgba(22,27,58,.62); backdrop-filter: blur(18px);
        border: 1px solid rgba(255,255,255,.10); border-radius: 22px; padding: 1.1rem 1.2rem;
        box-shadow: 0 24px 60px -28px rgba(80,70,200,.5);
    }
    div[data-testid="stMetric"] {
        background: rgba(22,27,58,.62); border: 1px solid rgba(255,255,255,.10);
        border-radius: 18px; padding: 14px 18px;
    }

    /* stepper */
    .stepper {display: flex; gap: 10px; margin: 18px 0;}
    .step {
        flex: 1; text-align: center; padding: 10px 8px; border-radius: 14px; font-weight: 700;
        font-size: .88rem; color: #9aa3c7; background: rgba(22,27,58,.6);
        border: 1px solid rgba(255,255,255,.10);
    }
    .step.active {color: #a5b4fc; border-color: #6366f1; box-shadow: 0 8px 24px -10px #6366f1;}
    .step.done {color: #34d399; border-color: #10b981;}

    /* buttons */
    .stButton > button {border-radius: 14px; font-weight: 700; transition: .2s;}
    .stButton > button:hover {transform: translateY(-2px);}
    button[kind="primary"] {
        border: 0 !important; color: #fff !important; padding: .8rem 1rem !important;
        background: linear-gradient(120deg, #6366f1, #ec4899 60%, #f59e0b) !important;
        box-shadow: 0 14px 30px -12px #ec4899 !important;
    }
    .stTextArea textarea {border-radius: 16px; background: rgba(255,255,255,.05);}

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {gap: 6px;}
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px; padding: 8px 16px; background: rgba(22,27,58,.6);
        border: 1px solid rgba(255,255,255,.10);
    }
    .stTabs [aria-selected="true"] {background: linear-gradient(120deg, #6366f1, #ec4899); color: #fff;}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}

    @media (max-width: 640px) {
        .stepper {flex-direction: column; gap: 6px;}
        .block-container {padding-left: 1rem; padding-right: 1rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------
TEXT = {
    "Hindi": {
        "badge": "✨ 6 AI एजेंट · एक यात्रा",
        "h1a": "सपनों की यात्रा,",
        "h1b": "अब मिनटों में प्लान",
        "sub": "फ्लाइट, होटल, मौसम और बजट. सब कुछ AI एजेंटों की टीम आपके लिए तैयार करती है।",
        "lang_label": "भाषा / Language",
        "lang_note": "AI का जवाब उसी भाषा में बनता है जो ड्राफ्ट बनाते समय चुनी हो।",
        "steps": ["1 · अनुरोध", "2 · समीक्षा", "3 · अंतिम योजना"],
        "session": "⚙️ सेशन",
        "user_id": "यूज़र ID",
        "new_thread": "🔄 नया थ्रेड",
        "query_label": "आप कहाँ जाना चाहते हैं?",
        "placeholder": "जैसे: 7 दिन की जापान यात्रा, बजट 2 लाख रुपये तक, सस्ते होटल और रात की फ्लाइट नहीं।",
        "examples": "उदाहरण आज़माएँ",
        "example_list": [
            "7 दिन की जापान यात्रा, बजट 2 लाख रुपये तक, सस्ते होटल",
            "दिल्ली से गोवा, 4 दिन, दोस्तों के साथ",
            "परिवार के साथ मनाली में 5 दिन की सर्दियों की छुट्टी",
        ],
        "create": "🚀 ड्राफ्ट प्लान बनाएं",
        "warn_empty": "पहले अपनी यात्रा का अनुरोध लिखें।",
        "working": "🧠 एजेंट प्लान बना रहे हैं...",
        "done": "✅ तैयार है",
        "error": "कुछ गड़बड़ हो गई",
        "blocked": "यह अनुरोध यात्रा योजना से जुड़ा नहीं लगता",
        "destination": "गंतव्य",
        "duration": "अवधि",
        "budget_m": "बजट",
        "supervisor": "🧭 सुपरवाइज़र की योजना",
        "selected": "चुने गए एजेंट",
        "tabs": ["📝 कार्यक्रम", "✈️ फ्लाइट", "🏨 होटल", "☁️ मौसम", "💰 बजट"],
        "no_data": "अभी कोई डेटा नहीं",
        "approval_title": "✅ आपकी मंज़ूरी",
        "approve_q": "क्या आप इस ड्राफ्ट को मंज़ूर करते हैं?",
        "yes": "हाँ, मंज़ूर है",
        "no": "नहीं, बदलाव करें",
        "feedback": "आप क्या बदलना चाहेंगे?",
        "submit": "📨 भेजें",
        "finalizing": "✍️ अंतिम योजना बन रही है...",
        "final_title": "🎉 आपकी अंतिम यात्रा योजना",
        "download": "⬇️ योजना डाउनलोड करें",
        "new_plan": "➕ नई यात्रा प्लान करें",
    },
    "English": {
        "badge": "✨ 6 AI agents · one trip",
        "h1a": "Dream trips,",
        "h1b": "planned in minutes",
        "sub": "Flights, hotels, weather and budget. A team of AI agents prepares it all for you.",
        "lang_label": "भाषा / Language",
        "lang_note": "The AI replies in the language selected when the draft is created.",
        "steps": ["1 · Request", "2 · Review", "3 · Final plan"],
        "session": "⚙️ Session",
        "user_id": "User ID",
        "new_thread": "🔄 New Thread",
        "query_label": "Where do you want to go?",
        "placeholder": "e.g. 7-day Japan trip under Rs. 2 lakh, budget hotels, no overnight flights.",
        "examples": "Try an example",
        "example_list": [
            "7-day Japan trip under Rs. 2 lakh, budget hotels",
            "4-day Goa trip from Delhi with friends",
            "5-day family winter trip to Manali",
        ],
        "create": "🚀 Create Draft Plan",
        "warn_empty": "Enter a travel request first.",
        "working": "🧠 Agents are planning...",
        "done": "✅ Ready",
        "error": "Something went wrong",
        "blocked": "This doesn't look like a travel planning request",
        "destination": "Destination",
        "duration": "Duration",
        "budget_m": "Budget",
        "supervisor": "🧭 Supervisor plan",
        "selected": "Selected agents",
        "tabs": ["📝 Itinerary", "✈️ Flights", "🏨 Hotels", "☁️ Weather", "💰 Budget"],
        "no_data": "No data yet",
        "approval_title": "✅ Your approval",
        "approve_q": "Do you approve this draft?",
        "yes": "Yes, approve",
        "no": "No, revise it",
        "feedback": "What would you like to change?",
        "submit": "📨 Submit",
        "finalizing": "✍️ Creating final plan...",
        "final_title": "🎉 Your final travel plan",
        "download": "⬇️ Download plan",
        "new_plan": "➕ Plan another trip",
    },
}

# ---------------------------------------------------------------------------
# Session defaults (Hindi is the default language)
# ---------------------------------------------------------------------------
st.session_state.setdefault("language", "Hindi")
st.session_state.setdefault("query", "")
st.session_state.setdefault("user_id", "demo_user")
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"{st.session_state.user_id}_{uuid.uuid4().hex[:8]}"

lang = st.session_state.language
t = TEXT[lang]


def set_query(text: str):
    st.session_state.query = text


def start_new_thread():
    st.session_state.thread_id = f"{st.session_state.user_id}_{uuid.uuid4().hex[:8]}"
    st.session_state.pop("waiting_for_approval", None)
    st.session_state.pop("latest_result", None)
    st.session_state.pop("last_error", None)


def new_plan():
    start_new_thread()
    st.session_state.query = ""


def show_error(tb: str):
    """Error ko status box ke BAHAR dikhata hai taaki saaf dikhe."""
    st.error(t["error"])
    st.code(tb, language="text")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader(t["session"])
    st.text_input(t["user_id"], key="user_id")
    st.button(t["new_thread"], use_container_width=True, on_click=start_new_thread)
    st.caption(f"🧵 {st.session_state.thread_id}")

# ---------------------------------------------------------------------------
# Top bar: language switch
# ---------------------------------------------------------------------------
_, lang_col = st.columns([3, 1])
with lang_col:
    st.radio(
        t["lang_label"],
        ["Hindi", "English"],
        format_func=lambda x: "हिंदी" if x == "Hindi" else "English",
        horizontal=True,
        key="language",
    )

st.markdown(
    f"""
    <div class="hero">
        <span class="badge">{t["badge"]}</span>
        <h1>{t["h1a"]} <em>{t["h1b"]}</em></h1>
        <p>{t["sub"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------
result = st.session_state.get("latest_result")
waiting = st.session_state.get("waiting_for_approval", False)

if result and result.get("final_response") and not waiting:
    stage = 3
elif waiting:
    stage = 2
else:
    stage = 1

steps_html = "".join(
    f'<div class="step {"done" if i < stage else "active" if i == stage else ""}">{label}</div>'
    for i, label in enumerate(t["steps"], start=1)
)
st.markdown(f'<div class="stepper">{steps_html}</div>', unsafe_allow_html=True)

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ---------------------------------------------------------------------------
# Step 1 — Request
# ---------------------------------------------------------------------------
if stage == 1:
    with st.container(key="card_request"):
        st.text_area(t["query_label"], placeholder=t["placeholder"], height=120, key="query")

        st.caption(t["examples"])
        cols = st.columns(len(t["example_list"]))
        for col, example in zip(cols, t["example_list"]):
            col.button(example, key=f"ex_{example}", use_container_width=True,
                       on_click=set_query, args=(example,))

        create_clicked = st.button(t["create"], type="primary", use_container_width=True)

    if create_clicked:
        query = st.session_state.query
        if not query.strip():
            st.warning(t["warn_empty"])
        else:
            new_result = None
            error_tb = None
            with st.status(t["working"], expanded=False) as status:
                try:
                    new_result = app.invoke(
                        {
                            "messages": [HumanMessage(content=query)],
                            "user_id": st.session_state.user_id,
                            "user_query": query,
                            "language": lang,
                            "flight_results": "",
                            "hotel_results": "",
                            "weather_results": "",
                            "budget_results": "",
                            "itinerary": "",
                            "final_response": "",
                            "llm_calls": 0,
                        },
                        config=config,
                    )
                    status.update(label=t["done"], state="complete")
                except Exception:  # noqa: BLE001
                    error_tb = traceback.format_exc()
                    print(error_tb)  # Manage app -> logs me bhi dikhega
                    status.update(label=t["error"], state="error")

            if error_tb:
                show_error(error_tb)

            if new_result is not None:
                st.session_state.latest_result = new_result
                st.session_state.waiting_for_approval = "__interrupt__" in new_result
                st.rerun()

# ---------------------------------------------------------------------------
# Results (step 2 and 3)
# ---------------------------------------------------------------------------
result = st.session_state.get("latest_result")

if result:
    blocked = not result.get("selected_agents") and result.get("final_response")

    if blocked:
        st.warning(f"**{t['blocked']}**\n\n{result.get('supervisor_reasoning', '')}")
        st.button(t["new_plan"], on_click=new_plan)
    else:
        c = result.get("trip_constraints", {}) or {}
        m1, m2, m3 = st.columns(3)
        m1.metric(t["destination"], c.get("destination") or "—")
        m2.metric(t["duration"], c.get("duration") or "—")
        m3.metric(t["budget_m"], c.get("budget") or "—")

        with st.expander(t["supervisor"]):
            st.write(result.get("supervisor_reasoning", ""))
            st.caption(f"{t['selected']}: {', '.join(result.get('selected_agents', [])) or '—'}")

        if "__interrupt__" in result:
            draft = result["__interrupt__"][0].value.get("draft_itinerary", "")
        else:
            draft = result.get("itinerary", "")

        contents = [
            draft,
            result.get("flight_results", ""),
            result.get("hotel_results", ""),
            result.get("weather_results", ""),
            result.get("budget_results", ""),
        ]
        for tab, text in zip(st.tabs(t["tabs"]), contents):
            with tab:
                if text:
                    st.markdown(text)
                else:
                    st.info(t["no_data"])

# ---------------------------------------------------------------------------
# Step 2 — Approval
# ---------------------------------------------------------------------------
if stage == 2:
    with st.container(key="card_approval"):
        st.subheader(t["approval_title"])
        approved = st.radio(
            t["approve_q"], ["yes", "no"], format_func=lambda x: t[x],
            horizontal=True, key="approval_choice",
        )
        feedback = st.text_area(t["feedback"], disabled=approved == "yes", key="feedback_text")
        submit_clicked = st.button(t["submit"], type="primary", use_container_width=True)

    if submit_clicked:
        final_result = None
        error_tb = None
        with st.status(t["finalizing"], expanded=False) as status:
            try:
                final_result = app.invoke(
                    Command(resume={"approved": approved == "yes", "feedback": feedback, "language": lang}),
                    config=config,
                )
                status.update(label=t["done"], state="complete")
            except Exception:  # noqa: BLE001
                error_tb = traceback.format_exc()
                print(error_tb)
                status.update(label=t["error"], state="error")

        if error_tb:
            show_error(error_tb)

        if final_result is not None:
            st.session_state.latest_result = final_result
            st.session_state.waiting_for_approval = "__interrupt__" in final_result
            st.rerun()

# ---------------------------------------------------------------------------
# Step 3 — Final plan
# ---------------------------------------------------------------------------
if stage == 3 and result and result.get("selected_agents"):
    with st.container(key="card_final"):
        st.subheader(t["final_title"])
        st.markdown(result["final_response"])
        d_col, n_col = st.columns(2)
        d_col.download_button(
            t["download"], data=result["final_response"], file_name="travel_plan.md",
            mime="text/markdown", use_container_width=True,
        )
        n_col.button(t["new_plan"], use_container_width=True, on_click=new_plan)