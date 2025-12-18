import streamlit as st

st.set_page_config(page_title="V2G Toolkit", layout="wide")

STEPS = [
    "Project Setup",
    "Load Profile",
    "PV System",
    "Storage (BESS + EV)",
    "Tariff & Policy",
    "Economics",
    "Scenarios",
    "Run & Results",
    "Compare",
    "Export / Report",
]

def init_state():
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "completed" not in st.session_state:
        st.session_state.completed = {name: False for name in STEPS}
    if "model" not in st.session_state:
        st.session_state.model = {}

init_state()

st.sidebar.title("V2G Toolkit Wizard")

# sidebar navigation with status icons
for i, name in enumerate(STEPS):
    status = "✅" if st.session_state.completed.get(name, False) else "⚠️"
    if st.sidebar.button(f"{status} {i+1}. {name}", use_container_width=True):
        st.session_state.step = i
        st.rerun()

st.sidebar.divider()
colA, colB = st.sidebar.columns(2)
with colA:
    if st.button("⬅ Back", use_container_width=True, disabled=(st.session_state.step == 0)):
        st.session_state.step -= 1
        st.rerun()
with colB:
    if st.button("Next ➡", use_container_width=True, disabled=(st.session_state.step == len(STEPS)-1)):
        st.session_state.step += 1
        st.rerun()

st.title(STEPS[st.session_state.step])
st.caption(f"Step {st.session_state.step+1} of {len(STEPS)}")

st.info("Wizard framework installed ✅ Next: we will add the individual pages for each step.")

