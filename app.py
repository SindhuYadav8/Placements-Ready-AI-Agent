import streamlit as st
from typing import TypedDict, Dict, List, Any


# --------------------------------------------------
# Agent State
# --------------------------------------------------

class AgentState(TypedDict, total=False):
    priority_skills: Dict[str, List[str]]
    missing_skills: List[str]
    final_report: Dict[str, Any]


# --------------------------------------------------
# Final Report Builder
# --------------------------------------------------

def final_report_builder(state: AgentState) -> AgentState:

    priority = state.get("priority_skills", {})

    high = priority.get("HIGH", []) or []
    medium = priority.get("MEDIUM", []) or []
    low = priority.get("LOW", []) or []

    # Combine skills according to importance
    all_priority = high + medium + low

    # Remove duplicate skills
    seen = set()
    required_skills = []

    for skill in all_priority:

        skill_clean = str(skill).strip()

        if skill_clean and skill_clean.lower() not in seen:

            seen.add(skill_clean.lower())

            if any(
                str(s).strip().lower() == skill_clean.lower()
                for s in high
            ):
                priority_label = "CRITICAL"

            elif any(
                str(s).strip().lower() == skill_clean.lower()
                for s in medium
            ):
                priority_label = "HIGH"

            else:
                priority_label = "MEDIUM"

            required_skills.append({
                "skill": skill_clean,
                "priority": priority_label
            })

    # Maximum 5 required skills
    required_skills = required_skills[:5]

    # --------------------------------------------------
    # Missing Skills
    # --------------------------------------------------

    missing = state.get("missing_skills", []) or []

    seen_missing = set()
    missing_skills = []

    for skill in missing:

        skill_clean = str(skill).strip()

        if skill_clean and skill_clean.lower() not in seen_missing:

            seen_missing.add(skill_clean.lower())
            missing_skills.append(skill_clean)

    # Maximum 3 missing skills
    missing_skills = missing_skills[:3]

    # --------------------------------------------------
    # Simple Final Report
    # --------------------------------------------------

    report = {
        "required_skills": required_skills,
        "missing_skills": missing_skills
    }

    return {
        **state,
        "final_report": report
    }


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(
    page_title="Skill Gap Report",
    page_icon="",
    layout="centered"
)

st.title("Skill Gap Report")
st.write("Your most important skills to focus on")


# --------------------------------------------------
# Example Input
# Replace this section with your actual agent output
# --------------------------------------------------

state: AgentState = {

    "priority_skills": {

        "HIGH": [
            "Python",
            "Machine Learning",
            "SQL"
        ],

        "MEDIUM": [
            "Deep Learning",
            "Git"
        ],

        "LOW": [
            "Docker",
            "AWS"
        ]
    },

    "missing_skills": [
        "Machine Learning",
        "SQL",
        "Docker"
    ]
}


# --------------------------------------------------
# Generate Report
# --------------------------------------------------

result = final_report_builder(state)

report = result["final_report"]


# --------------------------------------------------
# Display Required Skills
# --------------------------------------------------

st.subheader("Required Skills")

for item in report["required_skills"]:

    skill = item["skill"]
    priority = item["priority"]

    st.write(f"**{skill}** — {priority}")


# --------------------------------------------------
# Display Missing Skills
# --------------------------------------------------

if report["missing_skills"]:

    st.subheader("Skills You Need to Improve")

    for skill in report["missing_skills"]:
        st.write(f"- {skill}")

else:

    st.success("No major skill gaps identified.")
