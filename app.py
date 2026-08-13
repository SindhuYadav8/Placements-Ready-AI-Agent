import os
import requests
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# ============================================================
# 1. GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set."
    )


# ============================================================
# 2. LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)


# ============================================================
# 3. WEB SEARCH
# ============================================================

search_engine = DuckDuckGoSearchResults()


# ============================================================
# 4. JOB SEARCH TOOL
# ============================================================

@tool
def job_search(role: str) -> str:
    """
    Search the web for current job openings matching
    the target goal, especially fresher and entry-level
    opportunities in India.
    """

    query = (
        f"{role} fresher jobs India "
        f"entry level campus placement jobs"
    )

    try:
        result = search_engine.invoke(query)

        if not result:
            return "No matching job openings were found."

        return str(result)

    except Exception as e:
        return f"Job search failed: {str(e)}"


# ============================================================
# 5. SKILL GAP ANALYSIS TOOL
# ============================================================

@tool
def skill_gap_analysis(
    role: str,
    resume_text: str
) -> str:
    """
    Compare a student's resume against the target goal
    and identify existing and missing skills.
    """

    prompt = f"""
You are a technical recruiter helping an engineering student
prepare for campus placements.

TARGET GOAL:
{role}

STUDENT RESUME:
{resume_text}

Analyze the resume against the target goal.

Return the result using this structure:

CURRENT SKILLS:
- skill
- skill
- skill

MISSING SKILLS:
- skill
- skill
- skill

PRIORITY SKILLS TO LEARN:
1. skill
2. skill
3. skill

Only identify skills reasonably supported by the resume.
Do not invent experience.
Be concise.
"""

    try:
        response = llm.invoke(prompt)

        content = getattr(response, "content", None)

        if content is not None:
            return str(content)

        return str(response)

    except Exception as e:
        return f"Skill gap analysis failed: {str(e)}"


# ============================================================
# 6. PROJECT RECOMMENDATION TOOL
# ============================================================

@tool
def project_ideas(
    role: str,
    missing_skills: str
) -> str:
    """
    Recommend practical resume-worthy projects based on
    the missing skills for the target goal.
    """

    prompt = f"""
You are a technical mentor helping an engineering student
prepare for campus placements.

TARGET GOAL:
{role}

MISSING SKILLS:
{missing_skills}

Suggest exactly 3 practical and resume-worthy projects.

For every project include:

1. Project Title
2. What to Build
3. Technologies
4. Skills Demonstrated

The projects should be realistic for a student and suitable
for uploading to GitHub.
"""

    try:
        response = llm.invoke(prompt)

        content = getattr(response, "content", None)

        if content is not None:
            return str(content)

        return str(response)

    except Exception as e:
        return f"Project recommendation failed: {str(e)}"


# ============================================================
# 7. GITHUB CHECK TOOL
# ============================================================

@tool
def github_check(
    github_username: str
) -> str:
    """
    Check any public GitHub username and summarize recent
    repository activity and programming languages.
    """

    username = github_username.strip()

    if not username:
        return "GitHub username was not provided."

    github_url = (
        f"https://api.github.com/users/"
        f"{username}/repos"
    )

    params = {
        "sort": "updated",
        "per_page": 8
    }

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Placement-Ready-AI-Agent"
    }

    try:

        response = requests.get(
            github_url,
            params=params,
            headers=headers,
            timeout=10
        )

        if response.status_code == 404:
            return (
                f"GitHub user '{username}' was not found. "
                f"Please check the username."
            )

        if response.status_code != 200:
            return (
                f"Could not fetch GitHub data for "
                f"'{username}'. "
                f"GitHub returned status "
                f"{response.status_code}."
            )

        repos = response.json()

        if not repos:
            return (
                f"GitHub user '{username}' has "
                f"no public repositories."
            )

        lines = []

        for repo in repos:

            name = repo.get(
                "name",
                "Unknown"
            )

            language = repo.get(
                "language"
            ) or "N/A"

            stars = repo.get(
                "stargazers_count",
                0
            )

            updated = repo.get(
                "updated_at",
                ""
            )

            if updated:
                updated = updated[:10]

            lines.append(
                f"- {name} | "
                f"Language: {language} | "
                f"Stars: {stars} | "
                f"Updated: {updated}"
            )

        return (
            f"GitHub Profile: {username}\n\n"
            f"Recent Public Repositories:\n"
            + "\n".join(lines)
        )

    except requests.RequestException as e:
        return (
            f"GitHub request failed: {str(e)}"
        )


# ============================================================
# 8. TOOLS
# ============================================================

tools = [
    job_search,
    skill_gap_analysis,
    project_ideas,
    github_check,
]


# ============================================================
# 9. LANGCHAIN AGENT
# ============================================================

career_agent = create_agent(
    model=llm,
    tools=tools,

    system_prompt="""
You are a Placement-Ready AI Career Agent for engineering
students.

Your purpose is to help students prepare for campus placements.

The student provides:

1. Resume
2. Target goal
3. GitHub username

You should perform the following workflow:

STEP 1:
Use the job search tool to find relevant current job
opportunities.

Focus on:
- India
- Fresher jobs
- Entry-level jobs
- Campus placement opportunities

STEP 2:
Use the skill gap analysis tool to compare the resume
with the target goal.

STEP 3:
Use the project ideas tool to recommend projects based
on the missing skills.

STEP 4:
Use the GitHub check tool to evaluate the student's
public GitHub repositories.

IMPORTANT:

The GitHub username can be ANY public GitHub username.

Do NOT assume a specific username.

Use exactly the username supplied by the student.

After using the tools, provide one structured final report.

Use this format:

PLACEMENT-READY AI CAREER REPORT

1. TARGET GOAL
- Target goal

2. MATCHING JOB OPPORTUNITIES
- Job search findings

3. CURRENT RESUME SKILLS
- Skills found in resume

4. SKILL GAPS
- Missing skills

5. PRIORITY LEARNING PLAN
- Skills to learn first

6. RECOMMENDED PROJECTS
- Project recommendations

7. GITHUB EVALUATION
- Repository activity
- Languages
- Strengths
- Areas for improvement

8. PLACEMENT READINESS
- Overall assessment
- Recommended next steps

Do not invent information.

Only use information obtained from the student's resume
or from the tools.

Be practical, concise, and student-friendly.
"""
)


# ============================================================
# 10. INPUT MODEL
# ============================================================

class CareerAgentInput(BaseModel):

    resume_text: str = Field(
        ...,
        description=(
            "Full text extracted from the "
            "student's resume PDF"
        )
    )

    target_goal: str = Field(
        ...,
        description=(
            "Desired job role, for example "
            "Machine Learning Engineer"
        )
    )

    github_username: str = Field(
        ...,
        description=(
            "Any public GitHub username"
        )
    )


# ============================================================
# 11. OUTPUT MODEL
# ============================================================

class CareerAgentOutput(BaseModel):

    target_goal: str = Field(
        description="The student's desired job role"
    )

    github_username: str = Field(
        description="The GitHub username that was checked"
    )

    final_report: str = Field(
        description="Final placement readiness report"
    )


# ============================================================
# 12. EXTRACT FINAL AI TEXT
# ============================================================

def extract_final_text(result) -> str:

    if not isinstance(result, dict):
        return str(result)

    messages = result.get(
        "messages",
        []
    )

    for message in reversed(messages):

        class_name = (
            message.__class__.__name__
        )

        if class_name != "AIMessage":
            continue

        content = getattr(
            message,
            "content",
            ""
        )

        if isinstance(content, str):

            if content.strip():
                return content

        elif isinstance(content, list):

            text_parts = []

            for block in content:

                if isinstance(block, dict):

                    if block.get("type") == "text":

                        text = block.get(
                            "text",
                            ""
                        )

                        if text:
                            text_parts.append(
                                text
                            )

                elif block:

                    text_parts.append(
                        str(block)
                    )

            combined = "\n".join(
                text_parts
            ).strip()

            if combined:
                return combined

    return str(result)


# ============================================================
# 13. RUN CAREER AGENT
# ============================================================

def run_career_agent(
    payload: CareerAgentInput
) -> CareerAgentOutput:

    # LangServe can pass a dictionary.
    # Convert it to the Pydantic model.

    if isinstance(payload, dict):
        payload = CareerAgentInput.model_validate(
            payload
        )

    user_message = f"""
I am an engineering student preparing for campus placements.

TARGET GOAL:
{payload.target_goal}

GITHUB USERNAME:
{payload.github_username}

RESUME:
{payload.resume_text}

Please:

1. Find relevant job opportunities.
2. Analyze my resume.
3. Identify my skill gaps.
4. Recommend projects.
5. Check my GitHub profile.
6. Give me a placement-readiness report.
"""

    try:

        result = career_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            }
        )

        final_report = extract_final_text(
            result
        )

        return CareerAgentOutput(
            target_goal=payload.target_goal,
            github_username=payload.github_username,
            final_report=final_report
        )

    except Exception as e:

        return CareerAgentOutput(
            target_goal=payload.target_goal,
            github_username=payload.github_username,
            final_report=(
                "Agent execution failed:\n"
                f"{str(e)}"
            )
        )


# ============================================================
# 14. LANGSERVE CHAIN
# ============================================================

career_chain = (
    RunnableLambda(run_career_agent)
    .with_types(
        input_type=CareerAgentInput,
        output_type=CareerAgentOutput
    )
)


# ============================================================
# 15. FASTAPI
# ============================================================

app = FastAPI(
    title="Placement-Ready AI Career Agent",
    description=(
        "AI career agent for engineering students "
        "preparing for campus placements."
    ),
    version="1.0.0"
)


# ============================================================
# 16. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    career_chain,
    path="/career-agent",
    playground_type="default"
)


# ============================================================
# 17. HOME ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "Placement-Ready AI Career Agent",
        "endpoint": "/career-agent",
        "playground": "/career-agent/playground/",
        "docs": "/docs"
    }


# ============================================================
# 18. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# 19. START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )