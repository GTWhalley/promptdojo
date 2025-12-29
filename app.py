"""
Prompt Dojo - Prompt Engineering Trainer
A lightweight application that trains users in prompt engineering through
active A/B testing and interactive grading.
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import random
from datetime import datetime
import openai
import google.generativeai as genai
import anthropic


# =============================================================================
# Configuration & Constants
# =============================================================================

QUIZ_LENGTH = 10

# System prompt for generating A/B comparison questions
AB_GENERATOR_PROMPT = """You are a prompt engineering instructor creating a CHALLENGING A/B comparison question.

Generate a realistic scenario where someone would need to prompt an AI, then create two prompts:
- A FLAWED prompt (has subtle issues - could be too verbose, wrong focus, missing key constraint, or over-constrained)
- A BETTER prompt (addresses the flaw while being well-crafted)

IMPORTANT: Make this challenging! The difference should NOT just be length. Both prompts should be similar in length (within 20% of each other). The flaw should be SUBTLE:
- Wrong tone for the audience
- Missing ONE critical constraint while having others
- Over-specifying irrelevant details while missing important ones
- Asking for the wrong output format
- Missing context that would change the response
- Being specific about the wrong things
- Having contradictory instructions

The scenario should be practical and relatable (e.g., writing emails, summarizing text, coding help, creative writing, data analysis).

Respond in this EXACT JSON format (no markdown, just raw JSON):
{
    "scenario": "Brief description of what the user wants to accomplish",
    "weak_prompt": "The flawed version - similar length to strong but with subtle issues",
    "strong_prompt": "The better version - addresses the subtle flaw",
    "explanation": "2-3 sentences explaining the SPECIFIC flaw in the weak prompt and why the strong prompt fixes it. Be precise about what was wrong."
}"""

# System prompt for generating challenge scenarios
CHALLENGE_GENERATOR_PROMPT = """You are a prompt engineering instructor. Generate a challenging scenario for a student to practice writing prompts.

Create a scenario that requires the student to write a detailed, well-structured prompt. The scenario should be:
- Practical and realistic
- Moderately complex (requiring multiple considerations)
- Clear about what the end goal is

Respond in this EXACT JSON format (no markdown, just raw JSON):
{
    "title": "Short title for the challenge (3-5 words)",
    "scenario": "Detailed description of what the user needs to accomplish. Be specific about the context and requirements.",
    "ideal_prompt": "An example of a well-crafted prompt that would effectively accomplish this task",
    "key_elements": ["element1", "element2", "element3", "element4"]
}

The key_elements should list 4 critical components that a good prompt for this scenario should include."""

# System prompt for grading user prompts with detailed metrics
GRADING_SYSTEM_PROMPT = """You are a strict but encouraging Prompt Engineering Instructor. Grade the student's prompt based on these 4 metrics, each scored 1-5:

1. **CLARITY (1-5)**: Is the intent immediately clear? Is there any ambiguity about what the AI should do?
2. **SPECIFICITY (1-5)**: Does it provide enough detail? Are there concrete requirements, not vague requests?
3. **CONSTRAINTS (1-5)**: Are there appropriate boundaries? (format, length, tone, scope, edge cases)
4. **CONTEXT (1-5)**: Does it give the AI enough background? Role assignment, audience, purpose?

Respond in this EXACT format:

## Scores

| Metric | Score | Notes |
|--------|-------|-------|
| Clarity | X/5 | Brief note |
| Specificity | X/5 | Brief note |
| Constraints | X/5 | Brief note |
| Context | X/5 | Brief note |

**TOTAL: XX/20**

## What You Did Well
- Point 1
- Point 2

## Areas for Improvement
- Point 1 with specific suggestion
- Point 2 with specific suggestion

## The Ideal Prompt
```
[Write out what an ideal prompt for this scenario would look like]
```

## Why This Works Better
Explain in 2-3 sentences why the ideal prompt would produce better results from an LLM. Focus on how LLMs process instructions and why specificity matters."""

# System prompt for grading user's own general prompts (no scenario context)
GENERAL_GRADING_SYSTEM_PROMPT = """You are a strict but encouraging Prompt Engineering Instructor. A user has submitted their own prompt for evaluation before using it. Grade this prompt based on these 4 metrics, each scored 1-5:

1. **CLARITY (1-5)**: Is the intent immediately clear? Is there any ambiguity about what the AI should do?
2. **SPECIFICITY (1-5)**: Does it provide enough detail? Are there concrete requirements, not vague requests?
3. **CONSTRAINTS (1-5)**: Are there appropriate boundaries? (format, length, tone, scope, edge cases)
4. **CONTEXT (1-5)**: Does it give the AI enough background? Role assignment, audience, purpose?

Respond in this EXACT format:

## Scores

| Metric | Score | Notes |
|--------|-------|-------|
| Clarity | X/5 | Brief note |
| Specificity | X/5 | Brief note |
| Constraints | X/5 | Brief note |
| Context | X/5 | Brief note |

**TOTAL: XX/20**

## What You Did Well
- Point 1
- Point 2

## Areas for Improvement
- Point 1 with specific suggestion
- Point 2 with specific suggestion

## Improved Version
```
[Write out an improved version of their prompt that addresses the issues you identified]
```

## Why This Works Better
Explain in 2-3 sentences why the improved prompt would produce better results from an LLM. Focus on how LLMs process instructions and why specificity matters."""

DEMO_AB_QUESTION = {
    "scenario": "You need to write a prompt that helps summarize technical documentation for a general audience.",
    "weak_prompt": "Summarize this technical documentation in simple terms. Make it exactly 5 paragraphs long. Use bullet points for each section. Include all technical specifications. Target audience is general public. Remove all jargon. Keep the technical accuracy. Make it engaging and fun to read.",
    "strong_prompt": "Summarize this technical documentation for a general audience with no technical background. Focus on what the product DOES and why it matters, not how it works internally. Use analogies to everyday objects where helpful. Aim for 2-3 short paragraphs. Avoid jargon - if a technical term is essential, briefly define it.",
    "explanation": "The weak prompt has contradictory instructions (include all technical specs BUT remove jargon and target general public), over-specifies format (exactly 5 paragraphs with bullet points), and asks for incompatible goals (technical accuracy AND fun to read for non-technical readers). The strong prompt has a clear, consistent goal and appropriate constraints without contradictions."
}

DEMO_CHALLENGE = {
    "title": "Customer Support Response",
    "scenario": "You work for a software company and need to write a prompt that helps generate responses to customer support tickets. The responses should acknowledge the customer's issue, provide a solution or next steps, and maintain a helpful tone.",
    "ideal_prompt": "You are a friendly customer support agent for TechCorp software. A customer has submitted a support ticket. Write a response that: 1) Acknowledges their issue with empathy, 2) Provides a clear solution or next steps, 3) Offers additional help if needed. Keep the tone professional but warm. Use simple language avoiding technical jargon unless necessary. Response should be 3-5 sentences. Format: Start with a greeting using their name, end with your name 'Alex from TechCorp Support'.",
    "key_elements": ["Role/persona assignment", "Clear structure (acknowledge, solve, offer help)", "Tone and language guidelines", "Format specifications"]
}

DEMO_GRADING_RESULT = """## Scores

| Metric | Score | Notes |
|--------|-------|-------|
| Clarity | 4/5 | Intent is clear but could be more specific |
| Specificity | 3/5 | Missing some key details |
| Constraints | 3/5 | Limited format/length guidance |
| Context | 3/5 | Could use more background |

**TOTAL: 13/20**

## What You Did Well
- Clear main objective stated
- Basic structure is present

## Areas for Improvement
- Add specific tone guidelines (professional? casual? empathetic?)
- Include format constraints (length, structure, greeting/closing)
- Specify the role or persona the AI should adopt

## The Ideal Prompt
```
You are a friendly customer support agent for TechCorp software. A customer has submitted a support ticket. Write a response that: 1) Acknowledges their issue with empathy, 2) Provides a clear solution or next steps, 3) Offers additional help if needed. Keep the tone professional but warm. Use simple language avoiding technical jargon unless necessary. Response should be 3-5 sentences. Format: Start with a greeting using their name, end with your name 'Alex from TechCorp Support'.
```

## Why This Works Better
LLMs perform best when given explicit structure and constraints. The ideal prompt assigns a clear role (support agent), breaks down the task into numbered steps, specifies tone and language preferences, and defines the exact format. This eliminates guesswork and ensures consistent, on-brand responses every time.

---
*In the full release, this would be a real-time evaluation from an AI model using your API key.*"""

DEMO_GENERAL_GRADING_RESULT = """## Scores

| Metric | Score | Notes |
|--------|-------|-------|
| Clarity | 3/5 | Intent is understandable but could be more precise |
| Specificity | 3/5 | Lacks concrete details about requirements |
| Constraints | 2/5 | Missing format, length, and scope boundaries |
| Context | 3/5 | Some background but missing role and audience |

**TOTAL: 11/20**

## What You Did Well
- You have a clear main objective
- The prompt addresses a real task

## Areas for Improvement
- Add a specific role or persona for the AI to adopt (e.g., "You are an expert...")
- Include format constraints (length, structure, style)
- Specify the target audience and tone
- Add concrete examples or edge cases to handle

## Improved Version
```
You are an expert [role]. I need you to [specific task].

Requirements:
- Output format: [specify format]
- Length: [specify length]
- Tone: [specify tone]
- Target audience: [specify audience]

Please ensure you [additional constraints or considerations].
```

## Why This Works Better
The improved prompt provides explicit structure that guides the LLM's response. By specifying role, format, length, and audience, you eliminate ambiguity and ensure the AI understands exactly what output you need. LLMs perform best when they have clear boundaries and concrete requirements rather than open-ended requests.

---
*In the full release, this would be a real-time evaluation from an AI model using your API key.*"""


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "api_key": "",
        "api_provider": "OpenAI",
        "demo_mode": False,
        "api_validated": False,
        "lesson_selected": None,  # None, "compare", or "challenge"
        "quiz_score": 0,
        "quiz_index": 0,
        "module_unlocked": False,
        "show_feedback": False,
        "last_choice_correct": None,
        "last_explanation": "",
        "current_question": None,
        "correct_side": None,  # "A" or "B" - which side has the correct answer
        "challenge_scenario": None,
        "challenge_result": None,
        "challenge_graded": False,
        "gemini_model": None,
        # Grade My Prompt mode
        "grade_my_prompt_result": None,
        "grade_my_prompt_graded": False,
        # Analysis history
        "analysis_history": [],  # List of {"prompt": str, "result": str, "score": int, "timestamp": str}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# API Integration
# =============================================================================

def call_llm(prompt: str, system_prompt: str = None) -> str:
    """Make an API call to the selected LLM provider."""
    try:
        if st.session_state.api_provider == "OpenAI":
            client = openai.OpenAI(api_key=st.session_state.api_key)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1500
            )
            return response.choices[0].message.content
        elif st.session_state.api_provider == "Claude":
            client = anthropic.Anthropic(api_key=st.session_state.api_key)
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1500,
                system=system_prompt if system_prompt else "",
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        else:  # Gemini
            genai.configure(api_key=st.session_state.api_key)
            model_name = st.session_state.get("gemini_model", "models/gemini-1.5-flash")
            model = genai.GenerativeModel(model_name)
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = model.generate_content(full_prompt)
            return response.text
    except Exception as e:
        return f"Error: {str(e)}"


def test_api_connection(api_key: str, provider: str) -> tuple[bool, str]:
    """Test the API connection with the provided key."""
    try:
        if provider == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say 'OK'"}],
                max_tokens=5
            )
            return True, "Connected to OpenAI"
        elif provider == "Claude":
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'OK'"}]
            )
            return True, "Connected to Claude"
        else:  # Gemini
            genai.configure(api_key=api_key)
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if not available_models:
                return False, "No compatible Gemini models found."
            model_name = next((m for m in available_models if 'flash' in m.lower()), available_models[0])
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say 'OK'")
            st.session_state.gemini_model = model_name
            return True, "Connected to Gemini"
    except openai.AuthenticationError:
        return False, "Invalid OpenAI API key."
    except anthropic.AuthenticationError:
        return False, "Invalid Claude API key."
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


def generate_ab_question() -> dict:
    """Generate a new A/B comparison question using the LLM."""
    if st.session_state.demo_mode:
        return DEMO_AB_QUESTION

    response = call_llm("Generate a prompt engineering A/B comparison question for beginners.", AB_GENERATOR_PROMPT)

    try:
        # Clean up response - remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        return json.loads(response)
    except json.JSONDecodeError:
        # Fallback to demo question if parsing fails
        return DEMO_AB_QUESTION


def generate_challenge() -> dict:
    """Generate a new challenge scenario using the LLM."""
    if st.session_state.demo_mode:
        return DEMO_CHALLENGE

    response = call_llm("Generate a prompt engineering challenge scenario.", CHALLENGE_GENERATOR_PROMPT)

    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        return json.loads(response)
    except json.JSONDecodeError:
        return DEMO_CHALLENGE


def grade_prompt(user_prompt: str, scenario: str) -> str:
    """Grade the user's prompt with detailed metrics."""
    if st.session_state.demo_mode:
        return DEMO_GRADING_RESULT

    grading_request = f"""The student was given this scenario:
"{scenario}"

The student wrote this prompt:
"{user_prompt}"

Grade this prompt according to your rubric."""

    return call_llm(grading_request, GRADING_SYSTEM_PROMPT)


def grade_general_prompt(user_prompt: str) -> str:
    """Grade a user's own prompt without a specific scenario context."""
    if st.session_state.demo_mode:
        return DEMO_GENERAL_GRADING_RESULT

    grading_request = f"""The user has submitted the following prompt for evaluation:

"{user_prompt}"

Grade this prompt according to your rubric. Provide specific, actionable feedback and an improved version."""

    return call_llm(grading_request, GENERAL_GRADING_SYSTEM_PROMPT)


# =============================================================================
# Quiz Logic
# =============================================================================

def handle_answer(choice: str):
    """Process the user's answer choice."""
    st.session_state.last_choice_correct = (choice == st.session_state.correct_side)
    st.session_state.last_explanation = st.session_state.current_question["explanation"]

    if st.session_state.last_choice_correct:
        st.session_state.quiz_score += 1

    st.session_state.show_feedback = True


def advance_quiz():
    """Move to the next question or complete the quiz."""
    st.session_state.quiz_index += 1
    st.session_state.show_feedback = False
    st.session_state.last_choice_correct = None
    st.session_state.last_explanation = ""
    st.session_state.current_question = None
    st.session_state.correct_side = None

    if st.session_state.quiz_index >= QUIZ_LENGTH:
        st.session_state.module_unlocked = True


def reset_quiz():
    """Reset the quiz to start over."""
    st.session_state.quiz_score = 0
    st.session_state.quiz_index = 0
    st.session_state.module_unlocked = False
    st.session_state.show_feedback = False
    st.session_state.last_choice_correct = None
    st.session_state.last_explanation = ""
    st.session_state.current_question = None
    st.session_state.correct_side = None


# =============================================================================
# Constants for API Providers
# =============================================================================

API_PROVIDERS = {
    "OpenAI": {
        "name": "OpenAI",
        "url": "https://platform.openai.com/api-keys",
        "icon": "🟢"
    },
    "Gemini": {
        "name": "Google Gemini",
        "url": "https://aistudio.google.com/app/apikey",
        "icon": "🔵"
    },
    "Claude": {
        "name": "Anthropic Claude",
        "url": "https://console.anthropic.com/settings/keys",
        "icon": "🟠"
    }
}


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render the settings sidebar."""
    with st.sidebar:
        # API Provider Selection
        st.markdown("#### SELECT PROVIDER")

        provider_options = list(API_PROVIDERS.keys())
        current_index = provider_options.index(st.session_state.api_provider) if st.session_state.api_provider in provider_options else 0

        provider = st.radio(
            "API Provider",
            provider_options,
            index=current_index,
            disabled=st.session_state.demo_mode,
            label_visibility="collapsed",
            format_func=lambda x: f"{API_PROVIDERS[x]['icon']} {API_PROVIDERS[x]['name']}"
        )

        # Get API Key button
        provider_info = API_PROVIDERS[provider]
        st.link_button(
            f"GET {provider.upper()} API KEY →",
            provider_info["url"],
            use_container_width=True
        )

        st.divider()

        # If provider changed, invalidate the API connection (different key needed)
        if provider != st.session_state.api_provider:
            st.session_state.api_provider = provider
            st.session_state.api_validated = False
            st.session_state.api_key = ""  # Clear key when switching providers
        else:
            st.session_state.api_provider = provider

        demo_mode = st.checkbox(
            "Demo Mode (No API Key Required)",
            value=st.session_state.demo_mode,
            help="Run through the full experience with sample data - no API key needed!"
        )
        if demo_mode != st.session_state.demo_mode:
            st.session_state.demo_mode = demo_mode
            if demo_mode:
                st.session_state.api_validated = True
            else:
                st.session_state.api_validated = False
            st.rerun()

        if not st.session_state.demo_mode:
            api_key = st.text_input(
                f"{provider} API Key:",
                type="password",
                value=st.session_state.api_key,
                help="Your API key is stored in memory only and never saved to disk."
            )

            # Only update if the key changed (prevents losing validation on rerun)
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key
                # Invalidate if key changed
                if st.session_state.api_validated:
                    st.session_state.api_validated = False

            if st.button("Connect", type="primary"):
                if api_key:
                    with st.spinner("Testing connection..."):
                        success, message = test_api_connection(api_key, provider)
                        if success:
                            st.success(message)
                            st.session_state.api_validated = True
                            st.session_state.api_key = api_key  # Ensure key is saved
                        else:
                            st.error(message)
                            st.session_state.api_validated = False
                else:
                    st.warning("Please enter an API key first.")

            if st.session_state.api_validated:
                st.success("API Connected")
            else:
                st.info("Enter your API key to begin")
        else:
            st.success("Demo Mode Active")

        st.divider()

        # Show current lesson and option to change
        if st.session_state.lesson_selected:
            st.markdown("#### ACTIVE MODE")
            lesson_names = {
                "compare": "COMPARE",
                "challenge": "CHALLENGE",
                "grade": "ANALYZE"
            }
            lesson_name = lesson_names.get(st.session_state.lesson_selected, "UNKNOWN")
            st.write(f"**{lesson_name}**")

            if st.button("SWITCH MODE"):
                reset_quiz()
                st.session_state.lesson_selected = None
                st.session_state.challenge_scenario = None
                st.session_state.challenge_result = None
                st.session_state.challenge_graded = False
                st.session_state.grade_my_prompt_result = None
                st.session_state.grade_my_prompt_graded = False
                st.rerun()

            st.divider()

        # Progress section - only show for compare mode
        if st.session_state.lesson_selected == "compare":
            st.markdown("#### PROGRESS")
            st.write(f"**Score:** {st.session_state.quiz_score}/{QUIZ_LENGTH}")
            st.write(f"**Completed:** {st.session_state.quiz_index}/{QUIZ_LENGTH}")

            if st.session_state.quiz_index > 0 or st.session_state.current_question:
                if st.button("RESET"):
                    reset_quiz()
                    st.rerun()


def render_module1():
    """Render the A/B Testing (Discriminator) module."""
    st.header("COMPARE MODE")
    st.write("Identify the superior prompt. The differences are subtle — stay sharp.")

    # Progress bar
    progress = st.session_state.quiz_index / QUIZ_LENGTH
    st.progress(progress, text=f"Question {st.session_state.quiz_index + 1} of {QUIZ_LENGTH}")

    # Check if quiz is complete
    if st.session_state.quiz_index >= QUIZ_LENGTH:
        st.balloons()
        st.success(f"TRAINING COMPLETE")
        st.write(f"**FINAL SCORE: {st.session_state.quiz_score}/{QUIZ_LENGTH}**")

        if st.session_state.quiz_score >= 7:
            st.success("Strong performance. You've got a sharp eye for effective prompts.")
        elif st.session_state.quiz_score >= 5:
            st.info("Solid foundation. Review the explanations to sharpen your skills further.")
        else:
            st.warning("Keep training. Study the explanations to understand what makes prompts effective.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("RETAKE", use_container_width=True):
                reset_quiz()
                st.rerun()
        with col2:
            if st.button("TRY CHALLENGE MODE", type="primary", use_container_width=True):
                st.session_state.lesson_selected = "challenge"
                st.rerun()
        return

    # Generate new question button
    if st.session_state.current_question is None:
        st.info("Ready to test your skills? Generate a new comparison.")
        if st.button("GENERATE QUESTION", type="primary", use_container_width=True):
            with st.spinner("Generating question..."):
                question = generate_ab_question()
                st.session_state.current_question = question
                # Randomly decide which side gets the strong prompt
                st.session_state.correct_side = random.choice(["A", "B"])
            st.rerun()
        return

    # Display the question
    question = st.session_state.current_question
    correct_side = st.session_state.correct_side

    st.subheader("SCENARIO")
    st.info(question["scenario"])

    # Assign prompts to sides based on randomization
    if correct_side == "A":
        option_a = question["strong_prompt"]
        option_b = question["weak_prompt"]
    else:
        option_a = question["weak_prompt"]
        option_b = question["strong_prompt"]

    # Display options
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("OPTION A")
        st.text_area("", value=option_a, height=150, disabled=True, key="option_a_display", label_visibility="collapsed")

    with col2:
        st.subheader("OPTION B")
        st.text_area("", value=option_b, height=150, disabled=True, key="option_b_display", label_visibility="collapsed")

    # Answer buttons or feedback
    if not st.session_state.show_feedback:
        st.write("**Which prompt will get better results?**")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("SELECT A", use_container_width=True):
                handle_answer("A")
                st.rerun()

        with col2:
            if st.button("SELECT B", use_container_width=True):
                handle_answer("B")
                st.rerun()
    else:
        # Show feedback
        if st.session_state.last_choice_correct:
            st.success("CORRECT — Nice work.")
        else:
            st.error("INCORRECT — Study the explanation below.")

        st.subheader("ANALYSIS")
        st.write(st.session_state.last_explanation)

        # Show which was the better prompt
        st.subheader("THE SUPERIOR PROMPT:")
        st.code(question["strong_prompt"], language=None)

        if st.button("NEXT QUESTION", type="primary"):
            advance_quiz()
            st.rerun()


def render_module2():
    """Render the Active Grading (Generator) module."""
    st.header("CHALLENGE MODE")
    st.write("Write prompts for real-world scenarios. Get scored. Get better.")

    if not st.session_state.api_validated:
        st.warning("Please validate your API key in the sidebar to use this module.")
        return

    if st.session_state.demo_mode:
        st.info("**Demo Mode** — Generate a challenge and submit any prompt to see sample grading.")

    # Generate challenge button
    if st.session_state.challenge_scenario is None:
        st.info("Ready when you are. Generate your challenge below.")
        if st.button("GENERATE CHALLENGE", type="primary", use_container_width=True):
            with st.spinner("Generating challenge..."):
                challenge = generate_challenge()
                st.session_state.challenge_scenario = challenge
                st.session_state.challenge_result = None
                st.session_state.challenge_graded = False
            st.rerun()
        return

    # Display the challenge
    challenge = st.session_state.challenge_scenario

    st.subheader(f"CHALLENGE: {challenge['title'].upper()}")
    st.info(challenge["scenario"])

    # Show key elements to consider
    with st.expander("HINT: KEY ELEMENTS TO CONSIDER"):
        for i, element in enumerate(challenge.get("key_elements", []), 1):
            st.write(f"{i}. {element}")

    # User prompt input
    user_prompt = st.text_area(
        "YOUR PROMPT:",
        height=200,
        placeholder="Write your prompt here...",
        key="user_prompt_input"
    )

    # Submit for grading
    col1, col2 = st.columns([1, 1])

    with col1:
        submit_disabled = not user_prompt or st.session_state.challenge_graded
        if st.button("SUBMIT FOR GRADING", type="primary", disabled=submit_disabled, use_container_width=True):
            with st.spinner("Analyzing your prompt..."):
                result = grade_prompt(user_prompt, challenge["scenario"])
                st.session_state.challenge_result = result
                st.session_state.challenge_graded = True
            st.rerun()

    with col2:
        # Next challenge button - only enabled after grading
        next_disabled = not st.session_state.challenge_graded
        if st.button("NEXT CHALLENGE", disabled=next_disabled, use_container_width=True):
            st.session_state.challenge_scenario = None
            st.session_state.challenge_result = None
            st.session_state.challenge_graded = False
            st.rerun()

    if not st.session_state.challenge_graded:
        st.caption("Submit your prompt for grading before moving to the next challenge.")

    # Display grading result
    if st.session_state.challenge_result:
        st.divider()
        st.subheader("RESULTS")

        result = st.session_state.challenge_result

        # Try to extract score for color coding
        try:
            if "TOTAL:" in result:
                score_part = result.split("TOTAL:")[1].split("/")[0].strip()
                score = int(''.join(filter(str.isdigit, score_part)))

                if score >= 16:
                    st.success(f"**Score: {score}/20** - Excellent!")
                elif score >= 12:
                    st.info(f"**Score: {score}/20** - Good effort!")
                else:
                    st.warning(f"**Score: {score}/20** - Keep practicing!")
        except:
            pass

        st.markdown(result)


def render_module3():
    """Render the Grade My Prompt module for testing user's own prompts."""
    st.header("ANALYZE MODE")
    st.write("Paste any prompt. Get a detailed breakdown. Deploy with confidence.")

    if not st.session_state.api_validated:
        st.warning("Please validate your API key in the sidebar to use this module.")
        return

    if st.session_state.demo_mode:
        st.info("**Demo Mode** — Submit any prompt to see sample analysis.")

    # Information about the grading
    with st.expander("HOW SCORING WORKS"):
        st.markdown("""
        Your prompt is evaluated on **4 key metrics** (each scored 1-5):

        **CLARITY** — Is the intent immediately clear? Any ambiguity?

        **SPECIFICITY** — Enough detail? Concrete requirements?

        **CONSTRAINTS** — Appropriate boundaries? (format, length, tone, scope)

        **CONTEXT** — Enough background? (role, audience, purpose)

        ---

        **You'll receive:** Score breakdown • Strengths • Improvements • Optimized version
        """)

    # User prompt input with form for immediate submission
    if not st.session_state.grade_my_prompt_graded:
        with st.form(key="analyze_form"):
            user_prompt = st.text_area(
                "YOUR PROMPT:",
                height=250,
                placeholder="Paste the prompt you want to analyze...",
                key="grade_my_prompt_input"
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                submitted = st.form_submit_button("ANALYZE", type="primary", use_container_width=True)
            with col2:
                st.form_submit_button("CLEAR", use_container_width=True, disabled=True)

            if submitted and user_prompt:
                with st.spinner("Running analysis..."):
                    result = grade_general_prompt(user_prompt)
                    st.session_state.grade_my_prompt_result = result
                    st.session_state.grade_my_prompt_graded = True

                    # Extract score and save to history
                    score = None
                    try:
                        if "TOTAL:" in result:
                            score_part = result.split("TOTAL:")[1].split("/")[0].strip()
                            score = int(''.join(filter(str.isdigit, score_part)))
                    except:
                        pass

                    # Add to history
                    history_entry = {
                        "prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
                        "full_prompt": user_prompt,
                        "result": result,
                        "score": score,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    }
                    st.session_state.analysis_history.insert(0, history_entry)
                    # Keep only last 10 entries
                    st.session_state.analysis_history = st.session_state.analysis_history[:10]

                st.rerun()
            elif submitted and not user_prompt:
                st.warning("Please enter a prompt to analyze.")
    else:
        # Show new analysis button when graded - prominent at top
        if st.button("🔄 NEW ANALYSIS", type="primary", use_container_width=True):
            st.session_state.grade_my_prompt_result = None
            st.session_state.grade_my_prompt_graded = False
            st.rerun()

    # Display grading result
    if st.session_state.grade_my_prompt_result:
        st.divider()
        st.subheader("ANALYSIS RESULTS")

        result = st.session_state.grade_my_prompt_result

        # Try to extract score for color coding
        try:
            if "TOTAL:" in result:
                score_part = result.split("TOTAL:")[1].split("/")[0].strip()
                score = int(''.join(filter(str.isdigit, score_part)))

                if score >= 16:
                    st.success(f"**SCORE: {score}/20** — Excellent. Ready to deploy.")
                elif score >= 12:
                    st.info(f"**SCORE: {score}/20** — Good foundation. See improvements below.")
                else:
                    st.warning(f"**SCORE: {score}/20** — Needs work. Review the suggestions below.")
        except:
            pass

        # Try to extract the improved prompt for separate display with copy button
        improved_prompt = None
        display_result = result
        try:
            if "## Improved Version" in result:
                parts = result.split("## Improved Version")
                before_improved = parts[0]
                after_improved = parts[1]

                # Extract the code block content
                if "```" in after_improved:
                    code_start = after_improved.find("```") + 3
                    # Skip language identifier if present (e.g., ```text)
                    newline_after_backticks = after_improved.find("\n", code_start)
                    code_start = newline_after_backticks + 1
                    code_end = after_improved.find("```", code_start)
                    if code_end > code_start:
                        improved_prompt = after_improved[code_start:code_end].strip()
                        # Get content after the code block for "Why This Works Better"
                        after_code = after_improved[code_end + 3:]
                        display_result = before_improved
        except:
            pass

        # Display the main analysis (without improved version section)
        st.markdown(display_result)

        # Display improved prompt with copy functionality
        if improved_prompt:
            st.markdown("---")
            st.markdown("#### IMPROVED VERSION")

            # Store improved prompt in session state for copying
            st.session_state.improved_prompt_text = improved_prompt

            # Display in text area with word wrap
            st.text_area(
                "Improved prompt:",
                value=improved_prompt,
                height=200,
                key="improved_prompt_display",
                label_visibility="collapsed",
                disabled=True
            )

            # Copy button using components.html for working clipboard
            col1, col2 = st.columns([1, 3])
            with col1:
                # Escape the prompt for JavaScript
                escaped_prompt = improved_prompt.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$").replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")

                # Create a copy button using HTML/JS that actually works
                copy_button_html = f'''
                <button onclick="copyToClipboard()" style="
                    background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
                    border: 2px solid #00d4ff;
                    border-radius: 8px;
                    color: #00d4ff;
                    padding: 0.6rem 1.5rem;
                    font-weight: 600;
                    font-size: 0.85rem;
                    cursor: pointer;
                    width: 100%;
                    text-transform: uppercase;
                    letter-spacing: 0.02em;
                    transition: all 0.3s ease;
                " onmouseover="this.style.background='rgba(0, 212, 255, 0.2)'" onmouseout="this.style.background='linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%)'">
                    📋 COPY
                </button>
                <span id="copyStatus" style="color: #00c896; font-size: 0.8rem; margin-left: 10px; display: none;">✓ Copied!</span>
                <script>
                function copyToClipboard() {{
                    const text = `{escaped_prompt}`;
                    navigator.clipboard.writeText(text).then(function() {{
                        document.getElementById('copyStatus').style.display = 'inline';
                        setTimeout(function() {{
                            document.getElementById('copyStatus').style.display = 'none';
                        }}, 2000);
                    }});
                }}
                </script>
                '''
                components.html(copy_button_html, height=50)

            # Why this works better section
            if "## Why This Works Better" in result:
                why_section = result.split("## Why This Works Better")[1].split("---")[0].strip()
                st.markdown("#### WHY THIS WORKS BETTER")
                st.markdown(why_section)

        # Helpful tip at the bottom
        st.divider()
        st.caption("PRO TIP: Copy the improved version and analyze it again to verify the score increase.")

    # History section
    if st.session_state.analysis_history:
        st.divider()
        with st.expander(f"HISTORY ({len(st.session_state.analysis_history)} analyses)", expanded=False):
            for i, entry in enumerate(st.session_state.analysis_history):
                score_display = f"**{entry['score']}/20**" if entry['score'] else "N/A"
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"*{entry['timestamp']}* — {entry['prompt'][:80]}...")
                with col2:
                    st.markdown(score_display)
                with col3:
                    if st.button("VIEW", key=f"view_history_{i}", use_container_width=True):
                        st.session_state.grade_my_prompt_result = entry['result']
                        st.session_state.grade_my_prompt_graded = True
                        st.rerun()

                if i < len(st.session_state.analysis_history) - 1:
                    st.markdown("---")

            # Clear history button
            st.markdown("")
            if st.button("CLEAR HISTORY", use_container_width=True):
                st.session_state.analysis_history = []
                st.rerun()


def render_lesson_selection():
    """Render the lesson selection screen."""
    st.markdown("### SELECT YOUR TRAINING MODE")
    st.markdown("---")

    # Training modes in a clean 3-column grid
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon">🔍</div>
            <div class="mode-title">COMPARE</div>
            <div class="mode-desc">Spot the difference. Analyze two prompts side-by-side and identify which one will get better results.</div>
            <div class="mode-difficulty beginner">BEGINNER</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("START TRAINING", type="primary", use_container_width=True, key="select_compare"):
            st.session_state.lesson_selected = "compare"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon">⚔️</div>
            <div class="mode-title">CHALLENGE</div>
            <div class="mode-desc">Prove yourself. Write prompts for real-world scenarios and get graded by AI.</div>
            <div class="mode-difficulty intermediate">INTERMEDIATE</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("ACCEPT CHALLENGE", type="primary", use_container_width=True, key="select_challenge"):
            st.session_state.lesson_selected = "challenge"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon">📊</div>
            <div class="mode-title">ANALYZE</div>
            <div class="mode-desc">Test before you deploy. Paste any prompt and get a detailed breakdown before using it.</div>
            <div class="mode-difficulty all-levels">ALL LEVELS</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("ANALYZE PROMPT", type="primary", use_container_width=True, key="select_grade"):
            st.session_state.lesson_selected = "grade"
            st.rerun()


def render_main_content():
    """Render the main content area with tabs."""
    if not st.session_state.api_validated:
        st.warning("Connect your API to begin training.")

        # Welcome section with structured layout
        st.markdown("### LEVEL UP YOUR PROMPT GAME")
        st.markdown("Stop getting mediocre AI responses. Master the craft of prompt engineering.")
        st.markdown("---")

        # Feature overview in columns
        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            st.markdown("""
            <div class="feature-box">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">COMPARE</div>
                <div class="feature-desc">Train your eye to spot the difference between good and bad prompts</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-box">
                <div class="feature-icon">⚔️</div>
                <div class="feature-title">CHALLENGE</div>
                <div class="feature-desc">Write prompts under pressure and get scored on your performance</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="feature-box">
                <div class="feature-icon">📊</div>
                <div class="feature-title">ANALYZE</div>
                <div class="feature-desc">Test any prompt before you use it and get actionable improvements</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Quick start guide
        st.markdown("""
        <div class="quick-start">
            <div class="quick-start-title">QUICK START</div>
            <div class="quick-start-steps">
                <span class="step">1</span> Select your API provider in the sidebar
                <span class="step">2</span> Enter your API key
                <span class="step">3</span> Hit "Connect"
            </div>
            <div class="quick-start-note">No API key? Enable Demo Mode to try it out.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Show lesson selection if not yet chosen
    if st.session_state.lesson_selected is None:
        render_lesson_selection()
        return

    # Render the selected lesson
    if st.session_state.lesson_selected == "compare":
        render_module1()
    elif st.session_state.lesson_selected == "challenge":
        render_module2()
    elif st.session_state.lesson_selected == "grade":
        render_module3()


# =============================================================================
# Custom Styling
# =============================================================================

CUSTOM_CSS = """
<style>
    /* Import sharp, modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* FORCE DARK MODE - Override any light theme settings */
    :root {
        color-scheme: dark !important;
    }

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Force dark backgrounds on main containers */
    .stApp {
        background-color: #0a0a0f !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #0a0a0f !important;
    }

    [data-testid="stHeader"] {
        background-color: #0a0a0f !important;
    }

    .main, [data-testid="stMain"] {
        background-color: #0a0a0f !important;
    }

    /* Main content text - be specific to avoid breaking buttons */
    .main p, .main span:not([class*="st"]), .main label {
        color: #e0e0e0 !important;
    }

    /* Markdown text */
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li {
        color: #e0e0e0 !important;
    }

    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        background-color: #0a0a0f !important;
    }

    /* Header styling */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 50%, #006699 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding-bottom: 0.25rem !important;
    }

    h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        color: #e0e0e0 !important;
    }

    h4 {
        font-weight: 600 !important;
        color: #00d4ff !important;
        text-transform: uppercase;
        font-size: 0.9rem !important;
        letter-spacing: 0.05em !important;
    }

    /* Subheader caption */
    .main [data-testid="stCaptionContainer"] {
        font-size: 1.1rem;
        color: #888 !important;
        font-weight: 500;
        letter-spacing: 0.02em;
    }

    /* Mode selection cards */
    .mode-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }

    .mode-card:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 25px rgba(0, 212, 255, 0.2);
        transform: translateY(-3px);
    }

    .mode-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
    }

    .mode-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #00d4ff;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }

    .mode-desc {
        font-size: 0.9rem;
        color: #a0a0a0;
        line-height: 1.5;
        flex-grow: 1;
        margin-bottom: 1rem;
    }

    .mode-difficulty {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        padding: 0.35rem 0.75rem;
        border-radius: 4px;
        display: inline-block;
    }

    .mode-difficulty.beginner {
        background: rgba(0, 200, 150, 0.2);
        color: #00c896;
        border: 1px solid rgba(0, 200, 150, 0.3);
    }

    .mode-difficulty.intermediate {
        background: rgba(255, 180, 0, 0.2);
        color: #ffb400;
        border: 1px solid rgba(255, 180, 0, 0.3);
    }

    .mode-difficulty.all-levels {
        background: rgba(0, 212, 255, 0.2);
        color: #00d4ff;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }

    /* Feature boxes for welcome screen */
    .feature-box {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        min-height: 150px;
        transition: all 0.3s ease;
    }

    .feature-box:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.15);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .feature-title {
        font-size: 1rem;
        font-weight: 700;
        color: #00d4ff;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .feature-desc {
        font-size: 0.85rem;
        color: #888;
        line-height: 1.4;
    }

    /* Quick start section */
    .quick-start {
        background: linear-gradient(145deg, #0f0f1a 0%, #1a1a2e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }

    .quick-start-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #00d4ff;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }

    .quick-start-steps {
        font-size: 0.9rem;
        color: #a0a0a0;
        margin-bottom: 1rem;
    }

    .quick-start-steps .step {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        background: #00d4ff;
        color: #0a0a0f;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.75rem;
        margin: 0 0.5rem 0 1rem;
    }

    .quick-start-steps .step:first-child {
        margin-left: 0;
    }

    .quick-start-note {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
        color: #0a0a0f !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        font-size: 0.85rem !important;
    }

    /* Ensure button text is dark */
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] div {
        color: #0a0a0f !important;
        background: transparent !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #00e5ff 0%, #00b8d4 100%) !important;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary buttons */
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]) {
        background: transparent !important;
        color: #00d4ff !important;
        border: 2px solid #0f3460 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        font-size: 0.85rem !important;
    }

    /* Secondary button text */
    .stButton > button[kind="secondary"] p,
    .stButton > button[kind="secondary"] span,
    .stButton > button:not([kind="primary"]) p,
    .stButton > button:not([kind="primary"]) span {
        color: #00d4ff !important;
    }

    .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover {
        border-color: #00d4ff !important;
        background: rgba(0, 212, 255, 0.1) !important;
    }

    /* Disabled buttons */
    .stButton > button:disabled {
        opacity: 0.4 !important;
        cursor: not-allowed !important;
    }

    /* Text inputs and text areas */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #0a0a0f !important;
        border: 2px solid #1a1a2e !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
        font-family: 'Inter', monospace !important;
        transition: all 0.3s ease !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 1px rgba(0, 212, 255, 0.3) !important;
    }

    /* Placeholder text */
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #444 !important;
    }

    /* Improved prompt display area - read-only styling with word wrap */
    .stTextArea[data-testid="stTextArea"] textarea[disabled],
    .stTextArea[data-testid="stTextArea"] textarea:disabled,
    .stTextArea textarea:read-only {
        background-color: #1a1a2e !important;
        border: 2px solid #0f3460 !important;
        border-radius: 8px !important;
        color: #00d4ff !important;
        cursor: text !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
        line-height: 1.6 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
    }

    /* Copy button styling */
    button[kind="secondary"]:has(span:contains("COPY")),
    .stButton > button:contains("COPY") {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%) !important;
        border: 2px solid #00d4ff !important;
    }

    /* Radio buttons */
    .stRadio > div {
        background: transparent !important;
    }

    .stRadio > div > label {
        color: #888 !important;
        font-weight: 500 !important;
    }

    [data-testid="stRadio"] > div > div > label > div:first-child {
        background-color: #1a1a2e !important;
        border-color: #0f3460 !important;
    }

    [data-testid="stRadio"] > div > div > label[data-checked="true"] > div:first-child {
        background-color: #00d4ff !important;
        border-color: #00d4ff !important;
    }

    /* Checkboxes */
    .stCheckbox > label > div:first-child {
        background-color: #1a1a2e !important;
        border-color: #0f3460 !important;
        border-radius: 4px !important;
    }

    .stCheckbox > label > div[data-checked="true"]:first-child {
        background-color: #00d4ff !important;
        border-color: #00d4ff !important;
    }

    /* Info boxes - force dark backgrounds */
    .stAlert, .stAlert > div, [data-testid="stAlert"] {
        border-radius: 8px !important;
        border: none !important;
        background-color: #1a1a2e !important;
    }

    /* Success alert */
    [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]),
    .stAlert:has(.stSuccess) {
        background: linear-gradient(135deg, rgba(0, 200, 150, 0.2) 0%, rgba(0, 150, 100, 0.15) 100%) !important;
        border-left: 4px solid #00c896 !important;
    }

    /* Info alert */
    [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]),
    .stAlert:has(.stInfo) {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 150, 200, 0.15) 100%) !important;
        border-left: 4px solid #00d4ff !important;
    }

    /* Warning alert */
    [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]),
    .stAlert:has(.stWarning) {
        background: linear-gradient(135deg, rgba(255, 180, 0, 0.2) 0%, rgba(200, 140, 0, 0.15) 100%) !important;
        border-left: 4px solid #ffb400 !important;
    }

    /* Error alert */
    [data-testid="stAlert"]:has([data-testid="stAlertContentError"]),
    .stAlert:has(.stError) {
        background: linear-gradient(135deg, rgba(255, 80, 80, 0.2) 0%, rgba(200, 50, 50, 0.15) 100%) !important;
        border-left: 4px solid #ff5050 !important;
    }

    /* Alert text colors */
    [data-testid="stAlert"] p, [data-testid="stAlert"] span {
        color: #e0e0e0 !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
        border-radius: 4px !important;
    }

    .stProgress > div {
        background-color: #1a1a2e !important;
        border-radius: 4px !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a2e !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #00d4ff !important;
    }

    .streamlit-expanderContent {
        background-color: #0f0f1a !important;
        border: 1px solid #1a1a2e !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0f 0%, #0f0f1a 100%) !important;
        border-right: 1px solid #1a1a2e !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e0e0e0 !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        width: 100% !important;
    }

    /* Link buttons in sidebar */
    [data-testid="stSidebar"] a[data-testid="stBaseLinkButton-secondary"] {
        background: transparent !important;
        color: #00d4ff !important;
        border: 1px solid #0f3460 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.03em !important;
        padding: 0.5rem 0.75rem !important;
        transition: all 0.3s ease !important;
        text-decoration: none !important;
        display: block !important;
        text-align: center !important;
    }

    [data-testid="stSidebar"] a[data-testid="stBaseLinkButton-secondary"]:hover {
        border-color: #00d4ff !important;
        background: rgba(0, 212, 255, 0.1) !important;
    }

    /* Divider */
    hr {
        border-color: #1a1a2e !important;
        margin: 1.5rem 0 !important;
    }

    /* Code blocks */
    .stCodeBlock {
        background-color: #0a0a0f !important;
        border: 1px solid #1a1a2e !important;
        border-radius: 8px !important;
    }

    code {
        color: #00d4ff !important;
    }

    /* Tables */
    .stTable {
        background-color: #0f0f1a !important;
    }

    table {
        border-collapse: separate !important;
        border-spacing: 0 !important;
    }

    th {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: #00d4ff !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em !important;
        padding: 0.75rem 1rem !important;
        border-bottom: 2px solid #0f3460 !important;
    }

    td {
        background-color: #0f0f1a !important;
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid #1a1a2e !important;
    }

    /* Markdown tables */
    .stMarkdown table {
        width: 100% !important;
        margin: 1rem 0 !important;
    }

    .stMarkdown th, .stMarkdown td {
        text-align: left !important;
    }

    /* Caption text */
    .stCaption, small {
        color: #666 !important;
        font-size: 0.85rem !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: #00d4ff transparent transparent transparent !important;
    }

    /* Score badges - custom styling */
    .score-excellent {
        background: linear-gradient(135deg, #00c896 0%, #00a67d 100%);
        color: #0a0a0f;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }

    .score-good {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: #0a0a0f;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }

    .score-practice {
        background: linear-gradient(135deg, #ffb400 0%, #cc9000 100%);
        color: #0a0a0f;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Metric cards styling */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem;
    }

    [data-testid="metric-container"] label {
        color: #00d4ff !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e0e0e0 !important;
        font-weight: 700 !important;
    }
</style>
"""


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Prompt Dojo",
        page_icon="🥋",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("PROMPT DOJO")
    st.caption("MASTER THE ART OF PROMPT ENGINEERING")

    init_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
