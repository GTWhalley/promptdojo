"""
Prompt Dojo - Prompt Engineering Trainer
A lightweight application that trains users in prompt engineering through
active A/B testing and interactive grading.
"""

import streamlit as st
import json
import random
import openai
import google.generativeai as genai


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
            return True, "Connection successful!"
        else:  # Gemini
            genai.configure(api_key=api_key)
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if not available_models:
                return False, "No compatible Gemini models found for this API key."
            model_name = next((m for m in available_models if 'flash' in m.lower()), available_models[0])
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say 'OK'")
            st.session_state.gemini_model = model_name
            return True, f"Connection successful! Using {model_name}"
    except openai.AuthenticationError:
        return False, "Invalid OpenAI API key."
    except openai.APIError as e:
        return False, f"OpenAI API error: {str(e)}"
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
# UI Components
# =============================================================================

def render_sidebar():
    """Render the settings sidebar."""
    with st.sidebar:
        st.header("Settings")

        provider = st.radio(
            "Select API Provider:",
            ["OpenAI", "Gemini"],
            index=0 if st.session_state.api_provider == "OpenAI" else 1,
            disabled=st.session_state.demo_mode
        )
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

            if st.button("Test Connection", type="primary"):
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
            st.subheader("Current Lesson")
            lesson_names = {
                "compare": "Compare Prompts",
                "challenge": "Test Your Skills",
                "grade": "Grade My Prompt"
            }
            lesson_name = lesson_names.get(st.session_state.lesson_selected, "Unknown")
            st.write(f"**{lesson_name}**")

            if st.button("Change Lesson"):
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
            st.subheader("Progress")
            st.write(f"Quiz Score: {st.session_state.quiz_score}/{QUIZ_LENGTH}")
            st.write(f"Questions Completed: {st.session_state.quiz_index}/{QUIZ_LENGTH}")

            if st.session_state.quiz_index > 0 or st.session_state.current_question:
                if st.button("Reset Progress"):
                    reset_quiz()
                    st.rerun()


def render_module1():
    """Render the A/B Testing (Discriminator) module."""
    st.header("Compare Prompts")
    st.write("Identify the better prompt - but be careful, the differences are subtle!")

    # Progress bar
    progress = st.session_state.quiz_index / QUIZ_LENGTH
    st.progress(progress, text=f"Question {st.session_state.quiz_index + 1} of {QUIZ_LENGTH}")

    # Check if quiz is complete
    if st.session_state.quiz_index >= QUIZ_LENGTH:
        st.balloons()
        st.success(f"Congratulations! You completed the quiz!")
        st.write(f"**Final Score: {st.session_state.quiz_score}/{QUIZ_LENGTH}**")

        if st.session_state.quiz_score >= 7:
            st.success("Excellent work! You've demonstrated strong prompt discrimination skills.")
        elif st.session_state.quiz_score >= 5:
            st.info("Good effort! Consider reviewing the explanations to improve further.")
        else:
            st.warning("Keep practicing! Review the explanations to understand what makes a prompt effective.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Retake Quiz", use_container_width=True):
                reset_quiz()
                st.rerun()
        with col2:
            if st.button("Try Challenge Mode", type="primary", use_container_width=True):
                st.session_state.lesson_selected = "challenge"
                st.rerun()
        return

    # Generate new question button
    if st.session_state.current_question is None:
        st.info("Click the button below to generate a new comparison question.")
        if st.button("Generate Question", type="primary", use_container_width=True):
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

    st.subheader("Scenario")
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
        st.subheader("Option A")
        st.text_area("", value=option_a, height=150, disabled=True, key="option_a_display", label_visibility="collapsed")

    with col2:
        st.subheader("Option B")
        st.text_area("", value=option_b, height=150, disabled=True, key="option_b_display", label_visibility="collapsed")

    # Answer buttons or feedback
    if not st.session_state.show_feedback:
        st.write("**Which prompt is better?**")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Choose Option A", use_container_width=True):
                handle_answer("A")
                st.rerun()

        with col2:
            if st.button("Choose Option B", use_container_width=True):
                handle_answer("B")
                st.rerun()
    else:
        # Show feedback
        if st.session_state.last_choice_correct:
            st.success("Correct!")
        else:
            st.error("Incorrect")

        st.subheader("Why?")
        st.write(st.session_state.last_explanation)

        # Show which was the better prompt
        st.subheader("The Better Prompt Was:")
        st.code(question["strong_prompt"], language=None)

        if st.button("Next Question", type="primary"):
            advance_quiz()
            st.rerun()


def render_module2():
    """Render the Active Grading (Generator) module."""
    st.header("Test Your Skills")
    st.write("Write prompts for challenging scenarios and get detailed AI feedback.")

    if not st.session_state.api_validated:
        st.warning("Please validate your API key in the sidebar to use this module.")
        return

    if st.session_state.demo_mode:
        st.info("**Demo Mode**: Generate a challenge and submit any prompt to see sample grading.")

    # Generate challenge button
    if st.session_state.challenge_scenario is None:
        st.info("Click the button below to generate a new challenge scenario.")
        if st.button("Generate Challenge", type="primary", use_container_width=True):
            with st.spinner("Generating challenge..."):
                challenge = generate_challenge()
                st.session_state.challenge_scenario = challenge
                st.session_state.challenge_result = None
                st.session_state.challenge_graded = False
            st.rerun()
        return

    # Display the challenge
    challenge = st.session_state.challenge_scenario

    st.subheader(f"Challenge: {challenge['title']}")
    st.info(challenge["scenario"])

    # Show key elements to consider
    with st.expander("Hint: Key Elements to Consider"):
        for i, element in enumerate(challenge.get("key_elements", []), 1):
            st.write(f"{i}. {element}")

    # User prompt input
    user_prompt = st.text_area(
        "Write your prompt:",
        height=200,
        placeholder="Enter your carefully crafted prompt here...",
        key="user_prompt_input"
    )

    # Submit for grading
    col1, col2 = st.columns([1, 1])

    with col1:
        submit_disabled = not user_prompt or st.session_state.challenge_graded
        if st.button("Submit for Grading", type="primary", disabled=submit_disabled, use_container_width=True):
            with st.spinner("Grading your prompt..."):
                result = grade_prompt(user_prompt, challenge["scenario"])
                st.session_state.challenge_result = result
                st.session_state.challenge_graded = True
            st.rerun()

    with col2:
        # Next challenge button - only enabled after grading
        next_disabled = not st.session_state.challenge_graded
        if st.button("Next Challenge", disabled=next_disabled, use_container_width=True):
            st.session_state.challenge_scenario = None
            st.session_state.challenge_result = None
            st.session_state.challenge_graded = False
            st.rerun()

    if not st.session_state.challenge_graded:
        st.caption("You must submit your prompt for grading before moving to the next challenge.")

    # Display grading result
    if st.session_state.challenge_result:
        st.divider()
        st.subheader("Grading Result")

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
    st.header("Grade My Prompt")
    st.write("Paste any prompt you're planning to use and get detailed feedback before you use it.")

    if not st.session_state.api_validated:
        st.warning("Please validate your API key in the sidebar to use this module.")
        return

    if st.session_state.demo_mode:
        st.info("**Demo Mode**: Submit any prompt to see sample grading feedback.")

    # Information about the grading
    with st.expander("How does grading work?"):
        st.markdown("""
        Your prompt will be evaluated on **4 key metrics** (each scored 1-5):

        1. **Clarity** - Is the intent immediately clear? Is there any ambiguity?
        2. **Specificity** - Does it provide enough detail? Are there concrete requirements?
        3. **Constraints** - Are there appropriate boundaries? (format, length, tone, scope)
        4. **Context** - Does it give enough background? (role, audience, purpose)

        You'll receive:
        - A detailed score breakdown
        - What you did well
        - Areas for improvement
        - An improved version of your prompt
        """)

    # User prompt input
    user_prompt = st.text_area(
        "Paste your prompt here:",
        height=250,
        placeholder="Enter the prompt you want to evaluate...\n\nFor example:\n'Write me a blog post about AI'",
        key="grade_my_prompt_input"
    )

    # Action buttons
    col1, col2 = st.columns([1, 1])

    with col1:
        submit_disabled = not user_prompt or st.session_state.grade_my_prompt_graded
        if st.button("Grade My Prompt", type="primary", disabled=submit_disabled, use_container_width=True):
            with st.spinner("Analyzing your prompt..."):
                result = grade_general_prompt(user_prompt)
                st.session_state.grade_my_prompt_result = result
                st.session_state.grade_my_prompt_graded = True
            st.rerun()

    with col2:
        # Clear button to grade another prompt
        if st.button("Grade Another Prompt", disabled=not st.session_state.grade_my_prompt_graded, use_container_width=True):
            st.session_state.grade_my_prompt_result = None
            st.session_state.grade_my_prompt_graded = False
            st.rerun()

    # Display grading result
    if st.session_state.grade_my_prompt_result:
        st.divider()
        st.subheader("Grading Result")

        result = st.session_state.grade_my_prompt_result

        # Try to extract score for color coding
        try:
            if "TOTAL:" in result:
                score_part = result.split("TOTAL:")[1].split("/")[0].strip()
                score = int(''.join(filter(str.isdigit, score_part)))

                if score >= 16:
                    st.success(f"**Score: {score}/20** - Excellent! Your prompt is well-crafted.")
                elif score >= 12:
                    st.info(f"**Score: {score}/20** - Good start! See suggestions below to improve.")
                else:
                    st.warning(f"**Score: {score}/20** - Consider the improvements below before using this prompt.")
        except:
            pass

        st.markdown(result)

        # Helpful tip at the bottom
        st.divider()
        st.caption("Tip: Copy the improved version and paste it back above to see if your score improves!")


def render_lesson_selection():
    """Render the lesson selection screen."""
    st.markdown("### Choose Your Training Mode")
    st.write("Select how you want to practice prompt engineering today:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Compare Prompts")
        st.write("Learn by comparing two prompts side-by-side and identifying which one is better crafted.")
        st.caption("Best for: Beginners learning to recognize good prompts")
        if st.button("Start Comparing", type="primary", use_container_width=True, key="select_compare"):
            st.session_state.lesson_selected = "compare"
            st.rerun()

    with col2:
        st.markdown("#### Test Your Skills")
        st.write("Write your own prompts for challenging scenarios and get detailed AI feedback.")
        st.caption("Best for: Practicing and improving your prompt writing")
        if st.button("Start Challenge", type="primary", use_container_width=True, key="select_challenge"):
            st.session_state.lesson_selected = "challenge"
            st.rerun()

    with col3:
        st.markdown("#### Grade My Prompt")
        st.write("Paste any prompt you're planning to use and get detailed feedback before using it.")
        st.caption("Best for: Testing real prompts before you use them")
        if st.button("Grade a Prompt", type="primary", use_container_width=True, key="select_grade"):
            st.session_state.lesson_selected = "grade"
            st.rerun()


def render_main_content():
    """Render the main content area with tabs."""
    if not st.session_state.api_validated:
        st.warning("Please enter and validate your API key in the sidebar to begin training.")
        st.markdown("""
        ### Welcome to Prompt Dojo!

        This application will train you in the art of prompt engineering through:

        1. **Compare Prompts** - Learn to identify effective prompts through A/B testing
        2. **Test Your Skills** - Write your own prompts and receive AI-powered feedback
        3. **Grade My Prompt** - Test your real prompts before using them and get improvement suggestions

        To get started:
        1. Select your API provider (OpenAI or Gemini)
        2. Enter your API key
        3. Click "Test Connection"

        Or check "Demo Mode" to try without an API key!
        """)
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
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Prompt Dojo",
        page_icon="🥋",
        layout="wide"
    )

    st.title("Prompt Dojo")
    st.caption("Master the Art of Prompt Engineering")

    init_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
