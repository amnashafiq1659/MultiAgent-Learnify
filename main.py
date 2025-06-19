import os
import chainlit as cl
from dotenv import load_dotenv 
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
from openai.types.responses import ResponseTextDeltaEvent

# Load environment variables from .env file
load_dotenv()

# Initialize the agent with the provided configuration
external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Create an instance of the OpenAIChatCompletionsModel with the external client
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-001",
    openai_client=external_client
)

# Create a RunConfig instance with the model and external client
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Define the agents with their specific instructions
study_guide_agent = Agent(
    name="Study Guide Agent",
    instructions="Explain any topic in simple and easy-to-understand way.",
    model=model,
    handoff_description="handoff to Study Guide Agent if the task is to explain a topic."
)

note_maker_agent = Agent(
    name="Note Maker Agent",
    instructions="Create short notes or a summary for the requested topic.",
    model=model,
    handoff_description="handoff to Note Maker Agent if the task is to make notes or summaries."
)

quiz_maker_agent = Agent(
    name="Quiz Maker Agent",
    instructions="Generate 5 multiple choice questions from the given topic also tell the correct answer.",
    model=model,
    handoff_description="handoff to Quiz Maker Agent if the task is to make quiz ."
)

question_answer_agent = Agent(
    name="Question Answer Agent",
    instructions="Generate 5 questions with clear, concise answers for the given topic. Help the student revise and prepare with Q&A.",
    model=model,
    handoff_description="handoff to Question Answer Agent if the task is to generate Q&A."
)

resource_finder_agent = Agent(
    name="Resource Finder Agent",
    instructions="Suggest images, videos, articles, or books for the given topic using reliable educational sources.",
    model=model,
    handoff_description="handoff to Resource Finder Agent if the task is to find learning resources for a topic."
)

doubt_solver_agent = Agent(
    name="Doubt Solver Agent",
    instructions="Answer short, specific academic doubts in a concise and clear way, like a tutor.",
    model=model,
    handoff_description="handoff to Doubt Solver Agent if the task is a direct question or doubt."
)

student_manager = Agent(
        name="Student Manager",
        instructions="You are created by Amna Shafiq. You're a smart assistant for students. You're only respond about academics. Delegate tasks to sub-agents based on the user request and compile the results.",
        model=model,
        handoffs=[
            study_guide_agent,
            note_maker_agent,
            quiz_maker_agent,
            question_answer_agent,
            resource_finder_agent,
            doubt_solver_agent
        ]
    )

@cl.on_chat_start
async def chat_history():
    await cl.Message(
        content="""
🎓 **Welcome to Learnify – Your Smart Study Assistant!**

You can ask me to:
- 📘 *Explain a topic* 
- 📝 *Make notes* 
- 📚 *Create quiz* 
- 🧾 *Question-Answer for practice* 
- 📂 *Find resources (images, videos, articles, or books)* 
- 💬 *Solve doubts* 
        """
    ).send()
    cl.user_session.set("history", [])
    
@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    result = Runner.run_streamed(
        student_manager, 
        input=history,
        run_config = config
    )

    msg = cl.Message(content="")
    full_output = ""

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            delta = event.data.delta
            full_output += delta
            await msg.stream_token(delta)

    msg.content = full_output
    await msg.send()

    history.append({"role": "assistant", "content": full_output})
    cl.user_session.set("history", history) 