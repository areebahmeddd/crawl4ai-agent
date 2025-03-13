from __future__ import annotations
import os
import asyncio
from typing import Literal, TypedDict

import streamlit as st
from dotenv import load_dotenv
from supabase import Client

# from google import genai
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from agent import pydantic_agent, PydanticAIDependencies

load_dotenv()

# gemini_client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
supabase_client = Client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


class ChatMessage(TypedDict):
    role: Literal["user", "model"]
    content: str
    timestamp: str


async def generate_response(user_input: str):
    dependencies = PydanticAIDependencies(
        supabase_client=supabase_client,
        # gemini_client=gemini_client
    )

    async with pydantic_agent.run_stream(
        user_input, deps=dependencies, message_history=st.session_state.messages[:-1]
    ) as result:
        full_text = ""
        placeholder = st.empty()

        async for chunk in result.stream_text(delta=True):
            full_text += chunk
            placeholder.markdown(full_text)

        filtered_messages = []
        for message in result.new_messages():
            if hasattr(message, "parts"):
                if any(part.part_kind == "user-prompt" for part in message.parts):
                    continue
            filtered_messages.append(message)

        st.session_state.messages.extend(filtered_messages)
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=full_text)])
        )


def display_message(part):
    if part.part_kind == "system-prompt":
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")

    elif part.part_kind == "user-prompt":
        with st.chat_message("user"):
            st.markdown(part.content)

    elif part.part_kind == "text":
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def main():
    st.title("Pydantic AI Chatbot")
    st.write("Ask anything about Pydantic AI.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if isinstance(message, (ModelRequest, ModelResponse)):
            for part in message.parts:
                display_message(part)

    user_input = st.chat_input("What do you want to ask?")
    if user_input:
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            await generate_response(user_input)


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
