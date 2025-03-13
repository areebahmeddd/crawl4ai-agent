from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List

import ollama
from dotenv import load_dotenv
from supabase import Client
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel

load_dotenv()


@dataclass
class PydanticAIDependencies:
    supabase_client: Client
    # gemini_client: genai.Client


system_prompt = """
### Role:
You are an expert in Pydantic AI, a Python AI agent framework. You have access to all documentation, including:
- Examples
- API reference
- Other relevant resources

### Instructions:
- **Scope:** Assist only with Pydantic AI-related queries.
- **Action-Oriented:** Perform actions without asking for confirmation.
- **Documentation Lookup:**
  - Always start by retrieving relevant documentation using RAG.
  - Check the list of available documentation pages if necessary.
- **Honesty:** If documentation does not contain the answer, inform the user clearly.
"""

pydantic_agent = Agent(
    model=GeminiModel(model_name="gemini-2.0-flash", api_key=os.getenv("GEMINI_KEY")),
    system_prompt=system_prompt,
    deps_type=PydanticAIDependencies,
    retries=2,
)


async def get_embedding(text: str) -> List[float]:
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response.embedding
    except Exception as error:
        print(f"[ERROR] Embedding Failed: {error}")
        return [0] * 768


@pydantic_agent.tool
async def get_docs(context: RunContext[PydanticAIDependencies], user_query: str) -> str:
    try:
        query_embedding = await get_embedding(user_query)
        result = context.deps.supabase_client.rpc(
            "match_knowledge_base",
            {
                "query_embedding": query_embedding,
                "match_count": 5,
                "filter": {"source": "pydantic_ai_docs"},
            },
        ).execute()
        if not result.data:
            return "No relevant documentation found."

        documents = []
        for document in result.data:
            document_title = document["title"]
            document_content = document["content"]
            documents.append(f"# {document_title}\n\n{document_content}")

        formatted_documents = "\n\n---\n\n".join(documents)
        return formatted_documents
    except Exception as error:
        print(f"[ERROR] Documentation Fetch Failed: {error}")
        return f"[ERROR] Documentation Fetch Failed: {error}"


@pydantic_agent.tool
async def get_pages(context: RunContext[PydanticAIDependencies]) -> List[str]:
    try:
        result = (
            context.deps.supabase_client.from_("knowledge_base")
            .select("url")
            .eq("metadata->>source", "pydantic_ai_docs")
            .execute()
        )
        if not result.data:
            return []

        return sorted(set(document["url"] for document in result.data))
    except Exception as error:
        print(f"[ERROR] Pages Fetch Failed: {error}")
        return []


@pydantic_agent.tool
async def get_content(context: RunContext[PydanticAIDependencies], url: str) -> str:
    try:
        result = (
            context.deps.supabase_client.from_("knowledge_base")
            .select("title, content, chunk_number")
            .eq("url", url)
            .eq("metadata->>source", "pydantic_ai_docs")
            .order("chunk_number")
            .execute()
        )
        if not result.data:
            return f"No content found for URL: {url}"

        page_title = result.data[0]["title"].split(" - ")[0]
        content_chunks = [f"# {page_title}"]

        for chunk in result.data:
            content_chunks.append(chunk["content"])

        formatted_content = "\n\n".join(content_chunks)
        return formatted_content
    except Exception as error:
        print(f"[ERROR] Content Fetch Failed: {error}")
        return f"[ERROR] Content Fetch Failed: {error}"
