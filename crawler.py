import os
import json
import asyncio
import requests
from dataclasses import dataclass
from xml.etree import ElementTree
from urllib.parse import urlparse
from typing import List, Dict, Any
from datetime import datetime, timezone

import ollama
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from supabase import create_client
from google import genai
from google.genai import types

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


@dataclass
class KnowledgeChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


async def get_info(chunk: str, url: str) -> Dict[str, str]:
    system_prompt = """
You are an AI assistant that extracts meaningful titles and summaries from technical documentation.

Instructions:
- Identify whether this is the beginning or a middle part of a document.
- Generate a clear and concise title.
- Write a brief but informative summary.
- Use a professional and structured tone.

Output Format (JSON):
{
  "title": "<extracted title>",
  "summary": "<generated summary>"
}
"""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"URL: {url}\n\nContent:\n{chunk[:1000]}...",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt, response_mime_type="application/json"
            ),
        )
        response_json = json.loads(response.text)
        if isinstance(response_json, list) and len(response_json) > 0:
            response_json = response_json[0]
        if (
            isinstance(response_json, dict)
            and "title" in response_json
            and "summary" in response_json
        ):
            return response_json
        else:
            print(f"[ERROR] Invalid Response: {response_json}")
            raise ValueError("Invalid response format")
    except Exception as error:
        print(f"[ERROR] Info Extraction Failed: {error}")
        return {"title": "Title Unavailable", "summary": "Summary Unavailable"}


async def get_embedding(text: str) -> List[float]:
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response.embedding
    except Exception as error:
        print(f"[ERROR] Embedding Failed: {error}")
        return [0] * 768


async def insert_chunk(chunk: KnowledgeChunk):
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding,
        }

        existing_chunk = (
            supabase_client.table("knowledge_base")
            .select("*")
            .eq("url", chunk.url)
            .eq("chunk_number", chunk.chunk_number)
            .execute()
        )
        if existing_chunk.data:
            result = (
                supabase_client.table("knowledge_base")
                .update(data)
                .eq("url", chunk.url)
                .eq("chunk_number", chunk.chunk_number)
                .execute()
            )
            print(f"[INFO] Chunk {chunk.chunk_number} Updated - {chunk.url}")
        else:
            result = supabase_client.table("knowledge_base").insert(data).execute()
            print(f"[INFO] Chunk {chunk.chunk_number} Inserted - {chunk.url}")

        return result
    except Exception as error:
        print(f"[ERROR] Insert/Update Failed: {error}")
        return None


async def process_chunk(chunk: str, chunk_number: int, url: str) -> KnowledgeChunk:
    extracted = await get_info(chunk, url)
    embedding = await get_embedding(chunk)

    metadata = {
        "source": "pydantic_ai_docs",
        "url_path": urlparse(url).path,
        "size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).strftime("%d-%b-%Y %H:%M:%S"),
    }

    return KnowledgeChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted["title"],
        summary=extracted["summary"],
        content=chunk,
        metadata=metadata,
        embedding=embedding,
    )


def split_text(text: str, size: int = 5000) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        code_block = chunk.rfind("```")

        if code_block != -1 and code_block > size * 0.3:
            end = start + code_block

        elif "\n\n" in chunk:
            last_break = chunk.rfind("\n\n")
            if last_break > size * 0.3:
                end = start + last_break

        elif ". " in chunk:
            last_period = chunk.rfind(". ")
            if last_period > size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(start + 1, end)

    return chunks


async def process_doc(url: str, markdown: str):
    chunks = split_text(markdown)

    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)

    insert_tasks = [insert_chunk(chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_tasks)


async def crawl_sites(urls: List[str], max_concurrent: int = 5):
    browser_config = BrowserConfig(
        browser_type="chromium",
        proxy_config=None,
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url, config=crawl_config, session_id="crawl-worker"
                )
                if result.success:
                    print(f"[SUCCESS] Crawled: {url}")
                    await process_doc(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"[FAILED] {url} - Error: {result.error_message}")

        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()


def fetch_urls() -> List[str]:
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()

        root = ElementTree.fromstring(response.content)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        return [loc.text for loc in root.findall(".//ns:loc", namespace)]
    except Exception as error:
        print(f"[ERROR] Fetch URLs Failed: {error}")
        return []


async def main():
    urls = fetch_urls()
    if not urls:
        print("[WARNING] No URLs found in sitemap.")
        return

    print(f"[INFO] Crawling {len(urls)} URLs...")
    await crawl_sites(urls)


if __name__ == "__main__":
    asyncio.run(main())
