from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_web(query, top_k=5):
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=top_k
    )

    results = []

    for r in response["results"]:
        content = r.get("content", "")
        url = r.get("url", "")

        if content:
            results.append({
                "content": content,
                "url": url
            })

    return results