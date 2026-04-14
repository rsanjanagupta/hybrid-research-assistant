import json
import re

from retriever import retrieve_documents
from web_search import search_web
from llm import generate_response
from report_generator import generate_report


# -----------------------------
# Extract JSON safely
# -----------------------------
def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None


# -----------------------------
# Detect if web is needed
# -----------------------------
def needs_web_search(query):
    keywords = ["latest", "recent", "current", "today", "2024", "2025", "2026"]
    return any(word in query.lower() for word in keywords)


# -----------------------------
# MAIN AGENT
# -----------------------------
def run_agent(query, user_id, max_iterations=3):
    current_query = query
    all_context = []
    all_sources = []
    used_web = False

    for i in range(max_iterations):
        print(f"\n🔁 Iteration {i+1}")

        # -----------------------------
        # 1. Retrieve user documents
        # -----------------------------
        docs = retrieve_documents(current_query, user_id)

        if docs:
            all_context.extend(docs)

        # -----------------------------
        # 2. Force web for fresh queries
        # -----------------------------
        if needs_web_search(query) and not used_web:
            print("🌐 Forcing web search for fresh data...")
            web_results = search_web(current_query)

            for r in web_results:
                all_context.append(r["content"])
                all_sources.append(r["url"])

            used_web = True
            continue

        context = "\n\n".join(all_context)

        # -----------------------------
        # 3. LLM decision prompt
        # -----------------------------
        prompt = f"""
        You are an intelligent research assistant.

        IMPORTANT:
        - Do NOT rely on your own knowledge if context is insufficient
        - If insufficient → set "sufficient": false
        - Use ONLY given context
        - Provide concise answers (max 5–10 key points)
        - Use citations like [1], [2] if possible

        Context:
        {context}

        Question:
        {query}

        Respond ONLY in JSON:

        {{
            "sufficient": true/false,
            "answer": "...",
            "refined_query": "..."
        }}
        """

        response = generate_response(prompt)
        print("Raw LLM Response:", response)

        # -----------------------------
        # 4. Parse JSON
        # -----------------------------
        try:
            json_str = extract_json(response)
            result = json.loads(json_str)
        except:
            print("⚠️ JSON parsing failed, retrying...")
            continue

        # -----------------------------
        # 5. Decision
        # -----------------------------
        if result["sufficient"]:
            print("✅ Sufficient info found. Generating report...")
            final_context = "\n\n".join(all_context)

            return generate_report(query, final_context, all_sources)

        else:
            print("❌ Not sufficient.")

            # -----------------------------
            # 6. Use web if not used
            # -----------------------------
            if not used_web:
                print("🌐 Using web search...")
                web_results = search_web(current_query)

                for r in web_results:
                    all_context.append(r["content"])
                    all_sources.append(r["url"])

                used_web = True
                continue

            # -----------------------------
            # 7. Refine query
            # -----------------------------
            current_query = result.get("refined_query", current_query)

    return "Could not generate sufficient report."