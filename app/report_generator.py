from llm import generate_response


def generate_report(query, context, sources):
    """
    Generates a structured research report
    """
    sources_text = "\n".join([f"[{i+1}] {src}" for i, src in enumerate(sources)])

    prompt = f"""
      You are an advanced AI research assistant.

      Generate a structured research report.

      Query:
      {query}

      Context:
      {context}

      Sources:
      {sources_text}

      IMPORTANT:
      - Use citations like [1], [2] in the report
      - Reference only the given sources
      - Add a References section at the end

      Structure:
      1. Title
      2. Abstract
      3. Introduction
      4. Methodology
      5. Findings
      6. Discussion
      7. Conclusion
      8. References
         """
    return generate_response(prompt)