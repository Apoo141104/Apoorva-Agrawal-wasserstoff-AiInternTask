from groq import Groq
import json
import os
from dotenv import load_dotenv

load_dotenv()

class GroqProcessor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set via parameter or environment")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama3-8b-8192"

    def process_query(self, query, documents):
        # === Step 1: Per-document answer extraction ===
        document_responses = []
        for doc in documents:
            text = doc["text"]
            doc_id = doc["metadata"].get("doc_id", "UNKNOWN")
            page = doc["metadata"].get("page", "?")
            para = doc["metadata"].get("paragraph", "?")

            prompt = f"""Based on the user query: "{query}", extract the most relevant sentence or summary from the following document content:\n\n{text}"""

            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=300
            )
            answer = response.choices[0].message.content.strip()

            document_responses.append({
                "doc_id": doc_id,
                "answer": answer,
                "citation": f"Page {page}, Para {para}"
            })

        # === Step 2: Synthesized multi-paragraph answer + themes ===
        context = "\n\n".join([
            f"Document {doc['metadata']['doc_id']}, Page {doc['metadata']['page']}:\n{doc['text']}"
            for doc in documents
        ])

        prompt = f"""Analyze the following documents in response to the user query: "{query}"

        Documents:
        {context}

        Return structured JSON output with:
        1. 'answer': a 3–5 paragraph synthesized answer combining insights from all documents
        2. 'themes': array of themes. Each theme should have:
        - 'name': theme title
        - 'description': 2-3 sentence explanation
        - 'supporting_docs': array of {{'doc_id': '...', 'page': N}}
        """

        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=4000
        )

        theme_response = json.loads(chat_completion.choices[0].message.content)

        return {
            "answer": theme_response.get("answer", "No answer generated."),
            "themes": theme_response.get("themes", []),
            "doc_responses": document_responses
        }
