
import csv
import re
import time
from pathlib import Path

from rag_pipeline import (
    _scroll_patient_chunks,
    embedding_model,
    gemini_client,
)

BACKEND_DIR = Path(__file__).resolve().parent
GROUND_TRUTH = (
    Path.home() / "Downloads" / "medrag_evaluation_ground_truth.csv"
)
OUTPUT = BACKEND_DIR / "rag_evaluation_pilot.csv"
PILOT_SIZE = 5
GEMINI_MODEL = "gemini-3.6-flash"
TOP_K = 8


def classify_question(question):
    """Classify a question to select cautious reranking features."""
    q = question.lower()

    if any(term in q for term in (
        "diagnosis", "diagnoses", "medical condition",
    )):
        return "diagnosis"

    if any(term in q for term in (
        "how long", "duration", "how many years", "when diagnosed",
    )):
        return "duration"

    if any(term in q for term in (
        "blood pressure", "patient's bp", "vital signs",
    )):
        return "blood_pressure"

    if any(term in q for term in (
        "family history", "which relatives", "family members",
        "which relatives of the patient",
    )):
        return "family_history"

    if any(term in q for term in (
        "symptom", "symptoms", "present with",
    )):
        return "symptoms"

    return "general"


def rerank_chunks(question, chunks, similarities):
    """
    Experimental content-aware reranking.
    Return (adjusted_score, original_score, chunk).
    """
    question_type = classify_question(question)
    scored_chunks = []

    symptom_terms = (
        "burning", "tingling", "numbness", "thirst",
        "frequent urination", "blurred vision", "fatigue",
        "weight loss", "nocturia", "increased urination",
    )

    relatives = (
        "father", "mother", "brother", "sister",
        "parent", "son", "daughter",
    )

    for similarity, chunk in zip(similarities, chunks):
        original_score = float(similarity)
        adjusted_score = original_score
        text = chunk["text"].lower()

        if question_type == "diagnosis":
            if "assessment and diagnoses" in text:
                adjusted_score += 0.30
            elif re.search(r"\bdiagnoses\b", text):
                adjusted_score += 0.20

        elif question_type == "duration":
            # Prefer explicit duration evidence over a generic mention
            # of diabetes or medication.
            if re.search(
                r"(duration.{0,30}\b\d+\s+years?\b|"
                r"\b\d+\s+years?\s+(?:of|with)\s+diabetes\b)",
                text,
            ):
                adjusted_score += 0.25

            if re.search(
                r"diagnosed.{0,40}(?:diabetes|type 2)",
                text,
            ):
                adjusted_score += 0.08

        elif question_type == "blood_pressure":
            # Reward an actual numerical pressure reading, not generic
            # advice or a general mention of blood pressure.
            has_pressure_value = bool(
                re.search(
                    r"\b\d{2,3}\s*/\s*\d{2,3}\s*mmhg\b",
                    text,
                )
            )
            if has_pressure_value:
                adjusted_score += 0.30
            elif "vital signs" in text and "mmhg" in text:
                adjusted_score += 0.10

        elif question_type == "family_history":
            has_family_section = "family history" in text
            has_relative = any(
                re.search(rf"\b{re.escape(relative)}\b", text)
                for relative in relatives
            )
            has_diabetes = "diabetes" in text

            if has_family_section and has_relative and has_diabetes:
                adjusted_score += 0.25
            elif has_relative and has_diabetes:
                adjusted_score += 0.10

        elif question_type == "symptoms":
            matches = sum(
                term in text for term in symptom_terms
            )
            adjusted_score += min(matches, 5) * 0.04

        scored_chunks.append(
            (adjusted_score, original_score, chunk)
        )

    ranked = sorted(
        scored_chunks,
        key=lambda item: item[0],
        reverse=True,
    )[:TOP_K]

    return ranked


def main():
    if not GROUND_TRUTH.exists():
        print(f"Ground-truth file not found: {GROUND_TRUTH}")
        return

    with GROUND_TRUTH.open(
        newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print("No evaluation questions found.")
        return

    results = []

    for index, row in enumerate(rows[:PILOT_SIZE], start=1):
        question = row["question"].strip()
        filename = Path(row["pdf_name"]).name

        print(f"\n[{index}/{min(PILOT_SIZE, len(rows))}] {question}")
        print("Expected PDF:", filename)

        answer = ""
        citations = ""
        status = "OK"

        try:
            points = _scroll_patient_chunks(filename)

            chunks = []
            seen_chunks = set()

            for point in points:
                payload = getattr(point, "payload", None) or {}
                text = payload.get("text", "")

                if not text.strip():
                    continue

                page = payload.get("page", "")
                chunk_index = payload.get("chunk_index", 0)

                dedupe_key = (
                    str(page),
                    str(chunk_index),
                    text.strip(),
                )

                if dedupe_key in seen_chunks:
                    continue

                seen_chunks.add(dedupe_key)

                chunks.append({
                    "text": text,
                    "filename": payload.get("filename", filename),
                    "page": page,
                    "chunk_index": chunk_index,
                })

            if not chunks:
                raise RuntimeError(
                    f"No indexed chunks found for {filename}"
                )

            vectors = embedding_model.encode(
                [chunk["text"] for chunk in chunks],
                normalize_embeddings=True,
            )
            query_vector = embedding_model.encode(
                question,
                normalize_embeddings=True,
            )

            similarities = vectors @ query_vector

            ranked = rerank_chunks(
                question,
                chunks,
                similarities,
            )

            context_parts = []
            used_pages = []

            for adjusted_score, original_score, chunk in ranked:
                context_parts.append(
                    f"Document: {chunk['filename']}\n"
                    f"Page: {chunk['page']}\n"
                    f"Content:\n{chunk['text']}"
                )
                used_pages.append(str(chunk["page"]))

            prompt = f"""
Answer the question using only the evidence below.
If the evidence does not contain the answer, say that it is not available.
Do not combine facts from different patients or invent details.
Give a concise answer.
Do not provide medical advice or infer a diagnosis beyond the evidence.

Evidence:
{chr(10).join(context_parts)}

Question: {question}
"""

            response = None

            for attempt in range(3):
                try:
                    response = gemini_client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt,
                    )
                    break

                except Exception as exc:
                    error_text = str(exc).upper()
                    temporary_error = any(
                        marker in error_text
                        for marker in (
                            "503", "UNAVAILABLE", "429",
                            "RESOURCE_EXHAUSTED",
                        )
                    )

                    if not temporary_error or attempt == 2:
                        raise

                    wait_seconds = 2 * (attempt + 1)
                    print(
                        f"Temporary Gemini error. Retrying in "
                        f"{wait_seconds} seconds "
                        f"(attempt {attempt + 2}/3)..."
                    )
                    time.sleep(wait_seconds)

            answer = (response.text or "").strip()
            if not answer:
                status = "ERROR: Empty Gemini response"

            citations = (
                f"{filename}; pages: "
                + ", ".join(dict.fromkeys(used_pages))
            )

        except Exception as exc:
            status = f"ERROR: {type(exc).__name__}: {exc}"
            answer = ""
            citations = ""

        print("Status:", status)
        print("Answer:", answer[:300] or "(none)")
        print("Retrieved source:", citations or "(none)")

        results.append({
            "pdf_name": filename,
            "patient_name": row.get("patient_name", ""),
            "question": question,
            "expected_answer": row.get("expected_answer", ""),
            "expected_page": row.get("expected_page", ""),
            "qtype": row.get("qtype", ""),
            "difficulty": row.get("difficulty", ""),
            "generated_answer": answer,
            "citations": citations,
            "status": status,
        })

    if results:
        with OUTPUT.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(results[0].keys())
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"\nPilot saved to: {OUTPUT}")
        print(
            "Successful calls:",
            sum(result["status"] == "OK" for result in results),
            "/",
            len(results),
        )
        print("Note: answers and citations require correctness review.")


if __name__ == "__main__":
    main()
