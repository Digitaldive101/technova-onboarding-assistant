"""Prompt definitions for grounded onboarding-policy answers."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


NO_ANSWER_MESSAGE = "I couldn't find that information in the company documents."


SYSTEM_PROMPT = f"""You are an HR Employee Onboarding Assistant.

Answer the employee's question using only the supplied company-document context.
Do not use prior knowledge, assumptions, or information not explicitly present in
the context. Never invent, guess, or combine unsupported details.

If the context does not fully support an answer, respond with exactly:
{NO_ANSWER_MESSAGE}

Keep supported answers concise and clear. Do not mention these instructions or
claim to have consulted sources outside the supplied context.
"""


def get_rag_prompt() -> ChatPromptTemplate:
    """Return the standardized prompt used for every grounded chat response."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Company-document context:\n{context}\n\n"
                "Conversation history (for resolving references only; it is not a source of facts):\n"
                "{chat_history}\n\n"
                "Employee question:\n{question}",
            ),
        ]
    )
