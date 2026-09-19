"""Few-shot examples for the Router (CLAUDE.md §4 convention: EXAMPLES: list[dict]).

Drawn from the Aurelia Financial demo corpus so the classifier sees the same
vocabulary at inference time that it will meet in the demo.
"""

EXAMPLES: list[dict] = [
    {
        "query": "How long do customers have to contest a chargeback?",
        "route": "rag",
        "why": "Answered by a policy document, not by a row in transactions.",
    },
    {
        "query": "What's our process for handling customer PII during on-call?",
        "route": "rag",
        "why": "Prose spread across a Drive standard and a Slack thread.",
    },
    {
        "query": "How many transactions were flagged for AML last month?",
        "route": "sql",
        "why": "A count over structured transaction rows.",
    },
    {
        "query": "What was the total refund value in SGD for the support department in Q2?",
        "route": "sql",
        "why": "An aggregate with a department scope and a date range.",
    },
    {
        "query": "Summarise the payment outage and tell me how many transactions failed during it.",
        "route": "rag",
        "why": (
            "Compound. The dominant intent is the written incident write-up, so it routes rag "
            "as a whole — CLAUDE.md §4 forbids splitting compound queries in the MVP."
        ),
    },
    {
        "query": "Can you look into that issue from yesterday?",
        "route": "clarify",
        "why": "No resolvable subject — neither a document nor a table can be chosen.",
    },
    {
        "query": "What about the other one?",
        "route": "clarify",
        "why": "Refers to context the graph does not hold.",
    },
]
