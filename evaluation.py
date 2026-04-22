"""Evaluation harness for the 22 standard MOA questions.

Each item has:
  * question  — the natural-language question asked to the RAG pipeline
  * reference — a reference answer derived from the notarized MOA
  * keywords  — key facts that a faithful answer must contain

Metrics per question:
  * keyword_coverage   — fraction of reference keywords present in the answer
  * semantic_similarity — cosine similarity between answer and reference
                          embeddings (multilingual sentence transformer)
  * overall            — mean of the two
"""
from __future__ import annotations

from rag import cosine_sim

EVAL_QUESTIONS = [
    # ── 1. Company Identity ──────────────────────────────────────────────
    {
        "id": 1,
        "name": "Company Identity",
        "question": (
            "What is the full legal name of the company as stated in the "
            "Memorandum of Association?"
        ),
        "reference": (
            "Nagarro Mena L.L.C (Limited Liability Company - Single Owner)"
        ),
        "keywords": [
            "Nagarro Mena",
            "Limited Liability",
            "Single Owner",
        ],
    },
    {
        "id": 2,
        "name": "Company Identity",
        "question": (
            "What is the registered office address of the company?"
        ),
        "reference": (
            "Office No. 411, Own by Sheikh Ahmed bin Rashid Al Maktoum - "
            "Deira - Hor Al Anz, Dubai, UAE."
        ),
        "keywords": [
            "411",
            "Sheikh Ahmed",
            "Deira",
            "Hor Al Anz",
            "Dubai",
        ],
    },
    {
        "id": 3,
        "name": "Company Identity",
        "question": (
            "What is the company's license number?"
        ),
        "reference": "HRB213425",
        "keywords": ["HRB213425"],
    },
    {
        "id": 4,
        "name": "Company Identity",
        "question": (
            "What is the nationality of the owner (M/S. Nagarro SE)?"
        ),
        "reference": "Germany",
        "keywords": ["Germany"],
    },
    {
        "id": 5,
        "name": "Company Identity",
        "question": (
            "Who is the authorized representative of the owner mentioned "
            "in the MOA?"
        ),
        "reference": (
            "Abdullah Fadhl Abdullah Yaseen, according to a POA attested "
            "by MOFA under reference no. 13882707."
        ),
        "keywords": [
            "Abdullah Fadhl Abdullah Yaseen",
            "POA",
            "MOFA",
            "13882707",
        ],
    },
    # ── 2. Objectives and Activities ─────────────────────────────────────
    {
        "id": 6,
        "name": "Objectives and Activities",
        "question": (
            "List the primary business activities (objects) of the company "
            "as stated in the MOA."
        ),
        "reference": (
            "The objects include: Computer Systems & Communication Equipment "
            "Software Trading; Computers & Peripheral Equipment Trading; "
            "Computer Systems & Communication Equipment Software Design; "
            "Design Services; Data Entry Services; and Information Technology "
            "Network Services."
        ),
        "keywords": [
            "Software Trading",
            "Peripheral Equipment",
            "Software Design",
            "Design Services",
            "Data Entry",
            "Network Services",
        ],
    },
    {
        "id": 7,
        "name": "Objectives and Activities",
        "question": (
            "Is the company permitted to carry out insurance or banking "
            "activities?"
        ),
        "reference": (
            "No. The MOA explicitly states that the company may not carry "
            "out the business of insurance, banking, or investment of funds "
            "for the account of third parties."
        ),
        "keywords": [
            "insurance",
            "banking",
            "investment",
            "not",
        ],
    },
    {
        "id": 8,
        "name": "Objectives and Activities",
        "question": (
            "Can the company establish subsidiaries and branches?"
        ),
        "reference": (
            "Yes. The Managing Director is authorized to establish, "
            "administer, and close subsidiaries, branches, and/or "
            "representative offices of the company in the UAE."
        ),
        "keywords": [
            "Managing Director",
            "subsidiaries",
            "branches",
            "representative offices",
        ],
    },
    # ── 3. Share Capital and Liability ───────────────────────────────────
    {
        "id": 9,
        "name": "Share Capital and Liability",
        "question": (
            "What is the total authorized share capital of the company?"
        ),
        "reference": "AED 300,000 (Three Hundred Thousand AED).",
        "keywords": ["300,000", "AED"],
    },
    {
        "id": 10,
        "name": "Share Capital and Liability",
        "question": (
            "How many shares is the capital divided into, and what is the "
            "value of each share?"
        ),
        "reference": (
            "The capital is divided into 300 shares, each valued at "
            "AED 1,000."
        ),
        "keywords": ["300", "1,000"],
    },
    {
        "id": 11,
        "name": "Share Capital and Liability",
        "question": (
            "Who holds the shares and what is their ownership percentage?"
        ),
        "reference": (
            "Nagarro SE holds all 300 shares, representing 100% ownership."
        ),
        "keywords": ["Nagarro SE", "300", "100%"],
    },
    {
        "id": 12,
        "name": "Share Capital and Liability",
        "question": (
            "What is the extent of the owner's liability?"
        ),
        "reference": (
            "The company capital share owner shall only be liable to the "
            "extent of the company capital share amount (i.e., limited to "
            "AED 300,000)."
        ),
        "keywords": ["liable", "limited", "300,000"],
    },
    # ── 4. Association Clause ────────────────────────────────────────────
    {
        "id": 13,
        "name": "Association Clause",
        "question": (
            "What was the original commercial license number under which "
            "the company was established?"
        ),
        "reference": (
            "Commercial License No. 231980, issued by DED (Department of "
            "Economic Development)."
        ),
        "keywords": ["231980", "DED"],
    },
    {
        "id": 14,
        "name": "Association Clause",
        "question": (
            "What is the original notarized memorandum of association "
            "number and date?"
        ),
        "reference": (
            "Notary number 135345/1/2020, dated 06/09/2020."
        ),
        "keywords": ["135345", "2020", "06/09/2020"],
    },
    {
        "id": 15,
        "name": "Association Clause",
        "question": (
            "Why was the MOA being reinstated/amended?"
        ),
        "reference": (
            "The owner desired to amend the MOA pursuant to Federal Law "
            "No. 32 of 2021 regarding commercial companies, and the laws "
            "amending it from time to time, including any decisions, "
            "regulations, or systems issued in implementation of it."
        ),
        "keywords": [
            "Federal Law",
            "32",
            "2021",
            "commercial companies",
        ],
    },
    # ── 5. Capital Clause ────────────────────────────────────────────────
    {
        "id": 16,
        "name": "Capital Clause",
        "question": (
            "Are the shares cash shares or in-kind shares?"
        ),
        "reference": "All shares are cash shares.",
        "keywords": ["cash"],
    },
    {
        "id": 17,
        "name": "Capital Clause",
        "question": (
            "Has the share capital been fully paid up?"
        ),
        "reference": (
            "Yes. The owner declared that the value for the cash has been "
            "paid in full and has been deposited in the company's bank "
            "account."
        ),
        "keywords": ["paid in full", "bank account"],
    },
    {
        "id": 18,
        "name": "Capital Clause",
        "question": (
            "What percentage of net profits must be allocated to statutory "
            "reserve each year?"
        ),
        "reference": (
            "Not less than 10% of net profits each year, until the reserve "
            "reaches half of the capital. The owner may resolve to "
            "discontinue the allocation once the reserve reaches that "
            "threshold."
        ),
        "keywords": ["10%", "net profits", "half", "reserve"],
    },
    # ── Bonus / Cross-cutting Questions ──────────────────────────────────
    {
        "id": 19,
        "name": "Bonus / Cross-cutting",
        "question": (
            "Who is the Managing Director of the company, and what is his "
            "nationality?"
        ),
        "reference": (
            "Mr. Bachar Kassar, nationality Canadian, residing in Dubai, "
            "UAE."
        ),
        "keywords": ["Bachar Kassar", "Canadian", "Dubai"],
    },
    {
        "id": 20,
        "name": "Bonus / Cross-cutting",
        "question": (
            "What is the duration (term) of the company?"
        ),
        "reference": (
            "25 years, commencing from the date of commercial register "
            "registration, with automatic renewal for similar periods "
            "unless the owner decides otherwise."
        ),
        "keywords": ["25 years", "automatic renewal"],
    },
    {
        "id": 21,
        "name": "Bonus / Cross-cutting",
        "question": (
            "When does the company's financial year start and end?"
        ),
        "reference": (
            "It starts on January 1st and ends on December 31st of every "
            "year."
        ),
        "keywords": ["January 1", "December 31"],
    },
    {
        "id": 22,
        "name": "Bonus / Cross-cutting",
        "question": (
            "Which federal law governs this MOA?"
        ),
        "reference": (
            "Federal Law No. 32 of 2021 regarding commercial companies "
            "(the Commercial Companies Law)."
        ),
        "keywords": ["Federal Law", "32", "2021", "Commercial Companies"],
    },
]


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    a = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in a)
    return hits / len(keywords)


def evaluate_answer(answer: str, reference: str, keywords: list[str]) -> dict:
    coverage = keyword_coverage(answer, keywords)
    semantic = cosine_sim(answer, reference) if answer.strip() else 0.0
    overall = 0.5 * coverage + 0.5 * semantic
    return {
        "keyword_coverage": round(coverage, 3),
        "semantic_similarity": round(semantic, 3),
        "overall": round(overall, 3),
    }
