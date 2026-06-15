# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: llmzoomcamp (3.14.3)
#     language: python
#     name: python3
# ---

# %%
from sqlitesearch import TextSearchIndex
from rag_helper import RAGBase
from openai import OpenAI

# %%
sqlite_index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
    db_path="faq.db"
)

# %%
sqlite_index.count()

# %%
results = sqlite_index.search("Can I still join the course after it started?", num_results=5)
[doc["question"] for doc in results]

# %%
openai_client = OpenAI()

# %%
assistant = RAGBase(
    index=sqlite_index,
    llm_client=openai_client,
)


# %%
answer = assistant.rag("Can I still join the course after it started?")
print(answer)

# %%
sqlite_index.close()

# %%
