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
from ingest import load_faq_data

# %%
documents = load_faq_data()
print(f"Loaded {len(documents)} documents")

# %%
docs_llm = [doc for doc in documents if doc["course"]=="llm-zoomcamp"]
print(f"LLM Zoomcamp: {len(docs_llm)} documents")

# %%
import time
from sqlitesearch import TextSearchIndex

# %%
index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields= ["course"],
    db_path="faq.db"
)

# %%
for doc in docs_llm:
    index.add(doc)
    print(f"Added: {doc["question"][:60]}...")
    time.sleep(0.5)

index.close()
print("Done. Index saved to faq.db")

# %%
