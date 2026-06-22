# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: llmzoomcamp (3.14.3.final.0)
#     language: python
#     name: python3
# ---

# %%
from gitsource import GithubRepositoryDataReader

# %%
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()

# %%
documents = []

for file in files:
    doc = file.parse()
    documents.append(doc)

# %%

# %%
