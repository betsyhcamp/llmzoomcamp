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
from dotenv import load_dotenv
load_dotenv()

# %%
import requests
from minsearch import Index

# %%
from openai import OpenAI
openai_client = OpenAI()


# %% [markdown]
# ## Testing out the OpenAI API - sending a prompt

# %%
def llm(prompt):
    response = openai_client.responses.create(
        model = "gpt-5.4-mini",
        input=prompt
    )
    return response.output_text


# %%
llm("Hey, what's up?")

# %%
question = "I just discovered the course. Can I join now?"

answer = llm(question)

print(answer)

# %%
context = """
I just discovered the course. Can I still join?
Yes, but if you want to receive a certificate, you need to submit your project while we're still accepting submissions.

Course: I have registered for the LLM Zoomcamp. When can I expect to receive the confirmation email?
You don't need it. You're accepted. You can also just start learning and submitting homework (while the form is open) without registering. It is not checked against any registered list. Registration is just to gauge interest before the start date.

What is the video/zoom link to the stream for the "Office Hours" or live/workshop sessions?
The zoom link is only published to instructors/presenters/TAs. Students participate via YouTube Live and submit questions to Slido.

Cloud alternatives with GPU
Check the quota and reset cycle carefully. Potential options include Google Colab, Kaggle, Databricks.
"""

# %%
prompt = f"""
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."

Question:
{question}

Context:
{context}
"""

# %%
answer = llm(prompt)

# %%
print(answer)

# %% [markdown]
# ## Building an indexed knowledge base - search example to return the most relevent docs

# %%
docs_url = "https://datatalks.club/faq/json/courses.json"
response = requests.get(docs_url)
courses_raw = response.json()

# %%
response

# %%
courses_raw

# %%
documents = []
url_prefix = "https://datatalks.club/faq"

for course in courses_raw:
    course_url = f"""{url_prefix}{course["path"]}"""

    course_response = requests.get(course_url)
    course_response.raise_for_status()
    course_data = course_response.json()

    documents.extend(course_data)

len(documents)

# %%
course_data

# %%
documents[0:2]

# %%
index = Index(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"]
)

index.fit(documents)

# %%
question = "I just discovered the course. Can I join now?"

# %%
search_results = index.search(
    question,
    boost_dict={"question": 2.0, "section": 0.5},
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)

# %%
search_results

# %%
[doc["question"] for doc in search_results]


# %% [markdown]
# ## The finished `search` function

# %%
def search(question, course="llm-zoomcamp"):
    boost_dict = {"question":2.0, "section": 0.5}
    filter_dict = {"course":course}
    
    # note the index variable was created outside this function
    return index.search(
        question,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
        num_results=5
    )


# %% [markdown]
# ## Building the prompt

# %%
# system prompt (stays the same for every prompt)
INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

# %%
USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""


# %%
def build_context(search_results):
    lines=[]
    for doc in search_results:
        lines.append(doc["section"])
        lines.append("Q: " + doc["question"])
        lines.append("A: " + doc["answer"])
        lines.append("")

    return "\n".join(lines).strip()


# %%
search_results

# %%
build_context(search_results)


# %%
def build_prompt(question, search_results):
    context = build_context(search_results)
    prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )
    return prompt.strip()


# %%
prompt = build_prompt(question, search_results)

print(prompt)

# %%
response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=prompt
)

# %%
response.output

# %%
response.output[0].content[0].text

# %%
response.output[0]

# %%
response.output[0].content[0].text

# %%
response.output_text

# %%
response.usage

# %%
input_price = 0.75 / 1_000_000
output_price = 4.50 / 1_000_000

cost = (
    response.usage.input_tokens * input_price +
    response.usage.output_tokens * output_price
)

cost

# %%
message_history = [
    {"role": "developer", "content": INSTRUCTIONS},
    {"role": "user", "content": prompt}
]

response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=message_history
)


# %%
def llm(instructions, user_prompt, model="gpt-5.4-mini"):
    message_history = [
        {"role": "developer", "content":instructions},
        {"role": "user", "content":user_prompt}
    ]
    
    response = openai_client.responses.create(
        model=model,
        input=message_history
    )
    
    return response.output_text


# %%
def rag(query, model="gpt-5.4-mini"):
    search_results = search(query)
    prompt =build_prompt(query,search_results)
    answer = llm(INSTRUCTIONS, prompt, model=model)
    return answer


# %%
answer = rag("I just discovered the course. Can I join now?")
print(answer)

# %%
rag("How do I get a certificate?")

# %%

# %% [markdown]
# ## Using `rag_helper.py` and `ingest.py` functions in a notebook

# %%
from dotenv import load_dotenv
load_dotenv()

from ingest import load_faq_data, build_index
from rag_helper import RAGBase
from openai import OpenAI

# %%
documents = load_faq_data()
index = build_index(documents)

# %%
openai_client = OpenAI()

# %%
assistant = RAGBase(
    index=index,
    llm_client=openai_client
)

# %%
answer = assistant.rag("I just discovered the course. Can I join now?")
print(answer)

# %%
assistant.rag("How do I get a certificate?")

# %%
assistant.rag("Can I still join the course after it started?")

# %%
assistant.rag("How do I run Ollama locally?")

# %%
assistant.rag("How do I run Olama locally?")

# %%
messages = [
    {"role": "user", "content": "I just discovered the course. Can I join it?"}
]

response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
)

response.output_text

# %%

search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

# %%
response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=[search_tool],
)

response.output

# %%
import json

call = response.output[0]
args = json.loads(call.arguments)


# %%
results = search(**args)
result_json = json.dumps(results, indent=2)
