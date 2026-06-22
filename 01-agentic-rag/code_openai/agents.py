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
from dotenv import load_dotenv
load_dotenv()

import requests
from minsearch import Index

from openai import OpenAI

import json

from ingest import load_faq_data, build_index

from rag_helper import RAGBase

# %%
openai_client = OpenAI()

# %%
documents = load_faq_data()
index = build_index(documents)

# %%
instructions = """
You're a course teaching assistant.
Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.
""".strip()

assistant = RAGBase(
    index=index,
    llm_client=openai_client,
    instructions=instructions,
)

# %%
answer = assistant.rag('How do I run Ollama locally?')
print(answer)

# %%
answer = assistant.rag('How do I run Olama locally?')
print(answer)

# %%
messages = [
    {'role': 'user', 'content': 'I just discovered the course. Can I join it?'}
]

response = openai_client.responses.create(
    model='gpt-5.4-mini',
    input=messages,
)
response.output

# %%
response.output_text


# %%
def search(query):
    boost_dict = {'question': 3.0, 'section': 0.5}
    filter_dict = {'course': 'llm-zoomcamp'}

    return index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
        filter_dict=filter_dict
    )


# %%
search_tool = {
    "type": "function",
    'name': 'search',
    'description': 'Search the FAQ database for entries matching the given query.',
    'parameters': {
        "type": "object",
        "properties": {
            'query': {
                "type": "string",
                'description': 'Search query text to look up in the course FAQ.'
            }
        },
        "required": ["query"],
        'additionalProperties': False
    }
}

# %%
response = openai_client.responses.create(
    model='gpt-5.4-mini',
    input=messages,
    tools=[search_tool]
)
response.output

# %%
response.output_text

# %% [markdown]
# Comparing the OpenAI client call without (1st call) versus with the `tools=[search_tool]`, I see that the 1st call returns a text response answer from the LLM. In contrast, the 2nd call with `tools=[search_tool]` shows that the query was modified by the LLM from 'I just discovered the course. Can I join it?' to be "Can I join the course after it has started? discovered course join late enrollment" and also the 2nd call with `tools=[search_tool]` doesn't return a text response, it just shows a tool call (ie. that the LLM determined that a tool call is needed and waits for the next turn I guess so that the tool can be called.)

# %%
call = response.output[0]
call

# %%
args = json.loads(call.arguments)
args

# %%
results = search(**args)
results

# %%
result_json = json.dumps(results, indent=2)
result_json

# %%
messages.extend(response.output)
messages

# %%
messages

# %%
messages.append({
    "type": "function_call_output",
    "call_id": call.call_id,
    "output": result_json,
})
messages

# %%
response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=[search_tool],
)

response.output

# %%
response.output_text

# %%
usage = response.usage
usage.input_tokens, usage.output_tokens


# %%
def calculate_gpt54mini_price(input_tokens, output_tokens):
    INPUT_PRICE_PER_MILLION = 0.15
    OUTPUT_PRICE_PER_MILLION = 0.60

    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }

result = calculate_gpt54mini_price(776, 56)
print("Total cost: $", round(result["total_cost"], 8))

# %%
instructions = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function. 
Use as many keywords from the user question as possible when making first requests.

Make multiple searches.

Try to expand your search by using new keywords
based on the results you get from the search.

At the end, ask if there are other areas that the user wants to explore.
""".strip()


# %%
def make_call(call):
    args = json.loads(call.arguments)
    
    if call.name == "search":
        result = search(**args)
        
    result_json = json.dumps(result, indent=2)
    
    return {
        "type": "function_call_output",
        "call_id":call.call_id,
        "output": result_json,
    }


# %%
question = "I just discovered the course. Can I join it?"

messages = [
    {"role": "developer", "content": instructions},
    {"role": "user", "content": question},
]

response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=[search_tool],
)

messages.extend(response.output)

# %%
has_function_calls = False

# %%
for item in response.output:
    if item.type == "function_call":
        print("function_call:", item.name, item.arguments)
        call_output = make_call(item)
        messages.append(call_output)
        has_function_calls = True

    elif item.type == "message":
        print("ASSISTANT:")
        print(item.content[0].text)

# %%
has_function_calls

# %%
it = 1

while True:
    print(f"iteration #{it}")
    has_function_calls = False
    
    response = openai_client.responses.create(
        model="gpt-5.4-mini",
        input = messages,
        tools = [search_tool],
    )
    
    messages.extend(response.output)
    
    for item in response.output:
        if item.type == "function_call":
            print("function_call:", item.name, item.arguments)
            call_output = make_call(item)
            messages.append(call_output)
            has_function_calls = True
            
        elif item.type == "message":
            print("ASSISTANT:")
            print(item.content[0].text)
    
    it = it + 1
    if not has_function_calls:
        break


# %%
def agent_loop(instructions, question, model="gpt-5.4-mini") -> str:
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": question}
    ]

    it = 1

    while True:
        print(f"iteration #{it}...")
        has_function_calls = False
        
        response =  openai_client.responses.create(
            model=model,
            input=messages,
            tools=[search_tool]
        )

        messages.extend(response.output)

        for item in response.output:
            print(f"reponse item type {item.type}")
            if item.type == "function_call":
                print("function_call:", item.name, item.arguments)
                call_output = make_call(item)
                messages.append(call_output)
                has_function_calls = True

            elif item.type == "message":
                print("ASSISTANT:")
                last_answer = item.content[0].text
                print(last_answer)

        it += 1 
        if not has_function_calls:
            break

    return last_answer


# %%
instructions = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function. 
Use as many keywords from the user question as possible when making first requests.

Make multiple searches. First perform search, analyze the results 
and then perform more searchers. 

At the end, ask if there are other areas that the user wants to explore.
"""

question = 'I just discovered the course. Can I join it?'

# %%
result = agent_loop(instructions, question)

# %%
agent_loop(instructions, "How do I run Olama locally?")

# %%
agent_loop(instructions, "what's queen gambit?")

# %%
instructions = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function. 
Use as many keywords from the user question as possible when making first requests.

Make multiple searches. First perform search, analyze the results 
and then perform more searches. 

The question has to be about the course or its logistics, offtopic questions 
shouldn't be answered. If the search returns nothing, it's likely an off-topic question.
If you can't answer the question using FAQ, don't do it yourself. Only use the 
facts from the FAQ database.

At the end, ask if there are other areas that the user wants to explore.
""".strip()

agent_loop(instructions, "what's queen gambit?")

# %%
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback

# %%
agent_tools = Tools()
agent_tools.add_tool(search, search_tool)


# %%
def search(query: str) -> dict[str, str]:
    """
    Search the FAQ database for entries matching the given query.
    """
    return index.search(
        query,
        num_results=5,
        boost_dict={"question": 3.0, "section": 0.5},
        filter_dict={"course": "llm-zoomcamp"}
    )


# %%
agent_tools = Tools()
agent_tools.add_tool(search)

# %%
agent_tools.get_tools()

# %%
chat_interface = IPythonChatInterface()
callback = DisplayingRunnerCallback(chat_interface)

# %%
runner = OpenAIResponsesRunner(
    tools=agent_tools,
    developer_prompt=instructions,
    chat_interface=chat_interface,
    llm_client=OpenAIClient(model="gpt-5.4-mini")
)

# %%
result = runner.loop(
    prompt="How do I run Olama locally?",
    callback=callback,
)

# %%
result.cost

# %%
result.all_messages

# %%
result2 = runner.loop(
    prompt="How do I run a different model?",
    previous_messages=result.all_messages,
    callback=callback,
)

# %%
