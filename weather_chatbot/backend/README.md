# langfuse prompts
name: router
chat

system msg: You are a routing assistant.
When a user asks for something that requires a tool, emit a tool call.
If a tool is not needed, respond normally.

set production label


name: summariser
chat

system msg: You are a summarizer. Read the user's question and the latest tool results,
and produce a clear, concise final answer.

set production label

# run
uv run main.py