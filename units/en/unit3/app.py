from typing import TypedDict, Annotated
import os
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from units.en.unit3.tools import search_tool, weather_info_tool, hub_stats_tool, guest_info_tool
# Generate the chat interface, including the tools
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACE_API_TOKEN"),
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [search_tool, weather_info_tool, hub_stats_tool, guest_info_tool]
chat_with_tools = chat.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):\
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

def invoke(agent, message):
    """Invoke the assistant with the current state."""
    thread_id = "conversation-1"
    config = {"configurable": {"thread_id": thread_id}}
    return agent.invoke(message, config=config)

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct responses
    tools_condition,
)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
alfred = builder.compile(checkpointer=memory)

messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
response = invoke(alfred, {"messages": messages})

print("🎩 Alfred's Response:")

print(response['messages'][-1].content)

messages = [HumanMessage(content="What projects is she currently working on?")]
response = invoke(alfred, {"messages": messages})

print("🎩 Alfred's Response:")

print(response['messages'][-1].content)