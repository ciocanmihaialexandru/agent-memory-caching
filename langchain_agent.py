"""
Agent LangChain  — alimentat din TOOL_REGISTRY.
"""

import os
import argparse

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from prompts.registry import get_prompt_registry
from memory.conversation import ConversationMemory

import tools.basic_tools
from tools.tool_wrapper import ToolWrapper

load_dotenv()

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
MODEL = os.getenv("AGENT_MODEL", "claude")   # numele din litellm-config.yaml

def build_agent():
    system_prompt = get_prompt_registry().render(
        "rag_system",
        rol="un asistent bancar",
        domeniu="comisioanele și pachetele de cont ale CEC Bank",
        max_cuvinte=200,
    )
    llm = ChatOpenAI(
        model=MODEL,
        base_url=LITELLM_BASE_URL,
        api_key=os.getenv("LITELLM_API_KEY", "sk-not-needed"),  # LiteLLM local nu cere cheie reală
        temperature=0,
    )

    tools = ToolWrapper.to_langchain_tools()
    print(f"Tool-uri încărcate în agent: {[t.name for t in tools]}")

    return create_agent(llm, tools, system_prompt=system_prompt)


def main():
    parser = argparse.ArgumentParser(description="Agent LangChain peste documentul CEC")
    parser.add_argument(
        "question",
        nargs="?",
        default="Cât costă retragerea de la ATM la pachetul PREMIUM?",
        help="Întrebarea pentru agent",
    )
    parser.add_argument(
        "-s", "--session",
        default="default",
        help="ID-ul conversației (memoria se leagă de el)",
    )
    args = parser.parse_args()

    memory = ConversationMemory()
    agent = build_agent()

    # 1. încărcăm istoricul conversației și adăugăm întrebarea curentă
    history = memory.load_history(args.session)
    messages = history + [{"role": "user", "content": args.question}]

    # 2. rulăm agentul cu TOT contextul
    result = agent.invoke({"messages": messages})
    answer = result["messages"][-1].content

    # 3. salvăm turul (user + assistant) pentru request-urile viitoare
    memory.add_turn(args.session, args.question, answer)

    print("\n" + "=" * 60)
    print(f"RĂSPUNS (sesiune: {args.session}):")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()