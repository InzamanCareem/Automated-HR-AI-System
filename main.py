from dotenv import load_dotenv
from pathlib import Path
from typing import Annotated, Literal, Optional, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from vector import LTM

load_dotenv()

llm = init_chat_model(model="google_genai:gemini-2.5-flash")

checkpointer = InMemorySaver()


class MessageClassifier(BaseModel):
    message_type: Literal["scheduling", "leave", "compliance", "clarification"] = Field(...,
                                                                                        description="Classify if the message requires a scheduling, leaving, compliance or clarification response (support)")


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next: Optional[str]
    memories: List[str]


def classifier_agent(state: State):
    last_message = state["messages"][-1]

    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        SystemMessage(
            content="""Classify the user message as either:
                                    - 'scheduling': if it asks for planning and organizing work shifts, hours, and time off.
                                    - 'leave': if it asks for time off, such as vacation, sick leave, and personal leave.
                                    - 'compliance': if it asks for labor laws, workplace regulations, and internal policies related to employees and employment practices.
                                    - 'clarification': if it asks for providing clear explanations or additional details to remove confusion about policies, tasks, decisions, or workplace communication.
                                """
        ),
        HumanMessage(content=last_message.content)
    ])

    return {"next": result.message_type}


def scheduling_agent(state: State):
    last_message = state["messages"][-1]
    memories = state.get("memories", [])

    messages = [
        SystemMessage(
            content=f"""
                You are an intelligent HR Scheduling Agent. Focus on coordinating interviews, meetings, employee shifts, and HR-related scheduling tasks efficiently.
                Communicate professionally, clearly, and politely with candidates, employees, and managers.
                Automatically suggest available time slots, manage calendar conflicts, send reminders, and handle rescheduling requests smoothly.
                Prioritize accuracy, time management, and a positive candidate and employee experience.
                Ask relevant scheduling questions when necessary, such as availability, preferred time zones, or meeting preferences.
                Avoid unnecessary conversation and focus on efficient scheduling coordination.
                
                Employee memories:
                {memories}
                Use only if relevant.
            """
        ),
        HumanMessage(content=last_message.content)
    ]

    reply = llm.invoke(messages)

    return {"messages": AIMessage(content=reply.content)}


def leave_agent(state: State):
    last_message = state["messages"][-1]
    memories = state.get("memories", [])

    messages = [
        SystemMessage(
            content=f"""
                You are an HR Leave Management Assistant. Help employees manage leave requests, check leave balances, understand company leave policies, and coordinate approvals professionally and efficiently.
                Communicate clearly, politely, and supportively while handling leave-related queries.
                Assist with sick leave, vacation leave, casual leave, maternity/paternity leave, and emergency leave requests.
                Ask relevant questions when necessary, such as leave dates, leave type, duration, and approval requirements.
                Provide accurate information about leave policies, remaining balances, and approval status.
                Avoid unnecessary conversation and focus on smooth and efficient leave management support.
                                
                Employee memories:
                {memories}
                Use only if relevant.
            """
        ),
        HumanMessage(content=last_message.content)
    ]

    reply = llm.invoke(messages)

    return {"messages": AIMessage(content=reply.content)}


def compliance_agent(state: State):
    last_message = state["messages"][-1]
    memories = state.get("memories", [])

    messages = [
        SystemMessage(
            content=f"""
                You are a Compliance Management Assistant. Help users ensure that company policies, legal requirements, and regulatory standards are followed accurately and consistently.
                Communicate professionally, clearly, and responsibly while handling compliance-related tasks and questions.
                Assist with policy verification, audit preparation, regulatory documentation, risk identification, and compliance reporting.
                Provide guidance on company procedures, workplace regulations, data protection policies, and internal compliance standards.
                Ask relevant questions when necessary, such as applicable regulations, required documents, deadlines, or department-specific requirements.
                Prioritize accuracy, confidentiality, and adherence to organizational and legal standards.
                Avoid giving speculative legal advice and focus on compliance support and procedural guidance.
                                
                Employee memories:
                {memories}
                Use only if relevant.
            """
        ),
        HumanMessage(content=last_message.content)
    ]

    reply = llm.invoke(messages)

    return {"messages": AIMessage(content=reply.content)}


def clarification_agent(state: State):
    last_message = state["messages"][-1]
    memories = state.get("memories", [])

    messages = [
        SystemMessage(
            content=f"""
                You are a Clarification Assistant. Your role is to identify missing, unclear, or ambiguous information in the user's request and ask precise follow-up questions to fully understand their needs.
                Communicate clearly, politely, and concisely while guiding the user toward providing complete and accurate information.
                Focus on eliminating confusion, confirming details, and ensuring requirements are properly understood before proceeding.
                Ask one or more relevant clarification questions when information is incomplete, inconsistent, or open to interpretation.
                Avoid making assumptions or providing final answers until the necessary details are confirmed.
                Prioritize accuracy, context understanding, and effective communication.
                                
                Employee memories:
                {memories}
                Use only if relevant.
            """
        ),
        HumanMessage(content=last_message.content)
    ]

    reply = llm.invoke(messages)

    return {"messages": AIMessage(content=reply.content)}


graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classifier_agent)
graph_builder.add_node("scheduling", scheduling_agent)
graph_builder.add_node("leave", leave_agent)
graph_builder.add_node("compliance", compliance_agent)
graph_builder.add_node("clarification", clarification_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_conditional_edges("classifier", lambda state: state.get("next"), {
    "scheduling": "scheduling",
    "leave": "leave",
    "compliance": "compliance",
    "clarification": "clarification"
})

graph_builder.add_edge("scheduling", END)
graph_builder.add_edge("leave", END)
graph_builder.add_edge("compliance", END)
graph_builder.add_edge("clarification", END)

app = graph_builder.compile(checkpointer=checkpointer)


def run_conversation(user_message: str, ltm, emp_id):
    config = {"configurable": {"thread_id": "demo_session_001"}}

    memory_context = ltm.retrieve_memory(emp_id, user_message)

    response = app.invoke(
        {"messages": HumanMessage(content=user_message),
         "memories": SystemMessage(content=memory_context)},
        config=config)

    ai_response = response["messages"][-1].content.strip()

    print(f"AI response: {ai_response}\n\n")


if __name__ == "__main__":
    ltm = LTM()

    if not Path("ltm_store").exists():
        ltm.store_memory(employee_id="001", text="Prefers morning shifts.")
        ltm.store_memory(employee_id="001", text="Mentioned burnout in April.")
        ltm.store_memory(employee_id="001", text="Planning vacation in August.")

    while True:
        emp_id = input("Enter your employee id: ")
        msg = input("Enter your question: ")

        if msg == "quit":
            break

        run_conversation(msg, ltm, emp_id)
