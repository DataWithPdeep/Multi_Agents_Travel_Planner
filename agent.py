import asyncio
import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import get_llm
from mcpClient import current_weather, forecast, list_airlines, list_airports, tavily_search
from state import TravelState

llm = get_llm()


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------
def lang_name(state) -> str:
    """Language name used inside prompts."""
    if state.get("language", "Hindi") == "Hindi":
        return "Hindi (Devanagari script)"
    return "English"


def lang_rule(state) -> str:
    """Instruction appended to system prompts so the LLM answers in the selected language."""
    if state.get("language", "Hindi") == "Hindi":
        return (
            "\n\nLANGUAGE RULE: Write your entire answer in Hindi (Devanagari script). "
            "Keep proper nouns (city, hotel and airline names), airport codes, "
            "currency symbols and numbers in English/Latin script."
        )
    return "\n\nLANGUAGE RULE: Write your entire answer in English."


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def _llm_text(system: str, prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ]
    )
    return response.content


def json_from_llm(text: str) -> Any:
    print("\n========== RAW LLM RESPONSE ==========")
    print(text)
    print("======================================\n")

    start = text.index("{")
    end = text.rindex("}") + 1

    json_text = text[start:end]

    print("\n========== EXTRACTED JSON ==========")
    print(json_text)
    print("====================================\n")

    return json.loads(json_text)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------
def supervisor_agent(state: TravelState):
    query = state["user_query"]

    # INPUT GUARDRAIL
    guardrail_prompt = f"""
    Determine whether the following request is a valid travel planning request.

    The request may be written in Hindi or English.
    Write the "reason" value in {lang_name(state)}. Keep JSON keys in English.

    Return only JSON in this format:

    {{
        "allowed": true,
        "reason": ""
    }}

    User request:
    {query}
    """

    guardrail_raw = _llm_text(
        "You are an input validation guardrail. Return strict JSON only.",
        guardrail_prompt,
    )

    print("\n========== GUARDRAIL RAW RESPONSE ==========")
    print(guardrail_raw)
    print("============================================\n")

    guardrail_result = json_from_llm(guardrail_raw)

    print("\n========== GUARDRAIL PARSED RESPONSE ==========")
    print(json.dumps(guardrail_result, indent=2, ensure_ascii=False))
    print("================================================\n")

    if not guardrail_result.get("allowed", False):
        reason = guardrail_result.get(
            "reason",
            "Request rejected by input guardrail.",
        )

        return {
            "selected_agents": [],
            "trip_constraints": {},
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [
                AIMessage(content=f"Guardrail blocked request: {reason}")
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    # Supervisor logic
    prompt = f"""
You are the supervisor of a real-world multi-agent travel planning system.

Decide which specialist agents are needed for this user request.

Available agents:
- flight_agent: use when flights, airports, airlines, routes, or airfare guidance are needed
- hotel_agent: use when hotels, stays, neighborhoods, or accommodation are needed
- weather_agent: use when weather, climate, season, packing, or forecast is useful
- budget_agent: use when budget, affordability, cost, or price constraints are mentioned
- itinerary_agent: almost always needed to produce the travel plan

Rules:
- The request may be written in Hindi or English.
- JSON keys and agent names must stay in English.
- trip_constraints "destination" and "origin" MUST be in English (e.g. "Japan", "Delhi")
  even if the request is in Hindi, because they are used for airport and weather API lookups.
- Write the "reasoning" value in {lang_name(state)}.

Return only JSON with this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    raw = _llm_text(
        "You route work to specialist agents. Return strict JSON only.",
        prompt,
    )

    parsed = json_from_llm(raw)

    print("\n========== PARSED JSON ==========")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    print("=================================\n")

    selected = parsed["selected_agents"]

    return {
        "selected_agents": selected,
        "trip_constraints": parsed["trip_constraints"],
        "supervisor_reasoning": parsed["reasoning"],
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Flight agent
# ---------------------------------------------------------------------------
def flight_agent(state: TravelState):
    query = state["user_query"]
    constraints = state["trip_constraints"]
    destination = constraints["destination"]

    print("\n========== FLIGHT AGENT INPUT ==========")
    print("Query:", query)
    print("Constraints:", constraints)
    print("========================================\n")

    airports = asyncio.run(list_airports(destination, limit=10))
    airlines = asyncio.run(list_airlines("", limit=10))

    print("\n========== AIRPORT MCP DATA ==========")
    print(airports)
    print("======================================\n")

    print("\n========== AIRLINE MCP DATA ==========")
    print(airlines)
    print("======================================\n")

    prompt = f"""
Create flight guidance for this trip.

User request:
{query}

Trip constraints:
{constraints}

Airport MCP data:
{str(airports)[:3000]}

Airline MCP data:
{str(airlines)[:3000]}

Include likely departure/arrival airports, relevant airlines,
estimated duration, fare range, peak season warning,
and booking advice.
"""

    result = _llm_text(
        "You are a flight planning specialist." + lang_rule(state),
        prompt,
    )

    print("\n========== FLIGHT AGENT OUTPUT ==========")
    print(result)
    print("=========================================\n")

    return {
        "flight_results": result,
        "messages": [AIMessage(content="Flight agent completed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Hotel agent (search + LLM summary in selected language)
# ---------------------------------------------------------------------------
def hotel_agent(state: TravelState):
    query = f"Best hotels and areas to stay for: {state['user_query']}"

    print("\n========== HOTEL AGENT INPUT ==========")
    print(query)
    print("=======================================\n")

    raw = asyncio.run(tavily_search(query))

    print("\n========== HOTEL SEARCH RESULT ==========")
    print(raw)
    print("=========================================\n")

    prompt = f"""
User request:
{state['user_query']}

Hotel search results:
{str(raw)[:4000]}

Summarize the best hotels and areas to stay, with approximate prices if available.
Keep it concise and practical.
"""

    result = _llm_text(
        "You are a hotel recommendation specialist." + lang_rule(state),
        prompt,
    )

    print("\n========== HOTEL AGENT OUTPUT ==========")
    print(result)
    print("========================================\n")

    return {
        "hotel_results": result,
        "messages": [AIMessage(content="Hotel agent completed.")],
    }


# ---------------------------------------------------------------------------
# Weather agent (data + LLM summary in selected language)
# ---------------------------------------------------------------------------
def weather_agent(state: TravelState):
    constraints = state["trip_constraints"]
    city = constraints["destination"]

    print("\n========== WEATHER AGENT INPUT ==========")
    print("City:", city)
    print("=========================================\n")

    weather_data = asyncio.run(current_weather(city))
    forecast_data = asyncio.run(forecast(city))

    print("\n========== CURRENT WEATHER ==========")
    print(weather_data)
    print("=====================================\n")

    print("\n========== WEATHER FORECAST ==========")
    print(forecast_data)
    print("======================================\n")

    prompt = f"""
Current weather:
{weather_data}

Forecast:
{forecast_data}

Summarize the weather for the traveler and give short packing advice.
"""

    result = _llm_text(
        "You are a travel weather advisor." + lang_rule(state),
        prompt,
    )

    print("\n========== WEATHER AGENT OUTPUT ==========")
    print(result)
    print("==========================================\n")

    return {
        "weather_results": result,
        "messages": [AIMessage(content="Weather agent completed.")],
    }


# ---------------------------------------------------------------------------
# Budget agent
# ---------------------------------------------------------------------------
def budget_agent(state: TravelState):

    print("\n========== BUDGET AGENT INPUT ==========")
    print("Trip Constraints:")
    print(state.get("trip_constraints"))
    print("\nFlight Results:")
    print(str(state.get("flight_results", ""))[:500])
    print("\nHotel Results:")
    print(str(state.get("hotel_results", ""))[:500])
    print("\nWeather Results:")
    print(str(state.get("weather_results", ""))[:500])
    print("=========================================\n")

    prompt = f"""
Analyze whether this trip plan is realistic for the user's budget.

User request:
{state['user_query']}

Constraints:
{str(state.get('trip_constraints', {}))[:500]}

Flight results:
{str(state.get('flight_results', ''))[:500]}

Hotel results:
{str(state.get('hotel_results', ''))[:500]}

Weather results:
{str(state.get('weather_results', ''))[:500]}

Return a concise budget assessment with:

1. Estimated cost categories
2. Risk areas
3. Money-saving suggestions
4. Whether the plan seems feasible

Keep the response concise.
"""

    result = _llm_text(
        "You are a practical travel budget analyst." + lang_rule(state),
        prompt,
    )

    print("\n========== BUDGET AGENT OUTPUT ==========")
    print(result)
    print("=========================================\n")

    return {
        "budget_results": result,
        "messages": [AIMessage(content="Budget agent completed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Itinerary agent
# ---------------------------------------------------------------------------
def itinerary_agent(state: TravelState):

    print("\n========== ITINERARY AGENT INPUT ==========")
    print("Trip Constraints:")
    print(str(state.get("trip_constraints", ""))[:200])
    print("\nFlight Results:")
    print(str(state.get("flight_results", ""))[:200])
    print("\nHotel Results:")
    print(str(state.get("hotel_results", ""))[:200])
    print("\nWeather Results:")
    print(str(state.get("weather_results", ""))[:200])
    print("\nBudget Results:")
    print(str(state.get("budget_results", ""))[:200])
    print("===========================================\n")

    prompt = f"""
Create a clear draft travel itinerary.

User request:
{state['user_query']}

Trip constraints:
{str(state.get('trip_constraints', {}))[:500]}

Flight results:
{str(state.get('flight_results', ''))[:500]}

Hotel results:
{str(state.get('hotel_results', ''))[:500]}

Weather results:
{str(state.get('weather_results', ''))[:500]}

Budget results:
{str(state.get('budget_results', ''))[:500]}

Make the output structured, practical, concise, and ready for human review.
"""

    result = _llm_text(
        "You are an expert itinerary planner." + lang_rule(state),
        prompt,
    )

    print("\n========== ITINERARY OUTPUT ==========")
    print(result)
    print("======================================\n")

    approval_request = f"""
Please review this draft travel plan.

{result}

Reply with approval or feedback.
"""

    return {
        "itinerary": result,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Human approval
# ---------------------------------------------------------------------------
def human_approval_agent(state: TravelState):
    feedback = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional feedback for revision",
                "language": "Hindi or English",
            },
        }
    )

    approved = feedback["approved"]
    human_feedback = feedback["feedback"]

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        # If the user toggled language before approving, final plan follows it
        "language": feedback.get("language", state.get("language", "Hindi")),
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# ---------------------------------------------------------------------------
# Final response
# ---------------------------------------------------------------------------
def final_response_agent(state: TravelState):

    print("\n========== FINAL AGENT INPUT ==========")
    print("Approved:", state.get("approved"))
    print("Feedback:", state.get("human_feedback"))
    print("Language:", state.get("language"))
    print("=======================================\n")

    if state["approved"]:
        prompt = f"""
The human approved this draft itinerary.

Produce the final polished travel plan.

Draft itinerary:
{state['itinerary']}

Budget notes:
{state['budget_results']}
"""
    else:
        prompt = f"""
The human did not approve the draft.

Original user request:
{state['user_query']}

Draft itinerary:
{state['itinerary']}

Human feedback:
{state['human_feedback']}

Budget notes:
{state['budget_results']}
"""

    result = _llm_text(
        "You produce final user-ready travel plans." + lang_rule(state),
        prompt,
    )

    print("\n========== FINAL RESPONSE ==========")
    print(result)
    print("====================================\n")

    return {
        "final_response": result,
        "messages": [AIMessage(content=result)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }