"""
Comprehensive example demonstrating function calling / tools support.

This example shows how to use Azure OpenAI's function calling capabilities
to enable agent architectures, structured outputs, and tool use.
"""

import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv

from azure_llm_toolkit import AzureConfig, AzureLLMClient
from azure_llm_toolkit.tools import (
    FunctionDefinition,
    ToolRegistry,
    ToolCall,
    tool,
    get_default_registry,
)


# ==================== Example Tools ====================


def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """
    Get the current weather for a location.

    Args:
        location: City name or location
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        Weather information dictionary
    """
    # In a real implementation, this would call a weather API
    mock_data = {
        "Oslo": {"temperature": -2, "condition": "snowy", "humidity": 85},
        "London": {"temperature": 8, "condition": "rainy", "humidity": 90},
        "Tokyo": {"temperature": 15, "condition": "sunny", "humidity": 60},
        "New York": {"temperature": 5, "condition": "cloudy", "humidity": 70},
    }

    weather = mock_data.get(location, {"temperature": 20, "condition": "unknown", "humidity": 50})

    if unit == "fahrenheit":
        weather["temperature"] = weather["temperature"] * 9 / 5 + 32
        weather["unit"] = "°F"
    else:
        weather["unit"] = "°C"

    return {
        "location": location,
        "temperature": weather["temperature"],
        "condition": weather["condition"],
        "humidity": weather["humidity"],
        "unit": weather["unit"],
    }


def calculate(expression: str) -> dict:
    """
    Calculate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2")

    Returns:
        Result dictionary with answer
    """
    try:
        # Safe evaluation (in production, use a proper math parser)
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result, "error": None}
    except Exception as e:
        return {"expression": expression, "result": None, "error": str(e)}


def get_current_time(timezone: str = "UTC") -> dict:
    """
    Get the current time in a timezone.

    Args:
        timezone: Timezone name (e.g., "UTC", "EST", "PST")

    Returns:
        Current time information
    """
    # In real implementation, use proper timezone library
    now = datetime.now()

    return {
        "timezone": timezone,
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "day_of_week": now.strftime("%A"),
    }


async def search_database(query: str, limit: int = 5) -> dict:
    """
    Search a database (async example).

    Args:
        query: Search query
        limit: Maximum results to return

    Returns:
        Search results
    """
    # Simulate async database search
    await asyncio.sleep(0.1)

    mock_results = [
        {"id": 1, "title": f"Result for '{query}' #1", "relevance": 0.95},
        {"id": 2, "title": f"Result for '{query}' #2", "relevance": 0.87},
        {"id": 3, "title": f"Result for '{query}' #3", "relevance": 0.76},
    ]

    return {
        "query": query,
        "results": mock_results[:limit],
        "total": len(mock_results),
    }


# ==================== Examples ====================


async def example_basic_function_calling():
    """Basic function calling example."""
    print("=" * 80)
    print("Basic Function Calling Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    # Create tool registry
    registry = ToolRegistry()
    registry.register_function(get_current_weather)
    registry.register_function(calculate)
    registry.register_function(get_current_time)

    # Get tool definitions for API
    tools = registry.get_tool_definitions()

    print(f"\nRegistered {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool['function']['name']}: {tool['function']['description']}")

    # Chat with tools
    messages = [{"role": "user", "content": "What's the weather like in Oslo and Tokyo?"}]

    print(f"\nUser: {messages[0]['content']}")

    # First API call - model decides to use tools
    response = await client.chat_completion(
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    # Check if model wants to call tools
    if response.raw_response and hasattr(response.raw_response.choices[0].message, "tool_calls"):
        tool_calls_data = response.raw_response.choices[0].message.tool_calls

        if tool_calls_data:
            print(f"\nModel requested {len(tool_calls_data)} tool calls:")

            # Add assistant message with tool calls
            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls_data
                ],
            }
            messages.append(assistant_message)

            # Execute tool calls
            tool_results = []
            for tc in tool_calls_data:
                tool_call = ToolCall.from_openai_tool_call(tc)
                print(f"  - {tool_call.function_name}({tool_call.function_arguments})")

                result = await registry.execute_tool_call(tool_call)
                tool_results.append(result)

                print(f"    Result: {result.result}")

                # Add tool result to messages
                messages.append(result.to_message())

            # Second API call - model uses tool results
            print("\nSending tool results back to model...")
            final_response = await client.chat_completion(
                messages=messages,
                tools=tools,
            )

            print(f"\nAssistant: {final_response.content}")
        else:
            print(f"\nAssistant: {response.content}")
    else:
        print(f"\nAssistant: {response.content}")


async def example_decorator_registration():
    """Example using decorator for tool registration."""
    print("\n" + "=" * 80)
    print("Decorator-Based Tool Registration")
    print("=" * 80)

    load_dotenv()

    # Create a custom registry
    custom_registry = ToolRegistry()

    # Register tools using decorator
    @tool(
        description="Convert temperature between Celsius and Fahrenheit",
        registry=custom_registry,
    )
    def convert_temperature(value: float, from_unit: str, to_unit: str) -> dict:
        """Convert temperature between units."""
        if from_unit == "celsius" and to_unit == "fahrenheit":
            result = value * 9 / 5 + 32
        elif from_unit == "fahrenheit" and to_unit == "celsius":
            result = (value - 32) * 5 / 9
        else:
            result = value

        return {
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": round(result, 2),
            "converted_unit": to_unit,
        }

    @tool(description="Get stock price (mock)", registry=custom_registry)
    def get_stock_price(symbol: str) -> dict:
        """Get current stock price."""
        # Mock data
        prices = {
            "AAPL": 178.50,
            "GOOGL": 142.30,
            "MSFT": 385.20,
            "TSLA": 242.80,
        }
        return {
            "symbol": symbol,
            "price": prices.get(symbol, 0.0),
            "currency": "USD",
        }

    print(f"\nRegistered tools via decorator:")
    for tool_def in custom_registry.list_tools():
        print(f"  - {tool_def.name}: {tool_def.description}")

    # Use with client
    config = AzureConfig()
    client = AzureLLMClient(config=config)

    messages = [{"role": "user", "content": "What's the stock price of AAPL and convert 25°C to Fahrenheit?"}]

    tools = custom_registry.get_tool_definitions()

    print(f"\nUser: {messages[0]['content']}")

    response = await client.chat_completion(
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    # Process tool calls (simplified)
    if response.raw_response and hasattr(response.raw_response.choices[0].message, "tool_calls"):
        tool_calls_data = response.raw_response.choices[0].message.tool_calls

        if tool_calls_data:
            print(f"\nExecuting {len(tool_calls_data)} tool calls...")

            for tc in tool_calls_data:
                tool_call = ToolCall.from_openai_tool_call(tc)
                result = await custom_registry.execute_tool_call(tool_call)
                print(f"  {tool_call.function_name}: {result.result}")


async def example_multi_turn_agent():
    """Multi-turn agent conversation with tools."""
    print("\n" + "=" * 80)
    print("Multi-Turn Agent Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    # Setup tools
    registry = ToolRegistry()
    registry.register_function(get_current_weather)
    registry.register_function(calculate)
    registry.register_function(get_current_time)
    registry.register_function(search_database)

    tools = registry.get_tool_definitions()

    # Agent loop
    messages = []
    user_queries = [
        "What's the weather in Oslo?",
        "What's the temperature difference between Oslo and Tokyo in Celsius?",
        "What time is it now?",
    ]

    for user_query in user_queries:
        print(f"\n{'─' * 60}")
        print(f"User: {user_query}")

        messages.append({"role": "user", "content": user_query})

        # Agent can make multiple tool calls in a loop
        max_iterations = 5
        for iteration in range(max_iterations):
            response = await client.chat_completion(
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            # Check if model wants to call tools
            if response.raw_response and hasattr(response.raw_response.choices[0].message, "tool_calls"):
                tool_calls_data = response.raw_response.choices[0].message.tool_calls

                if tool_calls_data:
                    # Add assistant message with tool calls
                    assistant_message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls_data
                        ],
                    }
                    messages.append(assistant_message)

                    # Execute tools
                    print(f"  [Iteration {iteration + 1}] Calling {len(tool_calls_data)} tool(s)...")

                    for tc in tool_calls_data:
                        tool_call = ToolCall.from_openai_tool_call(tc)
                        result = await registry.execute_tool_call(tool_call)

                        print(f"    → {tool_call.function_name}() = {result.result}")

                        messages.append(result.to_message())

                    continue  # Continue loop to get final response

            # No more tool calls - we have the final answer
            messages.append({"role": "assistant", "content": response.content})
            print(f"\nAssistant: {response.content}")
            break


async def example_structured_output():
    """Use function calling for structured output extraction."""
    print("\n" + "=" * 80)
    print("Structured Output Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    # Define a tool for structured output
    extract_info_tool = FunctionDefinition(
        name="extract_person_info",
        description="Extract structured information about a person from text",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's full name"},
                "age": {"type": "integer", "description": "Person's age"},
                "occupation": {"type": "string", "description": "Person's occupation"},
                "location": {"type": "string", "description": "Person's location/city"},
                "skills": {"type": "array", "items": {"type": "string"}, "description": "List of skills"},
            },
            "required": ["name", "age", "occupation"],
        },
    )

    text = """
    John Smith is a 35-year-old software engineer living in San Francisco.
    He specializes in Python, machine learning, and cloud architecture.
    """

    messages = [{"role": "user", "content": f"Extract information from this text:\n{text}"}]

    print(f"\nExtracting structured data from text...")
    print(f"Text: {text.strip()}")

    response = await client.chat_completion(
        messages=messages,
        tools=[extract_info_tool.to_dict()],
        tool_choice={"type": "function", "function": {"name": "extract_person_info"}},
    )

    # Extract structured data from tool call
    if response.raw_response and hasattr(response.raw_response.choices[0].message, "tool_calls"):
        tool_calls_data = response.raw_response.choices[0].message.tool_calls

        if tool_calls_data:
            tool_call = ToolCall.from_openai_tool_call(tool_calls_data[0])

            print(f"\nExtracted Information:")
            print(json.dumps(tool_call.function_arguments_parsed, indent=2))


async def main():
    """Run all examples."""
    examples = [
        ("Basic Function Calling", example_basic_function_calling),
        ("Decorator Registration", example_decorator_registration),
        ("Multi-Turn Agent", example_multi_turn_agent),
        ("Structured Output", example_structured_output),
    ]

    print("\n" + "=" * 80)
    print("Function Calling / Tools Examples")
    print("=" * 80)
    print("\nThese examples demonstrate Azure OpenAI function calling capabilities")
    print("for building agents, tools, and structured outputs.\n")

    for name, example_func in examples:
        try:
            await example_func()
            await asyncio.sleep(1)  # Brief pause between examples
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
