"""
Function calling / tools support for Azure OpenAI.

This module provides first-class support for Azure OpenAI function calling,
enabling agent architectures, structured output generation, and tool use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


# -----------------------------
# Tool Definitions
# -----------------------------


class ToolChoiceType(str, Enum):
    """Tool choice types for chat completions."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


@dataclass
class ParameterProperty:
    """Property definition for a tool parameter."""

    type: str  # "string", "number", "integer", "boolean", "array", "object"
    description: str
    enum: list[str] | None = None
    items: dict[str, Any] | None = None  # For array types
    properties: dict[str, ParameterProperty] | None = None  # For object types
    required: list[str] | None = None  # For object types

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        result: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum is not None:
            result["enum"] = self.enum
        if self.items is not None:
            result["items"] = self.items
        if self.properties is not None:
            result["properties"] = {k: v.to_dict() for k, v in self.properties.items()}
        if self.required is not None:
            result["required"] = self.required
        return result


@dataclass
class FunctionDefinition:
    """Function definition for tool calling."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    handler: Callable[..., Any] | Callable[..., Awaitable[Any]] | None = None

    @classmethod
    def from_python_function(
        cls,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> FunctionDefinition:
        """Create FunctionDefinition from a Python function with type hints."""
        import inspect

        func_name = name or func.__name__
        func_desc = description or (func.__doc__ or "").strip().split("\n")[0]

        # Parse function signature
        sig = inspect.signature(func)
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = "string"  # Default
            param_desc = ""

            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                annotation = param.annotation
                if annotation == str:
                    param_type = "string"
                elif annotation == int:
                    param_type = "integer"
                elif annotation == float:
                    param_type = "number"
                elif annotation == bool:
                    param_type = "boolean"
                elif hasattr(annotation, "__origin__"):
                    if annotation.__origin__ == list:
                        param_type = "array"

            parameters["properties"][param_name] = {
                "type": param_type,
                "description": param_desc or f"{param_name} parameter",
            }

            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        return cls(
            name=func_name,
            description=func_desc,
            parameters=parameters,
            handler=func,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """Represents a tool call from the model."""

    id: str
    type: str
    function_name: str
    function_arguments: str
    function_arguments_parsed: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_openai_tool_call(cls, tool_call: Any) -> ToolCall:
        """Create from OpenAI API tool call object."""
        func = tool_call.function
        args_str = func.arguments

        # Try to parse arguments
        try:
            args_parsed = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call arguments: {args_str}")
            args_parsed = {}

        return cls(
            id=tool_call.id,
            type=tool_call.type,
            function_name=func.name,
            function_arguments=args_str,
            function_arguments_parsed=args_parsed,
        )


@dataclass
class ToolCallResult:
    """Result from executing a tool call."""

    tool_call_id: str
    function_name: str
    result: Any
    error: str | None = None

    def to_message(self) -> dict[str, Any]:
        """Convert to message format for chat completion."""
        content = json.dumps(self.result) if not self.error else self.error
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.function_name,
            "content": content,
        }


# -----------------------------
# Tool Registry
# -----------------------------


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: dict[str, FunctionDefinition] = {}

    def register(self, tool: FunctionDefinition) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def register_function(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> FunctionDefinition:
        """Register a Python function as a tool."""
        tool = FunctionDefinition.from_python_function(func, name, description)
        self.register(tool)
        return tool

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")

    def get(self, name: str) -> FunctionDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[FunctionDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions in OpenAI API format."""
        return [tool.to_dict() for tool in self._tools.values()]

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolCallResult:
        """Execute a tool call and return the result."""
        tool = self.get(tool_call.function_name)

        if tool is None:
            return ToolCallResult(
                tool_call_id=tool_call.id,
                function_name=tool_call.function_name,
                result=None,
                error=f"Tool '{tool_call.function_name}' not found",
            )

        if tool.handler is None:
            return ToolCallResult(
                tool_call_id=tool_call.id,
                function_name=tool_call.function_name,
                result=None,
                error=f"Tool '{tool_call.function_name}' has no handler",
            )

        try:
            # Execute the tool handler
            import inspect

            if inspect.iscoroutinefunction(tool.handler):
                result = await tool.handler(**tool_call.function_arguments_parsed)
            else:
                result = tool.handler(**tool_call.function_arguments_parsed)

            return ToolCallResult(
                tool_call_id=tool_call.id,
                function_name=tool_call.function_name,
                result=result,
            )

        except Exception as e:
            logger.error(f"Error executing tool {tool_call.function_name}: {e}")
            return ToolCallResult(
                tool_call_id=tool_call.id,
                function_name=tool_call.function_name,
                result=None,
                error=str(e),
            )


# -----------------------------
# Decorator for Tool Registration
# -----------------------------


def tool(
    name: str | None = None,
    description: str | None = None,
    registry: ToolRegistry | None = None,
):
    """Decorator to register a function as a tool.

    Args:
        name: Optional tool name (defaults to function name)
        description: Optional description (defaults to docstring)
        registry: Optional registry to register with (defaults to global registry)

    Example:
        @tool(description="Get the current weather")
        def get_weather(location: str, unit: str = "celsius") -> dict:
            return {"temperature": 22, "condition": "sunny"}
    """

    def decorator(func: Callable) -> Callable:
        nonlocal registry
        if registry is None:
            registry = get_default_registry()

        registry.register_function(func, name=name, description=description)
        return func

    return decorator


# -----------------------------
# Global Registry
# -----------------------------

_default_registry: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """Get the default global tool registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def reset_default_registry() -> None:
    """Reset the default registry (useful for testing)."""
    global _default_registry
    _default_registry = ToolRegistry()


__all__ = [
    "ToolChoiceType",
    "ParameterProperty",
    "FunctionDefinition",
    "ToolCall",
    "ToolCallResult",
    "ToolRegistry",
    "tool",
    "get_default_registry",
    "reset_default_registry",
]
