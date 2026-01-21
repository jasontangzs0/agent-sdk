from typing import Any

from pydantic import BaseModel, Field, field_validator


class Tool(BaseModel):
    """Defines a tool to be initialized for the agent.

    This is only used in agent-sdk for type schema for server use.
    """

    name: str = Field(
        ...,
        description=(
            "Name of the tool class, e.g., 'TerminalTool'. "
            "Import it from an `openhands.tools.<module>` subpackage."
        ),
        examples=["TerminalTool", "FileEditorTool", "TaskTrackerTool"],
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the tool's .create() method,"
        " e.g., {'working_dir': '/app'}",
        examples=[{"working_dir": "/workspace"}],
    )
    runtime_params: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        description=(
            "Runtime-only parameters (callbacks, cancellation tokens, etc.) "
            "that are passed to .create() but excluded from serialization. "
            "Use this for non-serializable objects like functions or runtime state."
        ),
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is not empty."""
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v

    @field_validator("params", mode="before")
    @classmethod
    def validate_params(cls, v: dict[str, Any] | None) -> dict[str, Any]:
        """Convert None params to empty dict."""
        return v if v is not None else {}
