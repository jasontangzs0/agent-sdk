"""Tests for FileEditorTool subclass."""

import os
import tempfile
from uuid import uuid4

from pydantic import SecretStr

from openhands.sdk.agent import Agent
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.llm import LLM
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.file_editor import (
    FileEditorAction,
    FileEditorObservation,
    FileEditorTool,
)


def _create_test_conv_state(temp_dir: str) -> ConversationState:
    """Helper to create a test conversation state."""
    llm = LLM(model="gpt-4o-mini", api_key=SecretStr("test-key"), usage_id="test-llm")
    agent = Agent(llm=llm, tools=[])
    return ConversationState.create(
        id=uuid4(),
        agent=agent,
        workspace=LocalWorkspace(working_dir=temp_dir),
    )


def test_file_editor_tool_initialization():
    """Test that FileEditorTool initializes correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        # Check that the tool has the correct name and properties
        assert tool.name == "file_editor"
        assert tool.executor is not None
        assert issubclass(tool.action_type, FileEditorAction)


def test_file_editor_tool_create_file():
    """Test that FileEditorTool can create files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        test_file = os.path.join(temp_dir, "test.txt")

        # Create an action to create a file
        action = FileEditorAction(
            command="create",
            path=test_file,
            file_text="Hello, World!",
        )

        # Execute the action
        result = tool(action)

        # Check the result
        assert result is not None
        assert isinstance(result, FileEditorObservation)
        assert not result.is_error
        assert os.path.exists(test_file)

        # Check file contents
        with open(test_file) as f:
            content = f.read()
        assert content == "Hello, World!"


def test_file_editor_tool_view_file():
    """Test that FileEditorTool can view files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        test_file = os.path.join(temp_dir, "test.txt")

        # Create a test file
        with open(test_file, "w") as f:
            f.write("Line 1\nLine 2\nLine 3")

        # Create an action to view the file
        action = FileEditorAction(command="view", path=test_file)

        # Execute the action
        result = tool(action)

        # Check the result
        assert result is not None
        assert isinstance(result, FileEditorObservation)
        assert not result.is_error
        assert "Line 1" in result.text
        assert "Line 2" in result.text
        assert "Line 3" in result.text


def test_file_editor_tool_str_replace():
    """Test that FileEditorTool can perform string replacement."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        test_file = os.path.join(temp_dir, "test.txt")

        # Create a test file
        with open(test_file, "w") as f:
            f.write("Hello, World!\nThis is a test.")

        # Create an action to replace text
        action = FileEditorAction(
            command="str_replace",
            path=test_file,
            old_str="World",
            new_str="Universe",
        )

        # Execute the action
        result = tool(action)

        # Check the result
        assert result is not None
        assert isinstance(result, FileEditorObservation)
        assert not result.is_error

        # Check file contents
        with open(test_file) as f:
            content = f.read()
        assert "Hello, Universe!" in content


def test_file_editor_tool_to_openai_tool():
    """Test that FileEditorTool can be converted to OpenAI tool format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        # Convert to OpenAI tool format
        openai_tool = tool.to_openai_tool()

        # Check the format
        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "file_editor"
        assert "description" in openai_tool["function"]
        assert "parameters" in openai_tool["function"]


def test_file_editor_tool_view_directory():
    """Test that FileEditorTool can view directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        # Create some test files
        test_file1 = os.path.join(temp_dir, "file1.txt")
        test_file2 = os.path.join(temp_dir, "file2.txt")

        with open(test_file1, "w") as f:
            f.write("File 1 content")
        with open(test_file2, "w") as f:
            f.write("File 2 content")

        # Create an action to view the directory
        action = FileEditorAction(command="view", path=temp_dir)

        # Execute the action
        result = tool(action)

        # Check the result
        assert result is not None
        assert isinstance(result, FileEditorObservation)
        assert not result.is_error
        assert "file1.txt" in result.text
        assert "file2.txt" in result.text


def test_file_editor_tool_includes_working_directory_in_description():
    """Test that FileEditorTool includes working directory info in description."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        # Check that the tool description includes working directory information
        assert f"Your current working directory is: {temp_dir}" in tool.description
        assert (
            "When exploring project structure, start with this directory "
            "instead of the root filesystem."
        ) in tool.description

        # Verify the original description is still there
        assert (
            "Custom editing tool for viewing, creating and editing files"
            in tool.description
        )


def test_file_editor_tool_openai_format_includes_working_directory():
    """Test that OpenAI tool format includes working directory info."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        # Convert to OpenAI tool format
        openai_tool = tool.to_openai_tool()

        # Check that the description includes working directory information
        function_def = openai_tool["function"]
        assert "description" in function_def
        description = function_def["description"]
        assert f"Your current working directory is: {temp_dir}" in description
        assert (
            "When exploring project structure, start with this directory "
            "instead of the root filesystem."
        ) in description


def test_file_editor_tool_image_viewing_line_with_vision_enabled():
    """Test that image viewing line is included when LLM supports vision."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create LLM with vision support (gpt-4o-mini supports vision)
        llm = LLM(
            model="gpt-4o-mini", api_key=SecretStr("test-key"), usage_id="test-llm"
        )
        agent = Agent(llm=llm, tools=[])
        conv_state = ConversationState.create(
            id=uuid4(),
            agent=agent,
            workspace=LocalWorkspace(working_dir=temp_dir),
        )

        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        # Check that the image viewing line is included in description
        assert (
            "If `path` is an image file (.png, .jpg, .jpeg, .gif, .webp, .bmp)"
            in tool.description
        )
        assert "view` displays the image content" in tool.description


def test_file_editor_tool_image_viewing_line_with_vision_disabled():
    """Test that image viewing line is excluded when LLM doesn't support vision."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create LLM without vision support (gpt-3.5-turbo doesn't support vision)
        llm = LLM(
            model="gpt-3.5-turbo", api_key=SecretStr("test-key"), usage_id="test-llm"
        )
        agent = Agent(llm=llm, tools=[])
        conv_state = ConversationState.create(
            id=uuid4(),
            agent=agent,
            workspace=LocalWorkspace(working_dir=temp_dir),
        )

        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        # Check that the image viewing line is NOT included in description
        assert "is an image file" not in tool.description
        assert "displays the image content" not in tool.description


def test_file_editor_tool_move_lines_basic():
    """Test that FileEditorTool can move lines within a file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        test_file = os.path.join(temp_dir, "test.txt")

        # Create a test file with numbered lines
        with open(test_file, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")

        # Move lines 2-3 to after line 4
        action = FileEditorAction(
            command="move_lines",
            path=test_file,
            move_range=[2, 3],
            insert_line=4,
        )

        result = tool(action)

        assert result is not None
        assert isinstance(result, FileEditorObservation)
        assert not result.is_error
        assert "Moved lines 2-3" in result.text

        # Check file contents - should be: Line 1, Line 4, Line 2, Line 3, Line 5
        with open(test_file) as f:
            content = f.read()
        lines = content.strip().split("\n")
        assert lines == ["Line 1", "Line 4", "Line 2", "Line 3", "Line 5"]


def test_file_editor_tool_move_lines_to_beginning():
    """Test moving lines to the beginning of the file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        test_file = os.path.join(temp_dir, "test.txt")

        with open(test_file, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\nLine 4\n")

        # Move lines 3-4 to the beginning (after line 0)
        action = FileEditorAction(
            command="move_lines",
            path=test_file,
            move_range=[3, 4],
            insert_line=0,
        )

        result = tool(action)

        assert not result.is_error

        with open(test_file) as f:
            content = f.read()
        lines = content.strip().split("\n")
        assert lines == ["Line 3", "Line 4", "Line 1", "Line 2"]


def test_file_editor_tool_move_lines_invalid_range():
    """Test that move_lines fails with invalid range."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        test_file = os.path.join(temp_dir, "test.txt")

        with open(test_file, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\n")

        # Try to move lines beyond file length
        action = FileEditorAction(
            command="move_lines",
            path=test_file,
            move_range=[1, 10],  # Line 10 doesn't exist
            insert_line=0,
        )

        result = tool(action)

        assert result.is_error
        assert "move_range" in result.text.lower() or "range" in result.text.lower()


def test_file_editor_tool_move_lines_supports_undo():
    """Test that move_lines supports undo."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        test_file = os.path.join(temp_dir, "test.txt")
        original_content = "Line 1\nLine 2\nLine 3\n"

        with open(test_file, "w") as f:
            f.write(original_content)

        # Move some lines
        action = FileEditorAction(
            command="move_lines",
            path=test_file,
            move_range=[1, 1],
            insert_line=2,
        )
        tool(action)

        # Now undo
        undo_action = FileEditorAction(
            command="undo_edit",
            path=test_file,
        )
        undo_result = tool(undo_action)

        assert not undo_result.is_error

        # Check that content is restored
        with open(test_file) as f:
            content = f.read()
        assert content == original_content


def test_file_editor_tool_delete_lines_basic():
    """Test that FileEditorTool can delete lines from a file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        test_file = os.path.join(temp_dir, "test.txt")

        with open(test_file, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")

        # Delete lines 2-4
        action = FileEditorAction(
            command="delete_lines",
            path=test_file,
            delete_range=[2, 4],
        )

        result = tool(action)

        assert result is not None
        assert isinstance(result, FileEditorObservation)
        assert not result.is_error
        assert "Deleted lines 2-4" in result.text
        assert "(3 lines)" in result.text

        # Check file contents - should only have Line 1 and Line 5
        with open(test_file) as f:
            content = f.read()
        lines = content.strip().split("\n")
        assert lines == ["Line 1", "Line 5"]


def test_file_editor_tool_delete_lines_single_line():
    """Test deleting a single line."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        test_file = os.path.join(temp_dir, "test.txt")

        with open(test_file, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\n")

        # Delete only line 2
        action = FileEditorAction(
            command="delete_lines",
            path=test_file,
            delete_range=[2, 2],
        )

        result = tool(action)

        assert not result.is_error
        assert "(1 lines)" in result.text

        with open(test_file) as f:
            content = f.read()
        lines = content.strip().split("\n")
        assert lines == ["Line 1", "Line 3"]


def test_file_editor_tool_delete_lines_all_lines():
    """Test deleting all lines from a file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        test_file = os.path.join(temp_dir, "test.txt")

        with open(test_file, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\n")

        # Delete all lines
        action = FileEditorAction(
            command="delete_lines",
            path=test_file,
            delete_range=[1, 3],
        )

        result = tool(action)

        assert not result.is_error
        assert "file is now empty" in result.text.lower()

        with open(test_file) as f:
            content = f.read()
        assert content == ""


def test_file_editor_tool_delete_lines_invalid_range():
    """Test that delete_lines fails with invalid range."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        test_file = os.path.join(temp_dir, "test.txt")

        with open(test_file, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\n")

        # Try to delete lines beyond file length
        action = FileEditorAction(
            command="delete_lines",
            path=test_file,
            delete_range=[1, 10],  # Line 10 doesn't exist
        )

        result = tool(action)

        assert result.is_error
        assert "delete_range" in result.text.lower() or "range" in result.text.lower()


def test_file_editor_tool_delete_lines_start_greater_than_end():
    """Test that delete_lines fails when start > end."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        test_file = os.path.join(temp_dir, "test.txt")

        with open(test_file, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\n")

        # Try with start > end
        action = FileEditorAction(
            command="delete_lines",
            path=test_file,
            delete_range=[3, 1],  # Invalid: start > end
        )

        result = tool(action)

        assert result.is_error


def test_file_editor_tool_delete_lines_supports_undo():
    """Test that delete_lines supports undo."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        test_file = os.path.join(temp_dir, "test.txt")
        original_content = "Line 1\nLine 2\nLine 3\n"

        with open(test_file, "w") as f:
            f.write(original_content)

        # Delete some lines
        action = FileEditorAction(
            command="delete_lines",
            path=test_file,
            delete_range=[2, 2],
        )
        tool(action)

        # Verify deletion happened
        with open(test_file) as f:
            content = f.read()
        assert "Line 2" not in content

        # Now undo
        undo_action = FileEditorAction(
            command="undo_edit",
            path=test_file,
        )
        undo_result = tool(undo_action)

        assert not undo_result.is_error

        # Check that content is restored
        with open(test_file) as f:
            content = f.read()
        assert content == original_content


def test_file_editor_tool_delete_lines_preserves_diff():
    """Test that delete_lines properly stores old and new content for diff."""
    with tempfile.TemporaryDirectory() as temp_dir:
        conv_state = _create_test_conv_state(temp_dir)
        tools = FileEditorTool.create(conv_state)
        tool = tools[0]

        test_file = os.path.join(temp_dir, "test.txt")

        with open(test_file, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\n")

        action = FileEditorAction(
            command="delete_lines",
            path=test_file,
            delete_range=[2, 2],
        )

        result = tool(action)

        # Check that old_content and new_content are populated
        assert isinstance(result, FileEditorObservation)
        assert result.old_content is not None
        assert result.new_content is not None
        assert "Line 2" in result.old_content
        assert "Line 2" not in result.new_content
