"""OpenHands Agent SDK — PDF Input Example.

This script demonstrates how to use PDFContent with LLM models to process
PDF documents with native document understanding.

PDFContent accepts PDF URLs that are accessible to the LLM provider.
This works with models like Claude (Bedrock/Anthropic) and Gemini that support
PDF input.

Note: The PDF URLs must be publicly accessible or accessible to the LLM provider.
"""

import os

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Message,
    PDFContent,
    TextContent,
    get_logger,
)


logger = get_logger(__name__)

# Configure LLM to use Gemini 2.5 Flash
model = "gemini/gemini-2.5-flash"
api_key = os.getenv("GEMINI_API_KEY")
assert api_key is not None, "GEMINI_API_KEY environment variable is not set."

llm = LLM(
    usage_id="pdf-llm",
    model=model,
    api_key=SecretStr(api_key),
)

cwd = os.getcwd()

agent = Agent(
    llm=llm,
    tools=[],
)

conversation = Conversation(agent=agent, workspace=cwd)

# Example 1: Processing a single PDF from URL
# -------------------------------------------------------------
print("=" * 80)
print("Example 1: Processing a single PDF from a public URL")
print("=" * 80)

if os.getenv("ANALYZE_SINGLE_PDF") == "true":
    # Using a publicly accessible PDF
    pdf_url = "https://arxiv.org/pdf/1706.03762"

    conversation.send_message(
        Message(
            role="user",
            content=[
                TextContent(
                    text=(
                        "Please analyze this PDF document and provide "
                        "a summary of its content."
                    )
                ),
                PDFContent(pdf_urls=[pdf_url]),
            ],
            vision_enabled=True,
        )
    )
    conversation.run()
    print("\n✓ Successfully processed PDF from URL")
else:
    print("⚠ Skipping Example 1: Set ANALYZE_SINGLE_PDF=true to enable")

# Example 2: Processing multiple PDFs
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("Example 2: Processing multiple PDF documents")
print("=" * 80)

if os.getenv("PROCESS_MULTIPLE_PDFS") == "true":
    # Multiple PDF URLs
    pdf_urls = [
        "https://arxiv.org/pdf/1706.03762",
        "https://arxiv.org/pdf/2110.12894",
    ]

    conversation.send_message(
        Message(
            role="user",
            content=[
                TextContent(
                    text=(
                        "Compare these two PDF documents and highlight "
                        "the key differences."
                    )
                ),
                PDFContent(pdf_urls=pdf_urls),
            ],
            vision_enabled=True,
        )
    )
    conversation.run()
    print("\n✓ Successfully processed multiple PDFs")
else:
    print("⚠ Skipping Example 2: Set PROCESS_MULTIPLE_PDFS=true to enable")

# Report cost
print("\n" + "=" * 80)
cost = llm.metrics.accumulated_cost
print(f"Total cost: ${cost:.4f}")
print("=" * 80)

print("\n" + "✓" * 40)
print("PDF processing examples completed!")
print("Note: To run these examples, ensure you have set:")
print("  - GEMINI_API_KEY environment variable")
print("  - ANALYZE_SINGLE_PDF=true: Enable single PDF example")
print("  - PROCESS_MULTIPLE_PDFS=true: Enable multiple PDF example")
print("✓" * 40)
