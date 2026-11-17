# ModernGPT-S 500M — Special-Token & Tool-Use Architecture Specification

**Phase:** Phase 1A Architecture & Feasibility  
**Date:** November 11, 2026  
**Status:** **SPECIFICATION COMPLETED**  
**Target Model:** ModernGPT-S 500M  
**Format Base:** ChatML + Structured Tool Delimiters  

---

## 1. Executive Summary & Design Principles

ModernGPT-S 500M incorporates native tool-calling capabilities. To ensure robust function invocation, unambiguous role transitions, and deterministic parsing without regex fragility, we specify a formal tool token grammar extending ChatML.

Key Architectural Principles:
1. **Unambiguous Delimiters:** Tool calls, tool results, and tool errors use dedicated reserved special tokens that cannot be generated as regular text.
2. **Deterministic Role Separation:** The `tool` role operates alongside `system`, `user`, and `assistant`.
3. **Structured JSON Payloads:** Tool arguments and outputs must adhere strictly to valid JSON payloads, enabling automated execution in sandboxes and runtime JSON Schema validation.
4. **Canonical Loss Masking:** In supervised fine-tuning (SFT), only assistant-generated tokens (the thought process and the `<|tool_call|>` invocation payload) receive gradient backpropagation. Tool results injected by the environment are strictly masked with `ignore_index = -100`.
5. **Multi-Turn & Parallel Execution Support:** The grammar natively supports parallel tool calls in a single turn and conversational recovery when tool execution fails.

---

## 2. Special Token Inventory & Vocabulary Reservations

The following special tokens are permanently reserved at the top of the 32K vocabulary:

| Special Token | Token ID | Role / Function | Training Loss Masking | Description |
| :--- | :---: | :--- | :---: | :--- |
| `<|pad|>` | 0 | Padding | Masked (`-100`) | Sequence alignment token |
| `<|unk|>` | 1 | Unknown | Masked / Supervised | Out-of-vocabulary fallback |
| `<|bos|>` | 2 | Beginning of String | Masked (`-100`) | Document / sequence starter |
| `<|eos|>` | 3 | End of String | Supervised | Autoregressive generation halt |
| `<|im_start|>` | 4 | Turn Boundary | Supervised (assistant) / Masked (others) | Marks opening of a ChatML message block |
| `<|im_end|>` | 5 | Turn Boundary | Supervised (assistant) / Masked (others) | Marks closing of a ChatML message block |
| `<|tool_call|>` | 6 | Control Delimiter | **Supervised** | Initiates a structured tool call payload |
| `<|tool_result|>` | 7 | Control Delimiter | Masked (`-100`) | Injects external tool execution output |
| `<|tool_error|>` | 8 | Control Delimiter | Masked (`-100`) | Injects tool execution or validation failure |

*Note: Tokens 9–31 are reserved for future control extensions (e.g., `<|thought_start|>`, `<|thought_end|>`).*

---

## 3. Formal EBNF Grammar

The conversation serialization complies with the following Extended Backus-Naur Form (EBNF):

```ebnf
Conversation      ::= [ SystemMessage ] { Message }
Message           ::= UserMessage | AssistantMessage | ToolMessage

SystemMessage     ::= "<|im_start|>system\n" SystemContent "<|im_end|>\n"
UserMessage       ::= "<|im_start|>user\n" TextContent "<|im_end|>\n"

AssistantMessage  ::= "<|im_start|>assistant\n" 
                      [ TextContent ] 
                      [ ToolCallBlock ] 
                      "<|im_end|>\n"

ToolCallBlock     ::= "<|tool_call|>\n" ToolCallPayload "\n<|im_end|>"
ToolCallPayload   ::= SingleToolCall | MultiToolCall
SingleToolCall    ::= "```json\n" JsonToolCall "\n```"
MultiToolCall     ::= "```json\n[" JsonToolCall { "," JsonToolCall } "]\n```"
JsonToolCall      ::= '{"name": "' ToolName '", "arguments": ' JsonObject '}'

ToolMessage       ::= "<|im_start|>tool\n" 
                      ( ToolSuccessBlock | ToolErrorBlock ) 
                      "<|im_end|>\n"

ToolSuccessBlock  ::= "<|tool_result|>\n" JsonToolResult
ToolErrorBlock    ::= "<|tool_error|>\n" JsonToolError

JsonToolResult    ::= '{"name": "' ToolName '", "output": ' JsonValue '}'
JsonToolError     ::= '{"name": "' ToolName '", "error": "' ErrorString '"}'
```

---

## 4. Multi-Turn Serialization Examples

### 4.1 Example 1: Web Search with Grounded Assistant Response

```text
<|im_start|>system
You are ModernGPT-S, a capable and truthful assistant with access to real-time tools.
Available tools:
- web_search(query: string): Search the live web for recent information.
- calculate(expression: string): Evaluate mathematical expressions.
<|im_end|>
<|im_start|>user
Who won the latest Nobel Prize in Chemistry in 2024 and what was it awarded for?
<|im_end|>
<|im_start|>assistant
Let me search the web for the official announcement of the 2024 Nobel Prize in Chemistry.
<|tool_call|>
```json
{
  "name": "web_search",
  "arguments": {
    "query": "2024 Nobel Prize in Chemistry winners announcement"
  }
}
```
<|im_end|>
<|im_start|>tool
<|tool_result|>
{
  "name": "web_search",
  "output": {
    "results": [
      {
        "title": "The Nobel Prize in Chemistry 2024",
        "snippet": "The Nobel Prize in Chemistry 2024 was divided, one half awarded to David Baker for computational protein design, and the other half jointly to Demis Hassabis and John M. Jumper for protein structure prediction."
      }
    ]
  }
}
<|im_end|>
<|im_start|>assistant
The 2024 Nobel Prize in Chemistry was awarded to three scientists for breakthroughs in protein research:

- **David Baker** (one half) for *computational protein design*.
- **Demis Hassabis** and **John M. Jumper** (jointly the other half) for *protein structure prediction* using AlphaFold.
<|im_end|>
```

---

### 4.2 Example 2: Python Code Execution & Data Analysis

```text
<|im_start|>user
Can you compute the standard deviation of the numbers [14.2, 18.5, 12.1, 19.8, 22.4, 15.6]?
<|im_end|>
<|im_start|>assistant
I will calculate the sample standard deviation using Python.
<|tool_call|>
```json
{
  "name": "python_interpreter",
  "arguments": {
    "code": "import statistics\ndata = [14.2, 18.5, 12.1, 19.8, 22.4, 15.6]\nresult = statistics.stdev(data)\nprint(f'{result:.4f}')"
  }
}
```
<|im_end|>
<|im_start|>tool
<|tool_result|>
{
  "name": "python_interpreter",
  "output": "3.7538"
}
<|im_end|>
<|im_start|>assistant
The sample standard deviation of the dataset $[14.2, 18.5, 12.1, 19.8, 22.4, 15.6]$ is approximately **$3.7538$**.
<|im_end|>
```

---

### 4.3 Example 3: Tool Error Handling and Self-Correction

```text
<|im_start|>user
Calculate the square root of -25 using the calculator.
<|im_end|>
<|im_start|>assistant
<|tool_call|>
```json
{
  "name": "calculator",
  "arguments": {
    "expression": "sqrt(-25)"
  }
}
```
<|im_end|>
<|im_start|>tool
<|tool_error|>
{
  "name": "calculator",
  "error": "ValueError: math domain error. Cannot compute real square root of negative number."
}
<|im_end|>
<|im_start|>assistant
The real square root of $-25$ is undefined because square roots of negative numbers do not yield real numbers. In complex numbers, the principal square root is:
$$\sqrt{-25} = 5i$$
where $i$ represents the imaginary unit $\sqrt{-1}$.
<|im_end|>
```

---

## 5. Parallel Tool Calling Specification

When a query requires independent lookups, the assistant may emit a JSON array of tool calls within a single `<|tool_call|>` block:

```text
<|im_start|>assistant
I will look up both Tokyo and London local times simultaneously.
<|tool_call|>
```json
[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "London"}}
]
```
<|im_end|>
<|im_start|>tool
<|tool_result|>
[
  {"name": "get_weather", "output": {"temperature_c": 19, "condition": "Clear"}},
  {"name": "get_weather", "output": {"temperature_c": 11, "condition": "Rain"}}
]
<|im_end|>
```

---

## 6. SFT Loss Masking Contract for Tool Use

To train the model to generate correct tool invocations without memorizing external execution results:

| Segment | Tokens | Loss Mask Value | Justification |
| :--- | :--- | :---: | :--- |
| System Prompt | `<|im_start|>system ... <|im_end|>` | `-100` (Masked) | Context provided to model |
| User Prompt | `<|im_start|>user ... <|im_end|>` | `-100` (Masked) | Input query |
| Assistant Reasoning | `<|im_start|>assistant\n` ... | **Supervised** | Model must learn when to use tools |
| Tool Call Delimiter | `<|tool_call|>` | **Supervised** | Model must learn tool triggering |
| Tool Invocation JSON | `{"name": ..., "arguments": ...}` | **Supervised** | Model must learn schema compliance |
| Tool Turn Delimiter | `<|im_end|>` | **Supervised** | Model must learn to yield turn |
| Tool Result Envelope | `<|im_start|>tool\n<|tool_result|>` | `-100` (Masked) | External environment feedback |
| Tool JSON Output | `{"name": ..., "output": ...}` | `-100` (Masked) | External environment feedback |
| Final Assistant Turn | `<|im_start|>assistant\n` ... `<|im_end|>` | **Supervised** | Model synthesis & response |

---

## 7. Runtime Validation & Guardrails

At inference time, the orchestration engine executes the following verification pipeline before executing any tool:

1. **Delimiter Detection:** Inspect generation for `<|tool_call|>`.
2. **JSON Syntactic Validation:** Parse the enclosed block with `json.loads()`. If syntax fails, inject `<|tool_error|>` with `"JSONDecodeError: invalid syntax"`.
3. **Schema Compliance:** Validate the presence of `"name"` (string) and `"arguments"` (dictionary) against the tool's registered JSON Schema using `jsonschema.validate()`.
4. **Execution Sandbox:** Execute tool in an isolated process with a strict 10-second timeout.
5. **Format Result:** Return either `<|tool_result|>` or `<|tool_error|>` formatted as valid JSON back into the context window.

