"""Conversational serializer and loss masking for ModernGPT-S 100M 4K SFT.

Supports:
- ChatML standard syntax: <|im_start|>role\ncontent<|im_end|>\n
- Roles: system, user, assistant, tool_call, tool_observation (or tool)
- Multi-turn conversation trees and sequences
- Assistant-only loss masking (-100 for system, user, tool observations, padding)
- Exact causal shift target alignment
- Deterministic truncation up to max_length (4096)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VALID_ROLES = {"system", "user", "assistant", "tool_call", "tool_observation", "tool"}


@dataclass
class Message:
    role: str
    content: str

    def __post_init__(self):
        self.role = self.role.strip().lower()
        if self.role == "tool":
            self.role = "tool_observation"
        if self.role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {self.role}. Expected one of {VALID_ROLES}")
        self.content = self.content.strip() if self.content else ""


class ConversationSerializer:
    """Serializes multi-turn conversations into ChatML token sequences with assistant loss masks."""

    def __init__(self, tokenizer, max_length: int = 4096, default_system: Optional[str] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.default_system = default_system or "You are ModernGPT-S, a helpful and precise AI assistant."
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        # Pre-tokenize markers for boundary validation
        self.im_start_tokens = tokenizer.encode(IM_START)
        self.im_end_tokens = tokenizer.encode(IM_END)

    def validate_conversation(self, messages: List[Union[Dict[str, str], Message]]) -> List[Message]:
        """Validate and normalize conversation messages."""
        if not messages:
            raise ValueError("Conversation messages cannot be empty.")

        normalized: List[Message] = []
        for m in messages:
            if isinstance(m, dict):
                normalized.append(Message(role=m.get("role", ""), content=m.get("content", "")))
            elif isinstance(m, Message):
                normalized.append(m)
            else:
                raise TypeError(f"Unsupported message type: {type(m)}")

        # Ensure first turn is system; if not, inject default system prompt
        if normalized[0].role != "system":
            normalized.insert(0, Message(role="system", content=self.default_system))

        # Check that there is at least one assistant turn
        has_assistant = any(m.role in {"assistant", "tool_call"} for m in normalized)
        if not has_assistant:
            raise ValueError("Conversation must contain at least one assistant or tool_call turn.")

        return normalized

    def serialize_turn(self, message: Message) -> Tuple[str, bool]:
        """Serialize a single turn into text and determine if it should be supervised.
        
        Returns:
            turn_text: The serialized string for this turn.
            is_supervised: True if the body and closing delimiter should contribute to loss.
        """
        role_tag = message.role
        is_supervised = message.role in {"assistant", "tool_call"}

        # Format ChatML block
        turn_text = f"{IM_START}{role_tag}\n{message.content}{IM_END}\n"
        return turn_text, is_supervised

    def encode_conversation(
        self,
        messages: List[Union[Dict[str, str], Message]],
        max_length: Optional[int] = None,
        append_eos: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Encode conversation into token IDs and assistant-only loss masks.
        
        Returns a dict with:
            input_ids: 1D torch.LongTensor of token IDs.
            labels: 1D torch.LongTensor with -100 for masked tokens, and token_id for supervised tokens.
            attention_mask: 1D torch.LongTensor (1 for valid tokens, 0 for pad).
        """
        limit = max_length or self.max_length
        valid_msgs = self.validate_conversation(messages)

        all_input_ids: List[int] = []
        all_labels: List[int] = []

        for msg in valid_msgs:
            role_header = f"{IM_START}{msg.role}\n"
            content_body = msg.content
            closing = f"{IM_END}\n"

            header_ids = self.tokenizer.encode(role_header)
            content_ids = self.tokenizer.encode(content_body) if content_body else []
            closing_ids = self.tokenizer.encode(closing)

            # Header is always context (masked out from loss)
            all_input_ids.extend(header_header := header_ids)
            all_labels.extend([-100] * len(header_ids))

            if msg.role in {"assistant", "tool_call"}:
                # Supervise assistant content and closing token
                all_input_ids.extend(content_ids)
                all_labels.extend(content_ids)

                all_input_ids.extend(closing_ids)
                all_labels.extend(closing_ids)
            else:
                # System, user, or tool observation are context (masked)
                all_input_ids.extend(content_ids)
                all_labels.extend([-100] * len(content_ids))

                all_input_ids.extend(closing_ids)
                all_labels.extend([-100] * len(closing_ids))

        if append_eos:
            all_input_ids.append(self.eos_id)
            # Supervise EOS if the last turn was assistant
            if valid_msgs[-1].role in {"assistant", "tool_call"}:
                all_labels.append(self.eos_id)
            else:
                all_labels.append(-100)

        # Truncation if exceeding limit
        if len(all_input_ids) > limit:
            all_input_ids = all_input_ids[:limit]
            all_labels = all_labels[:limit]

        input_ids_t = torch.tensor(all_input_ids, dtype=torch.long)
        labels_t = torch.tensor(all_labels, dtype=torch.long)
        attention_mask_t = torch.ones_like(input_ids_t)

        return {
            "input_ids": input_ids_t,
            "labels": labels_t,
            "attention_mask": attention_mask_t,
            "total_tokens": len(all_input_ids),
            "supervised_tokens": (labels_t != -100).sum().item(),
        }

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def deserialize_conversation(self, serialized_input: Union[str, List[int], torch.Tensor]) -> List[Dict[str, str]]:
        """Deserializes a ChatML tokenized sequence or string back into structured messages."""
        if isinstance(serialized_input, (list, torch.Tensor)):
            raw_text = self.decode(serialized_input, skip_special_tokens=False)
        else:
            raw_text = str(serialized_input)

        import re
        # Pattern matching <|im_start|>role\ncontent<|im_end|>
        pattern = r"<\|im_start\|>([a-zA-Z0-9_\-]+)\n(.*?)(?:<\|im_end\|>|<\|endoftext\|>|$)"
        matches = re.findall(pattern, raw_text, re.DOTALL)
        
        messages = []
        for role, content in matches:
            role = role.strip().lower()
            if role == "tool_observation":
                role = "tool"
            messages.append({
                "role": role,
                "content": content.strip()
            })
        return messages

