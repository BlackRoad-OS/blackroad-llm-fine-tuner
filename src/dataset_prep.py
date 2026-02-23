"""Dataset preparation utilities for fine-tuning."""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re


class DatasetPrep:
    """Prepare and validate datasets for LLM fine-tuning."""

    # Token estimation: roughly 1.3 tokens per word
    TOKENS_PER_WORD = 1.3

    @staticmethod
    def load_jsonl(path: str) -> List[Dict[str, Any]]:
        """Load conversation pairs from JSONL file."""
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def validate(records: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate dataset format.
        
        Each record must have either:
        - {prompt, response}
        - {messages: [...]}
        
        Returns (is_valid, error_messages)
        """
        errors = []

        for idx, record in enumerate(records):
            if isinstance(record, dict):
                # Check for chat format (messages)
                if "messages" in record:
                    if not isinstance(record["messages"], list):
                        errors.append(
                            f"Record {idx}: 'messages' must be a list"
                        )
                    elif len(record["messages"]) == 0:
                        errors.append(f"Record {idx}: 'messages' list is empty")
                    else:
                        for msg in record["messages"]:
                            if not isinstance(msg, dict):
                                errors.append(
                                    f"Record {idx}: message must be dict"
                                )
                            elif "role" not in msg or "content" not in msg:
                                errors.append(
                                    f"Record {idx}: message missing 'role' or 'content'"
                                )

                # Check for prompt/response format
                elif "prompt" in record and "response" in record:
                    if not isinstance(record["prompt"], str):
                        errors.append(
                            f"Record {idx}: 'prompt' must be string"
                        )
                    if not isinstance(record["response"], str):
                        errors.append(
                            f"Record {idx}: 'response' must be string"
                        )

                else:
                    errors.append(
                        f"Record {idx}: must have 'messages' or ('prompt', 'response')"
                    )
            else:
                errors.append(f"Record {idx}: must be a dictionary")

        return len(errors) == 0, errors

    @staticmethod
    def format_chat(
        records: List[Dict[str, Any]], template: str = "llama"
    ) -> List[Dict[str, str]]:
        """Format records for different model families.
        
        Supported templates: llama, qwen, mistral, phi, deepseek
        """
        templates = {
            "llama": "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>",
            "qwen": "<|im_start|>{role}\n{content}<|im_end|>",
            "mistral": "[INST] {content} [/INST]",
            "phi": "<|user|>\n{content}<|end|>\n<|assistant|>",
            "deepseek": "<｜User｜>{content}<｜Assistant｜>",
        }

        fmt = templates.get(template, templates["llama"])
        formatted = []

        for record in records:
            if "messages" in record:
                # Convert messages array to text
                text = ""
                for msg in record["messages"]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    text += fmt.format(role=role, content=content) + "\n"
                formatted.append({"text": text})
            elif "prompt" in record and "response" in record:
                # Combine prompt and response
                text = fmt.format(role="user", content=record["prompt"]) + "\n"
                text += fmt.format(role="assistant", content=record["response"])
                formatted.append({"text": text})

        return formatted

    @staticmethod
    def split(
        records: List[Dict[str, Any]], train: float = 0.9, val: float = 0.1
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split records into train and validation sets."""
        if abs(train + val - 1.0) > 0.01:
            raise ValueError("train + val must sum to 1.0")

        split_idx = int(len(records) * train)
        return records[:split_idx], records[split_idx:]

    @staticmethod
    def stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate dataset statistics."""
        if not records:
            return {
                "total_examples": 0,
                "avg_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "total_tokens": 0,
            }

        token_counts = []

        for record in records:
            text = ""
            if "messages" in record:
                for msg in record["messages"]:
                    text += msg.get("content", "") + " "
            elif "prompt" in record:
                text += record.get("prompt", "") + " "
                text += record.get("response", "") + " "
            elif "text" in record:
                text = record["text"]

            # Estimate tokens: word count * 1.3
            word_count = len(text.split())
            token_count = int(word_count * DatasetPrep.TOKENS_PER_WORD)
            token_counts.append(token_count)

        total_tokens = sum(token_counts)
        avg_tokens = total_tokens / len(token_counts) if token_counts else 0

        return {
            "total_examples": len(records),
            "avg_tokens": round(avg_tokens, 1),
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def export(
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        output_dir: str,
    ) -> Dict[str, str]:
        """Export train and validation datasets to JSONL files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_path = output_path / "train.jsonl"
        val_path = output_path / "val.jsonl"

        with open(train_path, "w") as f:
            for record in train_records:
                f.write(json.dumps(record) + "\n")

        with open(val_path, "w") as f:
            for record in val_records:
                f.write(json.dumps(record) + "\n")

        return {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_count": len(train_records),
            "val_count": len(val_records),
        }
