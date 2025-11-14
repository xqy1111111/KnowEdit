from typing import Dict

import torch

from data.base import BaseDataset


class ZSREDataset(BaseDataset):
    """ZSRE-style samples with edit/equiv/locality prompts."""

    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.LongTensor]]:
        row = self.data[idx]
        prompt = row.get("src") or row.get("prompt") or ""
        rephrase = row.get("rephrase") or row.get("rephrase_prompt") or prompt
        locality_prompt = row.get("loc") or row.get("loc_prompt") or ""
        answer = (
            row.get("ans")
            or row.get("alt")
            or row.get("target_new")
            or row.get("new_answer")
            or ""
        )
        locality_answer = (
            row.get("loc_ans")
            or row.get("loc_answer")
            or row.get("target_true")
            or ""
        )

        return {
            "edit_tuples": self.tok_tuples(prompt, answer),
            "equiv_tuples": self.tok_tuples(rephrase, answer),
            "unrel_tuples": self.tok_tuples(locality_prompt, locality_answer),
        }

    def tok_tuples(
        self,
        prompt: str,
        answer: str,
    ) -> Dict[str, torch.LongTensor]:
        answer = " " + answer if answer else ""
        tok_prompt = self.tok(
            prompt,
            return_tensors="pt",
        )
        tok_answer = self.tok(
            answer,
            return_tensors="pt",
            add_special_tokens=False,
        )

        tok_tuples = {
            key: torch.cat((value, tok_answer[key][:, :-1]), -1)
            for key, value in tok_prompt.items()
        }
        tok_tuples["labels"] = torch.cat(
            (
                torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
                tok_answer["input_ids"],
            ),
            -1,
        )

        return tok_tuples
