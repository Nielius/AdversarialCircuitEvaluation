import random
from functools import cached_property
from string import Template
from typing import Iterable

import torch
from jaxtyping import Int
from transformers import AutoTokenizer


class IOIPromptTemplate:
    """pre_template should be something like

        "Then, ${name1} and ${name2} went to the ${place}. ${name3} gave a ${object} to ${name4}",

    And names_order can be something like ["A", "B", "B", "A"]
    """

    _template: Template
    names_order: list[str]
    _io_position: int  # e.g. if this is 3, then ${name3} represents the indirect object
    _subject_position: int

    def __init__(self, pre_template: Template, names_order: Iterable[str], io_position: int, subject_position: int):
        self._template = pre_template
        self.names_order = list(names_order)
        self._io_position = io_position
        self._subject_position = subject_position

    def render(self, **values: str) -> "IOIRenderedPromptTemplate":
        """The 'values' argument is expected to use the names in 'names_order' as keys (not name1, name2, ...)."""
        return IOIRenderedPromptTemplate(self, values)

    def with_different_order(self, new_names_order: Iterable[str]) -> "IOIPromptTemplate":
        return IOIPromptTemplate(self._template, new_names_order, self._io_position, self._subject_position)

    @property
    def io_name_key(self) -> str:
        """E.g. if the names_order is ABBA and the _ioi_position is 4 (corresponding to name4),
        then io_name_key is A"""
        return self.names_order[self._io_position - 1]

    @property
    def subject_name_key(self) -> str:
        """E.g. if the names_order is ABBA and the _ioi_position is 4 (corresponding to name4),
        then io_name_key is A"""
        return self.names_order[self._subject_position - 1]


class IOIRenderedPromptTemplate:
    template: IOIPromptTemplate
    values: dict[str, str]

    def __init__(self, template: IOIPromptTemplate, values: dict[str, str]):
        self.template = template
        self.values = values

    @cached_property
    def text(self) -> str:
        # If names_order is ["A", "B", "B", "A"], that means mapping "name1" -> "A", "name2" -> "B", "name3" -> "B", "name4" -> "A"
        template_values = {f"name{i + 1}": self.values[name] for i, name in enumerate(self.template.names_order)} | {
            k: v for k, v in self.values.items() if k not in self.template.names_order
        }
        return self.template._template.substitute(template_values)

    @cached_property
    def indirect_object(self) -> str:
        return self.values[self.template.io_name_key]

    @cached_property
    def subject(self) -> str:
        return self.values[self.template.subject_name_key]

    def patch_with_new_order(
        self, new_order: Iterable[str], new_values: dict[str, str] | None = None
    ) -> "IOIRenderedPromptTemplate":
        new_template = self.template.with_different_order(new_order)
        return self.patch(new_template, new_values)

    def patch(
        self, new_template: IOIPromptTemplate, new_values: dict[str, str] | None = None
    ) -> "IOIRenderedPromptTemplate":
        return IOIRenderedPromptTemplate(
            template=new_template,
            values=self.values | (new_values or {}),
        )


IOI_PROMPT_PRETEMPLATES: list[Template] = [
    Template("Then, ${name1} and ${name2} had a lot of fun at the ${place}. ${name3} gave a ${object} to ${name4}"),
    Template(
        "Then, ${name1} and ${name2} were working at the ${place}. ${name3} decided to give a ${object} to ${name4}"
    ),
    Template("Then, ${name1} and ${name2} went to the ${place}. ${name3} gave a ${object} to ${name4}"),
    Template(
        "Then, ${name1} and ${name2} were thinking about going to the ${place}. ${name3} wanted to give a ${object} to ${name4}"
    ),
    Template("Then, ${name1} and ${name2} had a long argument, and afterwards ${name3} said to ${name4}"),
    Template("After ${name1} and ${name2} went to the ${place}, ${name3} gave a ${object} to ${name4}"),
    Template("When ${name1} and ${name2} got a ${object} at the ${place}, ${name3} decided to give it to ${name4}"),
    Template(
        "When ${name1} and ${name2} got a ${object} at the ${place}, ${name3} decided to give the ${object} to ${name4}"
    ),
    Template("While ${name1} and ${name2} were working at the ${place}, ${name3} gave a ${object} to ${name4}"),
    Template("While ${name1} and ${name2} were commuting to the ${place}, ${name3} gave a ${object} to ${name4}"),
    Template("After the lunch, ${name1} and ${name2} went to the ${place}. ${name3} gave a ${object} to ${name4}"),
    Template("Afterwards, ${name1} and ${name2} went to the ${place}. ${name3} gave a ${object} to ${name4}"),
    Template("Then, ${name1} and ${name2} had a long argument. Afterwards ${name3} said to ${name4}"),
    Template("The ${place} ${name1} and ${name2} went to had a ${object}. ${name3} gave it to ${name4}"),
    Template("Friends ${name1} and ${name2} found a ${object} at the ${place}. ${name3} gave it to ${name4}"),
]

# Additional of my own
IOI_PROMPT_PRETEMPLATES_OOD: list[Template] = [
    Template("That evening, when ${name1} and ${name2} where at the ${place}, ${name3} handed a ${object} to ${name4}"),
    Template(
        "At the ${place}, where ${name1} and ${name2} had decided to meet, ${name3} handed a ${object} to ${name4}"
    ),
    Template(
        "There was a ${object} at the ${place}, where ${name1} and ${name2} had decided to meet. ${name3} found it and gave it to ${name4}"
    ),
    Template(
        "Then, ${name1} and ${name2} met at the ${place} and argued over the ${object}. ${name3} gave it to ${name4}"
    ),
    Template(
        "Then, ${name1} and ${name2} met at the ${place} and argued over the ${object}. It was getting late. ${name3} gave it to ${name4}"
    ),
]


def _select_random_values(
    rng: random.Random, names_order: list[str], names: list[str], template_values: dict[str, list[str]]
) -> dict[str, str]:
    """Selects random values for the template values and the names.

    The reason names are treated differently from the other template values
    is that we always want to select distinct names that need to be spread across
    the name keys in 'names_order'."""
    random_names = iter(rng.sample(names, k=len(names_order)))

    return {key: rng.choice(values) for key, values in template_values.items()} | {
        name_key: next(random_names) for name_key in names_order
    }


def generate_prompts_uniform_v2(
    template: IOIPromptTemplate,
    template_values: dict[str, list[str]],
    names: list[str],
    num_prompts: int,
    rng: random.Random,
) -> list[IOIRenderedPromptTemplate]:
    return [
        template.render(
            **_select_random_values(rng, names_order=template.names_order, names=names, template_values=template_values)
        )
        for _ in range(num_prompts)
    ]


def get_ioi_tokenizer() -> AutoTokenizer:
    """Returns the tokenizer that is used by the old IOIDataset (if no tokenizer is passed in)."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


class IOIPromptCollection:
    prompts: list[IOIRenderedPromptTemplate]
    num_examples: int

    _prepend_bos: bool

    def __init__(
        self,
        prompts: list[IOIRenderedPromptTemplate],
        num_examples: int,
        prepend_bos: bool = False,
    ):
        self.prompts = prompts
        self.num_examples = num_examples
        self._prepend_bos = prepend_bos

    def tokens(self, tokenizer: AutoTokenizer, device: str) -> Int[torch.Tensor, "batch seq"]:
        # Maybe use tokenizer.add_bos_token? (I'm using prepend_bos to keep backwards compatibility to some extent.)
        prompt_texts = [(tokenizer.bos_token if self._prepend_bos else "") + prompt.text for prompt in self.prompts]
        # you can add padding=True to the tokenizer, but I'm not sure if that breaks implicit assumptions elsewhere...
        return torch.tensor(tokenizer(prompt_texts).input_ids, device=device).type(torch.int)

    def io_tokenIDs(self, tokenizer: AutoTokenizer) -> list[int]:
        """Token IDs for the indirect object of the prompt."""
        return [tokenizer.encode(" " + prompt.indirect_object)[0] for prompt in self.prompts]

    def s_tokenIDs(self, tokenizer: AutoTokenizer) -> list[int]:
        """Token IDs for the subject of the prompt."""
        return [tokenizer.encode(" " + prompt.subject)[0] for prompt in self.prompts]

    def patch_random(
        self,
        rng: random.Random,
        new_template: IOIPromptTemplate,
        new_template_values: dict[str, list[str]],
        names: list[str],
    ) -> "IOIPromptCollection":
        """Patches the dataset with new prompts generated from the new template.

        You can leave new_template_values and names empty if you don't want to change them.
        The new_template_values are merged into the old template_values, so if you e.g. want to only change
        the '${object}', but not the '${place}', then you can pass in a new_template_values dict that only
        has an 'object' key.

        At the moment, this changes all the names. Maybe a future version can be nicer."""
        new_prompts = [
            prompt.patch(
                new_template,
                _select_random_values(
                    rng, names_order=new_template.names_order, names=names, template_values=new_template_values
                ),
            )
            for prompt in self.prompts
        ]

        return IOIPromptCollection(prompts=new_prompts, num_examples=self.num_examples, prepend_bos=self._prepend_bos)
