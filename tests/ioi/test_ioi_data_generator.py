import random

import pytest

from acdc.ioi.ioi_data_fetchers import IOIExperimentDataGenerator
from acdc.ioi.ioi_dataset import BABA_EARLY_IOS, BABA_LATE_IOS, BABA_TEMPLATES, IOIDataset, generate_prompts_uniform
from acdc.ioi.ioi_dataset_constants import NAMES, OBJECTS, PLACES
from acdc.ioi.ioi_dataset_v2 import IOI_PROMPT_PRETEMPLATES, IOIPromptTemplate, get_ioi_tokenizer


@pytest.mark.slow
def test_experiment_data_generator():
    """This test is very basic: we only test that we can actually run the code."""
    generator = IOIExperimentDataGenerator(
        num_examples=10,
        device="cpu",
        metric_name="kl_div",
    )
    rng = random.Random(42)

    template = IOIPromptTemplate(
        pre_template=IOI_PROMPT_PRETEMPLATES[0],
        names_order=list("ABBA"),
        subject_position=3,
        io_position=4,
    )

    # Just check that we can run these things
    generator.get_all_data_things(rng, template)
    get_ioi_tokenizer()


class TestsFromV1DataGenerators:
    def test_generate_prompt(self):
        generate_prompts_uniform(
            BABA_TEMPLATES,
            names=NAMES,
            nouns_dict={
                "[PLACE]": PLACES,
                "[OBJECT]": OBJECTS,
            },
            N=3,
            symmetric=True,
            seed=0,
            prefixes=None,
        ) == [
            {
                "[PLACE]": "store",
                "[OBJECT]": "computer",
                "text": "The store Bradley and Crystal went to had a computer. Bradley gave it to Crystal",
                "IO": "Crystal",
                "S": "Bradley",
                "TEMPLATE_IDX": 13,
            },
            {
                "text": "The store Crystal and Bradley went to had a computer. Crystal gave it to Bradley",
                "IO": "Bradley",
                "S": "Crystal",
                "TEMPLATE_IDX": 13,
            },
            {
                "[PLACE]": "station",
                "[OBJECT]": "necklace",
                "text": "While Tiffany and Sean were working at the station, Tiffany gave a necklace to Sean",
                "IO": "Sean",
                "S": "Tiffany",
                "TEMPLATE_IDX": 8,
            },
        ]

    def test_ioi_datagen(self):
        assert len(BABA_EARLY_IOS) == len(BABA_LATE_IOS), (len(BABA_EARLY_IOS), len(BABA_LATE_IOS))
        for i in range(len(BABA_EARLY_IOS)):
            d1 = IOIDataset(prompt_type=BABA_EARLY_IOS[i : i + 1], seed=0, N=1)
            d2 = IOIDataset(prompt_type=BABA_LATE_IOS[i : i + 1], seed=0, N=1)
            for tok in ["IO", "S"]:  # occur one earlier and one later
                assert d1.word_idx[tok] + 1 == d2.word_idx[tok], (d1.word_idx[tok], d2.word_idx[tok])
            for tok in ["S2"]:
                assert d1.word_idx[tok] == d2.word_idx[tok], (d1.word_idx[tok], d2.word_idx[tok])
