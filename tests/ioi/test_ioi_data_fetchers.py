import random

from acdc.ioi.ioi_data_fetchers import IOIExperimentInputData
from acdc.ioi.ioi_dataset_v2 import IOI_PROMPT_PRETEMPLATES, IOIPromptTemplate, get_ioi_tokenizer


def test_generate_ioi_experiment_input_data():
    template = IOIPromptTemplate(
        pre_template=IOI_PROMPT_PRETEMPLATES[0],
        names_order=list("ABBA"),
        io_position=4,
        subject_position=3,
    )
    experiment_input_data = IOIExperimentInputData.generate(
        rng=random.Random(293487),
        num_examples=10,
        template=template,
        patch_template=template.with_different_order(list("ABCA")),
        template_values={"object": ["book", "ball", "toy"], "place": ["store", "garden"]},
        names=["Michael", "Christopher", "Jessica", "Matthew"],
        device="cpu",
    )

    tokenizer = get_ioi_tokenizer()

    assert [tokenizer.decode(t.tolist()) for t in experiment_input_data.default_data] == [
        "Then, Michael and Christopher had a lot of fun at the store. Christopher gave a toy to",
        "Then, Michael and Jessica had a lot of fun at the garden. Jessica gave a book to",
        "Then, Michael and Christopher had a lot of fun at the store. Christopher gave a book to",
        "Then, Jessica and Matthew had a lot of fun at the store. Matthew gave a book to",
        "Then, Christopher and Matthew had a lot of fun at the store. Matthew gave a ball to",
        "Then, Matthew and Christopher had a lot of fun at the garden. Christopher gave a toy to",
        "Then, Christopher and Matthew had a lot of fun at the store. Matthew gave a book to",
        "Then, Matthew and Christopher had a lot of fun at the garden. Christopher gave a book to",
        "Then, Michael and Jessica had a lot of fun at the store. Jessica gave a book to",
        "Then, Matthew and Michael had a lot of fun at the garden. Michael gave a toy to",
        "Then, Michael and Jessica had a lot of fun at the store. Jessica gave a book to",
        "Then, Jessica and Christopher had a lot of fun at the store. Christopher gave a ball to",
        "Then, Matthew and Michael had a lot of fun at the store. Michael gave a book to",
        "Then, Jessica and Matthew had a lot of fun at the garden. Matthew gave a book to",
        "Then, Christopher and Jessica had a lot of fun at the garden. Jessica gave a book to",
        "Then, Matthew and Jessica had a lot of fun at the store. Jessica gave a ball to",
        "Then, Christopher and Michael had a lot of fun at the garden. Michael gave a book to",
        "Then, Christopher and Matthew had a lot of fun at the store. Matthew gave a toy to",
        "Then, Christopher and Matthew had a lot of fun at the garden. Matthew gave a ball to",
        "Then, Jessica and Michael had a lot of fun at the store. Michael gave a book to",
    ]
    assert [tokenizer.decode(t) for t in experiment_input_data.labels] == [
        " Michael",
        " Michael",
        " Michael",
        " Jessica",
        " Christopher",
        " Matthew",
        " Christopher",
        " Matthew",
        " Michael",
        " Matthew",
        " Michael",
        " Jessica",
        " Matthew",
        " Jessica",
        " Christopher",
        " Matthew",
        " Christopher",
        " Christopher",
        " Christopher",
        " Jessica",
    ]
    assert [tokenizer.decode(t.tolist()) for t in experiment_input_data.patch_data] == [
        "Then, Christopher and Jessica had a lot of fun at the store. Michael gave a toy to",
        "Then, Michael and Matthew had a lot of fun at the garden. Christopher gave a book to",
        "Then, Matthew and Michael had a lot of fun at the store. Christopher gave a book to",
        "Then, Jessica and Michael had a lot of fun at the store. Christopher gave a book to",
        "Then, Matthew and Michael had a lot of fun at the store. Jessica gave a ball to",
        "Then, Michael and Jessica had a lot of fun at the garden. Christopher gave a toy to",
        "Then, Christopher and Matthew had a lot of fun at the store. Jessica gave a book to",
        "Then, Jessica and Christopher had a lot of fun at the garden. Michael gave a book to",
        "Then, Christopher and Jessica had a lot of fun at the store. Michael gave a book to",
        "Then, Jessica and Michael had a lot of fun at the garden. Christopher gave a toy to",
        "Then, Christopher and Michael had a lot of fun at the store. Matthew gave a book to",
        "Then, Jessica and Michael had a lot of fun at the store. Matthew gave a ball to",
        "Then, Christopher and Matthew had a lot of fun at the store. Michael gave a book to",
        "Then, Matthew and Michael had a lot of fun at the garden. Christopher gave a book to",
        "Then, Christopher and Michael had a lot of fun at the garden. Matthew gave a book to",
        "Then, Matthew and Michael had a lot of fun at the store. Jessica gave a ball to",
        "Then, Matthew and Jessica had a lot of fun at the garden. Michael gave a book to",
        "Then, Jessica and Christopher had a lot of fun at the store. Matthew gave a toy to",
        "Then, Christopher and Jessica had a lot of fun at the garden. Matthew gave a ball to",
        "Then, Michael and Jessica had a lot of fun at the store. Matthew gave a book to",
    ]
