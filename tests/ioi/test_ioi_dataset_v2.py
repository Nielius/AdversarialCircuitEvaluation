import random
from string import Template

from acdc.ioi.ioi_dataset_v2 import (
    IOIPromptCollection,
    IOIPromptTemplate,
    generate_prompts_uniform_v2,
    get_ioi_tokenizer,
)


def test_patch_ioi_rendered_prompt_template():
    prompt_template = IOIPromptTemplate(
        Template("Then, ${name1} and ${name2} went to the ${place}. ${name3} gave a ${object} to ${name4}"),
        ["A", "B", "B", "A"],
        io_position=4,
        subject_position=3,
    )
    rendered_template = prompt_template.render(A="Adam", B="Brian", place="park", object="ball")
    new_rendered_template = rendered_template.patch_with_new_order(["A", "B", "A", "B"], {"object": "toy"})
    corrupted_rendered_template = rendered_template.patch_with_new_order(
        ["A", "B", "C", "A"], {"A": "David", "B": "Eric", "C": "Gregory"}
    )

    assert rendered_template.text == "Then, Adam and Brian went to the park. Brian gave a ball to Adam"
    assert rendered_template.indirect_object == "Adam"

    assert new_rendered_template.text == "Then, Adam and Brian went to the park. Adam gave a toy to Brian"
    assert new_rendered_template.indirect_object == "Brian"

    assert corrupted_rendered_template.text == "Then, David and Eric went to the park. Gregory gave a ball to David"
    assert corrupted_rendered_template.indirect_object == "David"


def test_generate_prompts_uniform_v2():
    prompt_template = IOIPromptTemplate(
        Template("Then, ${name1} and ${name2} went to the ${place}. ${name3} gave a ${object} to ${name4}"),
        ["A", "B", "B", "A"],
        io_position=4,
        subject_position=3,
    )

    prompts = generate_prompts_uniform_v2(
        template=prompt_template,
        template_values={"object": ["book", "ball", "toy"], "place": ["store", "garden"]},
        names=["Michael", "Christopher", "Jessica", "Matthew"],
        num_prompts=10,
        rng=random.Random(93458),
    )

    assert [prompt.text for prompt in prompts] == [
        "Then, Matthew and Christopher went to the store. Christopher gave a ball to Matthew",
        "Then, Matthew and Jessica went to the store. Jessica gave a toy to Matthew",
        "Then, Christopher and Matthew went to the garden. Matthew gave a book to Christopher",
        "Then, Jessica and Christopher went to the garden. Christopher gave a ball to Jessica",
        "Then, Matthew and Michael went to the garden. Michael gave a ball to Matthew",
        "Then, Matthew and Michael went to the store. Michael gave a toy to Matthew",
        "Then, Jessica and Matthew went to the store. Matthew gave a toy to Jessica",
        "Then, Matthew and Christopher went to the store. Christopher gave a toy to Matthew",
        "Then, Christopher and Jessica went to the garden. Jessica gave a ball to Christopher",
        "Then, Christopher and Matthew went to the garden. Matthew gave a toy to Christopher",
    ]


def test_dataset_v2_subject_and_io_tokens():
    rng = random.Random("923847")
    num_examples = 10

    dataset = IOIPromptCollection(
        prompts=generate_prompts_uniform_v2(
            template=IOIPromptTemplate(
                pre_template=Template(
                    "Then, ${name1} and ${name2} went to the ${place}. ${name3} gave a ${object} to ${name4}"
                ),
                names_order=["A", "B", "A", "B"],
                io_position=4,
                subject_position=3,
            ),
            template_values={"object": ["book", "ball", "toy"], "place": ["store", "garden"]},
            names=["Michael", "Christopher", "Jessica", "Matthew"],
            num_prompts=num_examples,
            rng=rng,
        ),
        num_examples=num_examples,
    )

    tokenizer = get_ioi_tokenizer()

    # This is an assumption in the original get_all_ioi_things: there is never any padding and the last token is always the indirect object
    assert dataset.io_tokenIDs(tokenizer) == dataset.tokens(tokenizer, device="cpu")[:, -1].tolist()
    # This particular prompt has the subject as the third token: ["Then", ",", " Michael", ...]
    assert dataset.s_tokenIDs(tokenizer) == dataset.tokens(tokenizer, device="cpu")[:, 2].tolist()


def test_dataset_v2_generate_patch_data():
    rng = random.Random("923847")
    num_examples = 10

    names_original = ["Michael", "Christopher", "Jessica", "Matthew"]
    names_new = ["Ashley", "Jennifer", "Daniel", "Robert"]

    ioi_prompt_template = IOIPromptTemplate(
        pre_template=Template(
            "Then, ${name1} and ${name2} went to the ${place}. ${name3} gave a ${object} to ${name4}"
        ),
        names_order=["A", "B", "A", "B"],
        io_position=4,
        subject_position=3,
    )
    dataset = IOIPromptCollection(
        prompts=generate_prompts_uniform_v2(
            template=ioi_prompt_template,
            template_values={"object": ["book", "ball", "toy"], "place": ["store", "garden"]},
            names=names_original,
            num_prompts=num_examples,
            rng=rng,
        ),
        num_examples=num_examples,
    )

    new_dataset = dataset.patch_random(
        rng,
        new_template=ioi_prompt_template.with_different_order(list("ABCA")),
        names=names_new,
        new_template_values={},  # don't change any of the objects
    )

    assert [prompt.text for prompt in new_dataset.prompts] == [
        "Then, Jennifer and Robert went to the garden. Ashley gave a book to Jennifer",
        "Then, Jennifer and Daniel went to the garden. Robert gave a book to Jennifer",
        "Then, Daniel and Robert went to the garden. Jennifer gave a toy to Daniel",
        "Then, Robert and Daniel went to the store. Ashley gave a book to Robert",
        "Then, Ashley and Jennifer went to the garden. Daniel gave a toy to Ashley",
        "Then, Jennifer and Ashley went to the store. Daniel gave a ball to Jennifer",
        "Then, Robert and Jennifer went to the store. Ashley gave a toy to Robert",
        "Then, Ashley and Robert went to the garden. Jennifer gave a ball to Ashley",
        "Then, Jennifer and Ashley went to the garden. Daniel gave a ball to Jennifer",
        "Then, Ashley and Daniel went to the store. Jennifer gave a book to Ashley",
    ]

    # Now also assert that the places and objects are the same
    # ----

    tokenizer = get_ioi_tokenizer()
    original_tokens = dataset.tokens(tokenizer, "cpu")
    new_tokens = new_dataset.tokens(tokenizer, "cpu")

    place_token_position = 8
    object_token_position = 13
    # For debugging: print([(i, token, tokenizer.decode(token)) for i, token in enumerate(new_tokens[0, :].tolist())])
    assert original_tokens[0, place_token_position] == tokenizer.encode(" garden")[0]
    assert original_tokens[0, object_token_position] == tokenizer.encode(" book")[0]

    assert (original_tokens[:, place_token_position] == new_tokens[:, place_token_position]).all()
    assert (original_tokens[:, object_token_position] == new_tokens[:, object_token_position]).all()
