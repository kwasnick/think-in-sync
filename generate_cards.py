"""
Usage:
python generate_cards.py --existing cards.json
"""

import argparse
import json
import os
import sys
import logging

from openai import OpenAI
from typing import Annotated
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class Card(BaseModel):
    category: str
    prompts: list[str]


class PotentialPrompt(BaseModel):
    prompt: Annotated[
        str, Field(description="The prompt text, eg 'A popular pizza topping'")
    ]
    possible_responses: Annotated[
        list[str],
        Field(
            description="Possible responses to the prompt, eg 'Pepperoni', 'Mushrooms', 'Sausage', ..."
        ),
    ]
    analysis: Annotated[
        str,
        Field(
            description="An analysis of whether or not the prompt meets our quality guidelines, and how well it meets them. Some prompts are better than others."
        ),
    ]


class PotentialCard(BaseModel):
    category: Annotated[
        str, Field(description="The category of the card, eg 'Cooking'")
    ]
    potential_prompts: Annotated[
        list[PotentialPrompt],
        Field(
            description="Possible prompts being considered for the card", min_length=4
        ),
    ]
    chosen_prompts: Annotated[
        list[str],
        Field(
            description="The perfect, chosen prompts to actually use for the card",
            min_length=4,
            max_length=4,
        ),
    ]

    def to_card(self) -> Card:
        if len(self.chosen_prompts) != 4:
            raise ValueError("Card must have 4 prompts")
        for prompt in self.chosen_prompts:
            if not prompt:
                raise ValueError("Prompt cannot be empty")
        return Card(category=self.category, prompts=self.chosen_prompts)


API_MODEL = "gpt-4o"

GAME_RULES = """
**Game rules**
The game is played by two or more players.
They choose a card and read a prompt.
Then, all players answer the prompt simultaneously.
Their goal is to say the same answer.
Therefore, prompts should not be too obvious and easy to guess.

Example game:
Player 1: "Okay, the category is Sports. First prompt: A sport with a half-time."
Player 1: "3, 2, 1...."
<simultaneously>
Player 2: "Football!"
Player 1: "Soccer!"
Player 1: "Dang, didn't get the same thing. Okay, next prompt: A sport where the ball is hit over a net."
Player 1: "3, 2, 1...."
<simultaneously>
Player 2: "Tennis!"
Player 1: "Tennis!"
Player 2: "Yes! Nailed it!"
"""

QUALITY_RULES = """
**Quality rules (ALL must be met)**
- The prompt should have multiple possible responses.
- This is NOT a trivia game. The prompts are not supposed to have a 'correct' answer, or to be tricky to know an answer for.
- Keep the prompt **neither too broad nor too narrow**:
*Good*: “A musical instrument made of brass” (several obvious answers)
*Bad*: “A vegetable often used for pizza sauce" (obviously only 'tomatoes' is a good answer)
- Do **not** reuse any categories or prompts.
- Prompts should be relatively concise and easy to read. Don't make them too convoluted.
- All cards should have prompts chosen.
- Prompts should be unique; do not have multiple prompts on the same card that are just variations of each other.
"""


def load_existing_cards(path: str) -> list[Card]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return [Card(**card) for card in data]
    except Exception as e:
        logging.error(f"Error reading existing cards from {path}: {e}")
        sys.exit(1)


def save_card(path: str, card: Card) -> None:
    logging.info(f"Saving card '{card.category}'")
    try:
        with open(path, "r+") as f:
            data = json.load(f)
            cards = [Card(**card) for card in data]
            cards.append(card)
            f.seek(0)
            json.dump([c.model_dump() for c in cards], f, indent=2)
            f.truncate()
        logging.info(
            f"Appended card '{card.category}' with prompts {card.prompts} to {path}"
        )
    except Exception as e:
        logging.error(f"Error writing updated cards to {path}: {e}")
        sys.exit(1)


def generate_card(existing: list[Card]) -> PotentialCard:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_message = (
        "You are an expert game designer that generates a cards for a 'partner guessing in sync' game.\n"
        f"{GAME_RULES}\n"
        f"{QUALITY_RULES}"
    )
    user_message = (
        "Do not re-use any of the following categories:\n"
        + "\n".join([card.category for card in existing])
        + "\n\n"
        "Generate a new card."
    )
    logging.info("Requesting new card generation...")
    resp = client.beta.chat.completions.parse(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        response_format=PotentialCard,
    )

    assert (
        resp is not None
        and resp.choices is not None
        and len(resp.choices) > 0
        and resp.choices[0].message is not None
        and resp.choices[0].message.content is not None
    )

    card = PotentialCard.model_validate_json(resp.choices[0].message.content)
    return card


def main(path: str):
    existing = load_existing_cards(path)
    while True:
        try:
            potential_card = generate_card(existing)
            card = potential_card.to_card()
            save_card(path, card)
            existing = load_existing_cards(path)
        except Exception as e:
            logging.error(f"Error generating card: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and evaluate partner-guessing cards."
    )
    parser.add_argument(
        "--existing",
        "-e",
        type=str,
        required=True,
        help="Path to JSON file of existing cards",
    )
    args = parser.parse_args()
    main(args.existing)
