import json

from openai import OpenAI


class LLMRelator:
    """
    LLM-based judge classifier, drop-in replacement for TransformersRelator.
    """

    def __init__(
        self,
        model="gpt-4.1-mini",
        temperature=0.0,
    ):
        self.client = OpenAI(
            api_key="YOUR_OPENAI_API_KEY"  # Replace with your actual API key
        )
        self.model = model
        self.temperature = temperature

        # fixed allowed labels from your base class
        self.allowed_labels = ["NO_RELATION", "EXACT", "BROAD", "NARROW"]

    def _build_prompt(
        self,
        gold_concepts: list[dict],
        pred_concepts: list[dict],
    ) -> str:
        """
        Build the LLM prompt for one source-target pair.
        s and t are objects with `text`, `code`, `title`, `synonyms` and `semantic_group` attributes.
        """

        # Extract context info from the first gold concept
        context_sentence = (
            gold_concepts[0].get("context_sentence", "") if gold_concepts else ""
        )
        mention = gold_concepts[0].get("mention", "") if gold_concepts else ""

        # Helper to format a list of concepts
        def format_concepts(concepts):
            out = ""
            for i, c in enumerate(concepts):
                syns = ", ".join(c.get("synonyms", [])) if c.get("synonyms") else "None"
                out += f"""
Concept {i + 1}:
Code: {c.get("code", "")}
Semantic group: {c.get("semantic_group", "")}
Title: {c.get("title", "")}
Synonyms: {syns}
Representative synonym: "{c.get("text", "")}"
"""
            return out

        gold_str = format_concepts(gold_concepts)
        pred_str = format_concepts(pred_concepts)

        prompt = f"""
You are an expert biomedical ontology curator.
Your task is to determine the semantic relationship between two sets of concepts: GOLD CONCEPTS and PREDICTED CONCEPTS.

# POSSIBLE LABELS
- EXACT: Both sets of concepts refer to the same medical idea.
- BROAD: The PREDICTED concepts are more general than the GOLD concepts.
- NARROW: The PREDICTED concepts are more specific than the GOLD concepts.
- NO_RELATION: The two sets of concepts describe different medical ideas.

# RULES
- Use ONLY the provided titles, codes, synonyms and semantic groups.
- The context sentence is for disambiguation only.
- Output ONLY a JSON dictionary: {{"label": "<LABEL>"}}

# QUERY

### CONTEXT SENTENCE
{context_sentence}

### GOLD CONCEPTS
Mention in the text: "{mention}"
{gold_str}

### PREDICTED CONCEPTS
{pred_str}

Return only: {{"label": "<LABEL>"}}
"""
        return prompt

    def _call_llm(self, prompt):
        """
        Batch call the LLM with multiple prompts.
        prompts: list of strings
        returns: list of labels
        """

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict biomedical concept relation classifier.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content

        try:
            parsed = json.loads(raw)  # type: ignore
            label = parsed.get("label", "NO_RELATION").upper()
        except Exception:
            print(f"Error parsing response: {raw}")
            label = "NO_RELATION"

        if label not in self.allowed_labels:
            label = "NO_RELATION"

        return label

    def compute_relation(
        self,
        gold_concepts: list[dict],
        pred_concepts: list[dict],
    ) -> str:
        """
        Inputs:
           gold_concepts = list of concept objects
           pred_concepts = list of concept objects
        Output:
           str relation of the two sets of concepts: "EXACT", "BROAD", "NARROW", or "NO_RELATION"
        """
        # Check if there is an empty concept
        for concept in pred_concepts:
            if not concept.get("text") or not concept.get("code"):
                return "NO_RELATION"
        prompt = self._build_prompt(gold_concepts, pred_concepts)
        # print(prompt)
        return self._call_llm(prompt)
