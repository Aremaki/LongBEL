import json

from openai import OpenAI


class LLMRelator:
    """
    LLM-based judge classifier
    """

    def __init__(
        self,
        model="gpt-5.2-2025-12-11",
        temperature=0.1,
    ):
        self.client = OpenAI(
            api_key="YOUR_OPENAI_API_KEY"  # Replace with your actual API key or use environment variable
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
# LABEL DEFINITIONS

## EXACT
EXACT applies when the PREDICTED concept (or combination of concepts) is clinically appropriate for what the mention represents in its context. This includes cases where:
- PREDICTED accurately represents the mention's meaning
- Both PREDICTED and GOLD are imperfect matches for the mention, but PREDICTED is not clinically worse than GOLD

## BROAD
BROAD applies when the PREDICTED concept (or combination of concepts) is clinically correct in essence but too general compared to the medical meaning expressed by the mention in context. This includes cases where:
- PREDICTED represents a higher-level category that includes the intended concept
- PREDICTED omits clinically important details such as subtype, location, severity, or etiology that are clearly expressed in the mention or context

## NARROW
NARROW applies when the PREDICTED concept (or combination of concepts) is clinically related to the mention but more specific or restrictive than what the mention supports. This includes cases where:
- PREDICTED adds unjustified specificity (e.g., subtype, severity, laterality, etiology) not stated or implied by the mention or context

## NO_RELATION
NO_RELATION applies when the PREDICTED concept is clinically unrelated to the mention and represents a different medical entity with no meaningful clinical connection. Use this label only when no reasonable clinical relationship exists.

# EXAMPLES FROM CLINICIAN VALIDATION

## EXAMPLE 1 (EXACT)
Context: "Los padres activaban el aparato en casa para lograr la expansión y una vez por semana nosotros [modificábamos la tensión de las cadenetas]{{PROCEDIMIENTO}} para lograr la retrusión de la premaxila."
Mention: "modificábamos la tensión de las cadenetas"
Gold: aplicación de tracción mediante un sistema de tracción (procedimiento)
Predicted: ajuste de dispositivo ortodóncico (procedimiento)
Label: EXACT

## EXAMPLE 2 (NARROW)
Context: "Pautan férula posicional bilateral nocturna más tratamiento de fisioterapia ([tonificación musculatura intrínseca de mano]{{PROCEDIMIENTO}} y electroterapia analgésica)."
Mention: "tonificación musculatura intrínseca de mano"
Gold: movilización de la mano (régimen/tratamiento)
Predicted: ejercicios para los dedos de la mano (régimen/tratamiento)
Label: NARROW

## EXAMPLE 3 (BROAD)
Context: "Se realizó arteriografía selectiva de tronco celíaco y de arteria hepática, visualizándose un [pseudoaneurisma de 1 cm en rama de arteria hepática]{{ENFERMEDAD}} derecha."
Mention: "pseudoaneurisma de 1 cm en rama de arteria hepática"
Gold: seudoaneurisma de arteria hepática (trastorno)
Predicted: seudoaneurisma (trastorno)
Label: BROAD

## EXAMPLE 4 (EXACT)
Context: "Cuatro años más tarde presentó una nueva recidiva, más acusada en OD, con pérdida visual al nivel de 20/60 OD y 20/30 OI, sin evidenciarse [alteraciones campimétricas]{{SINTOMA}}."
Mention: "alteraciones campimétricas"
Gold: visión periférica anormal (hallazgo)
Predicted: defecto del campo visual (hallazgo)
Label: EXACT

## EXAMPLE 5 (EXACT)
Context: "En quirófano, se realiza drenaje de la [colección de material purulento]{{SINTOMA}} y desbridamiento del tejido necrótico..."
Mention: "colección de material purulento"
Gold: absceso (trastorno)
Predicted: secreción purulenta (anomalía morfológica)
Label: EXACT

## EXAMPLE 6 (NO_RELATION)
Context: "Varón de 45 años afecto de [ERC estadio 5 secundaria a poliquistosis hepatorrenal]{{ENFERMEDAD}} con antecedente de hipertensión arterial e hiperuricemia."
Mention: "ERC estadio 5 secundaria a poliquistosis hepatorrenal"
Gold: quiste de riñón, insuficiencia renal crónica, quiste hepático
Predicted: quistes renales múltiples, gota tofácea crónica debida a disfunción renal
Label: NO_RELATION

# QUERY TO EVALUATE

## CONTEXT SENTENCE
{context_sentence}

## GOLD CONCEPTS
Mention in the text: "{mention}"
{gold_str}

## PREDICTED CONCEPTS
{pred_str}

Output a JSON in the format: {{"label": "<LABEL>"}}."""
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
                    "content": """You are a clinical concept evaluation expert. Your task is to classify a PREDICTED concept for a medical mention in context into one of four categories: EXACT, BROAD, NARROW, or NO_RELATION.""",
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
            print(f"Invalid label from LLM: {label}")
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
