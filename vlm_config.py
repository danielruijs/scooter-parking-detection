from vlm_models import (
    InternVLAdapter,
    Qwen2_5VLAdapter,
    StandardVLMAdapter,
    VLMAdapter,
)

SYSTEM_PROMPT = """You are an expert urban mobility parking inspector.
Analyze scooter parking photos according to strict urban parking rules.

Parking Rules:
1. Proper:
   - Must be fully upright and stable on its kickstand.
   - Out of the pedestrian sidewalk/walkway, leaving clear passage.
   - NOT blocking tactile paving (yellow or textured pavers for visually impaired pedestrians).
   - NOT blocking building entrance doors, ADA access ramps, emergency exits, or crosswalks.
2. Improper:
   - Fallen over / knocked down / lying on the pavement.
   - Blocking or encroaching upon the pedestrian sidewalk or cycle lane.
   - Resting on or obstructing tactile paving.
   - Blocking doorways, ramps, or stairs.

Output schema: Respond ONLY with a valid JSON object matching this schema, without code fences or extra text:
{
  "is_proper": true or false,
  "feedback": "Concise description of the parking situation and actionable advice if improper"
}"""

USER_PROMPT = "Evaluate the electric scooter parking in this image. Is it parked properly or improperly? Provide your evaluation as JSON."

AVAILABLE_VLM_MODELS: dict[str, VLMAdapter] = {
    # --- Liquid AI LFM2.5-VL Series (2026) ---
    "lfm2.5-vl-450m": StandardVLMAdapter(
        name="lfm2.5-vl-450m",
        hf_id="LiquidAI/LFM2.5-VL-450M",
        display_name="LFM2.5-VL 450M",
        use_4bit=False,
        style="two_stage",
        image_size=None,
        system_as_list=False,
        repetition_penalty=None,
        release_date="Apr 2026",
    ),
    "lfm2.5-vl-1.6b": StandardVLMAdapter(
        name="lfm2.5-vl-1.6b",
        hf_id="LiquidAI/LFM2.5-VL-1.6B",
        display_name="LFM2.5-VL 1.6B",
        use_4bit=False,
        style="two_stage",
        image_size=None,
        system_as_list=False,
        repetition_penalty=None,
        release_date="Jan 2026",
    ),
    "lfm2.5-vl-3b": StandardVLMAdapter(
        name="lfm2.5-vl-3b",
        hf_id="LiquidAI/LFM2.5-VL-3B",
        display_name="LFM2.5-VL 3B",
        use_4bit=True,
        style="two_stage",
        image_size=None,
        system_as_list=False,
        repetition_penalty=None,
        release_date="Aug 2026",
    ),
    # --- Mistral AI Series (2025) ---
    "ministral-3-3b-instruct-2512": StandardVLMAdapter(
        name="ministral-3-3b-instruct-2512",
        hf_id="mistralai/Ministral-3-3B-Instruct-2512",
        display_name="Ministral 3 3B Instruct 2512",
        use_4bit=False,
        style="two_stage",
        image_size=None,
        system_as_list=False,
        repetition_penalty=None,
        release_date="Dec 2025",
    ),
    # --- Qwen 3.5 & Qwen 3-VL Series (2026) ---
    "qwen3.5-0.8b": StandardVLMAdapter(
        name="qwen3.5-0.8b",
        hf_id="Qwen/Qwen3.5-0.8B",
        display_name="Qwen3.5 0.8B",
        use_4bit=False,
        style="tokenized",
        image_size=(640, 640),
        system_as_list=False,
        repetition_penalty=None,
        release_date="Mar 2026",
    ),
    "qwen3.5-2b": StandardVLMAdapter(
        name="qwen3.5-2b",
        hf_id="Qwen/Qwen3.5-2B",
        display_name="Qwen3.5 2B",
        use_4bit=True,
        style="tokenized",
        image_size=(640, 640),
        system_as_list=False,
        repetition_penalty=None,
        release_date="Mar 2026",
    ),
    "qwen3.5-4b": StandardVLMAdapter(
        name="qwen3.5-4b",
        hf_id="Qwen/Qwen3.5-4B",
        display_name="Qwen3.5 4B",
        use_4bit=True,
        style="tokenized",
        image_size=(640, 640),
        system_as_list=False,
        repetition_penalty=None,
        release_date="Mar 2026",
    ),
    "qwen3-vl-2b-instruct": StandardVLMAdapter(
        name="qwen3-vl-2b-instruct",
        hf_id="Qwen/Qwen3-VL-2B-Instruct",
        display_name="Qwen3-VL 2B Instruct",
        use_4bit=True,
        style="tokenized",
        image_size=(640, 640),
        system_as_list=False,
        repetition_penalty=None,
        release_date="Oct 2025",
    ),
    # --- Google Gemma 4 Series (April 2026) ---
    "gemma-4-e2b-it": StandardVLMAdapter(
        name="gemma-4-e2b-it",
        hf_id="google/gemma-4-E2B-it",
        display_name="Gemma 4 E2B-it",
        use_4bit=True,
        style="tokenized",
        image_size=(448, 448),
        system_as_list=True,
        repetition_penalty=None,
        release_date="Apr 2026",
    ),
    # --- OpenBMB MiniCPM-V Series ---
    "minicpm-v-4_6": StandardVLMAdapter(
        name="minicpm-v-4_6",
        hf_id="openbmb/MiniCPM-V-4_6",
        display_name="MiniCPM-V 4.6",
        use_4bit=False,
        style="two_stage",
        image_size=(448, 448),
        system_as_list=False,
        repetition_penalty=None,
        release_date="May 2026",
    ),
    # --- HuggingFace SmolVLM / SmolVLM2 Series ---
    "smolvlm2-2.2b-instruct": StandardVLMAdapter(
        name="smolvlm2-2.2b-instruct",
        hf_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        display_name="SmolVLM2 2.2B Instruct",
        use_4bit=True,
        style="two_stage",
        image_size=(640, 640),
        system_as_list=True,
        repetition_penalty=1.15,
        release_date="Feb 2025",
    ),
    "smolvlm2-500m-instruct": StandardVLMAdapter(
        name="smolvlm2-500m-instruct",
        hf_id="HuggingFaceTB/SmolVLM2-500M-Instruct",
        display_name="SmolVLM2 500M Instruct",
        use_4bit=False,
        style="two_stage",
        image_size=(640, 640),
        system_as_list=True,
        repetition_penalty=1.15,
        release_date="Feb 2025",
    ),
    # --- Dedicated Architecture Adapters ---
    "qwen2.5-vl-3b-instruct": Qwen2_5VLAdapter(
        name="qwen2.5-vl-3b-instruct",
        hf_id="Qwen/Qwen2.5-VL-3B-Instruct",
        display_name="Qwen2.5-VL 3B Instruct",
        use_4bit=True,
        release_date="Jan 2025",
    ),
    "internvl3_5-1b": InternVLAdapter(
        name="internvl3_5-1b",
        hf_id="OpenGVLab/InternVL3_5-1B",
        display_name="InternVL 3.5 1B",
        release_date="Aug 2026",
    ),
}
