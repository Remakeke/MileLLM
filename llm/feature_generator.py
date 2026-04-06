from llm.llm_client import LLMClient
from llm.prompt_builder import build_stage2_prompt


class FeatureGenerator:
    def __init__(self, api_key, base_url):
        """
        Initialize LLM-based feature generator.
        """
        self.client = LLMClient(api_key, base_url)

    def generate_features(self, stage1_prompt):
        """
        Generate feature definitions and executable code via LLM.

        Args:
            stage1_prompt (str): Prompt for feature definition generation.

        Returns:
            tuple:
                - feature_defs (str): Generated feature definitions.
                - feature_code (str): Executable feature engineering code.
        """

        # Stage 1: feature definitions
        feature_defs = self.client.call(stage1_prompt, max_tokens=600)

        # Stage 2: code generation
        stage2_prompt = build_stage2_prompt(feature_defs)
        feature_code = self.client.call(stage2_prompt, max_tokens=800)

        return feature_defs, feature_code