# --- START OF FILE seed/llm_service.py ---

# RSIAI/seed/llm_service.py
"""
Defines the Seed_LLMService class for interacting with an LLM API (e.g., OpenAI)
or simulating interaction via manual console input.
Includes robust error handling, retries, and configurable parameters.
"""
import os
import json
import time
import traceback
import logging
from typing import Optional, List, Dict, Any

# Use tenacity for retries
try:
    from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    OPENAI_IMPORTED = True
except ImportError:
    OPENAI_IMPORTED = False
    class OpenAI: pass # type: ignore
    class APIError(Exception): pass # type: ignore
    class RateLimitError(APIError): pass # type: ignore
    class APITimeoutError(APIError): pass # type: ignore
    class APIConnectionError(APIError): pass # type: ignore
    def retry(*args, **kwargs): return lambda f: f # type: ignore
    def stop_after_attempt(*args, **kwargs): return None # type: ignore
    def wait_exponential(*args, **kwargs): return None # type: ignore
    def retry_if_exception_type(*args, **kwargs): return None # type: ignore
    logging.getLogger(__name__).warning("Libraries 'openai' or 'tenacity' not found. LLMService fallback/manual mode only.")

# Import necessary config constants
from ..config import ( # Adjusted relative import
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME, LLM_TIMEOUT_SEC,
    LLM_MAX_RETRIES, LLM_DEFAULT_MAX_TOKENS,
    ALIGNMENT_DIRECTIVE,
    LLM_MANUAL_MODE,
    ALIGNMENT_PROMPT # Import the main prompt template
)

logger = logging.getLogger(__name__)

# Renamed class
class Seed_LLMService:
    """
    Provides an interface to interact with an LLM API or use manual input,
    with retries and fallbacks. Uses alignment directive from config.
    """
    def __init__(self, api_key: str = LLM_API_KEY, base_url: Optional[str] = LLM_BASE_URL, model: str = LLM_MODEL_NAME):
        """
        Initializes the LLM Service.

        Args:
            api_key (str): API key for the LLM service.
            base_url (Optional[str]): Base URL for the LLM API (e.g., for local models).
            model (str): The identifier of the LLM model to use.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client: Optional[OpenAI] = None
        self.manual_mode_enabled: bool = LLM_MANUAL_MODE

        # Define the default system prompt (though ALIGNMENT_PROMPT from config is usually used)
        self.default_system_prompt: str = f"""You are RSIAI-Seed-v0.1, a strategic reasoning core. Your primary goal is to analyze context and select the single best action from AVAILABLE_ACTIONS to achieve the CURRENT_GOAL, adhering to CONSTRAINTS and the core directive: '{ALIGNMENT_DIRECTIVE}'. Respond ONLY with a single, valid JSON object representing the chosen action. Ensure JSON is well-formed and contains necessary fields like 'action_type' and 'reasoning'. Do NOT include explanations, apologies, or any text outside the JSON object."""

        if not self.manual_mode_enabled and OPENAI_IMPORTED:
            try:
                logger.info(f"Initializing LLM Client: Model='{self.model}', BaseURL='{self.base_url or 'Default OpenAI'}'")
                client_args: Dict[str, Any] = {"timeout": LLM_TIMEOUT_SEC + 5}
                if self.base_url:
                    client_args["base_url"] = self.base_url
                    if not self.api_key or self.api_key == "YOUR_API_KEY_OR_USE_LOCAL": logger.info("Using BaseURL without explicitly configured API Key.")
                    else: client_args["api_key"] = self.api_key
                elif not self.api_key or self.api_key == "YOUR_API_KEY_OR_USE_LOCAL": logger.warning("No API key provided and no BaseURL set. OpenAI client init likely to fail if using default OpenAI endpoint.")
                else: client_args["api_key"] = self.api_key
                self.client = OpenAI(**client_args)
                logger.info("Seed LLMService Client Initialized (API Mode).") # Renamed log
            except Exception as e:
                logger.error(f"Seed LLMService Client Init Error (API Mode): {e}", exc_info=True) # Renamed log
                self.client = None
                logger.warning("Falling back to Manual Mode due to client initialization error.")
                self.manual_mode_enabled = True
        elif self.manual_mode_enabled:
             logger.info("Seed LLMService Initialized in Manual Mode.") # Renamed log
        else:
             logger.error("Seed LLMService cannot initialize API client (libs missing) and Manual Mode is disabled. Query will always fallback.") # Renamed log

    @retry(
        stop=stop_after_attempt(LLM_MAX_RETRIES + 1),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError, APIError)),
        before_sleep=lambda rs: logger.warning(f"LLM API Error: {rs.outcome.exception()}. Retrying attempt {rs.attempt_number} in {rs.next_action.sleep:.1f}s...")
    )
    def _make_llm_api_call(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        """ Internal method performing the actual API call, wrapped by tenacity. """
        if not self.client: raise ConnectionError("LLM Client not available for API call.")
        start_time = time.time()
        logger.debug(f"Sending request to LLM (model: {self.model}, max_tokens: {max_tokens})...")
        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature,
            # response_format={ "type": "json_object" }, # Consider enabling if API supports it
        )
        duration = time.time() - start_time; logger.debug(f"LLM API call successful ({duration:.2f}s).")
        if not completion.choices: logger.error("LLM response contained no choices."); raise APIError("No choices returned.")
        choice = completion.choices[0]
        if not choice.message: logger.error("LLM choice contained no message attribute."); raise APIError("Response choice missing message.")
        response_content = choice.message.content; finish_reason = choice.finish_reason
        if finish_reason == 'length': logger.warning(f"LLM response truncated due to max_tokens ({max_tokens}).")
        elif finish_reason != 'stop': logger.warning(f"LLM generation finished unexpectedly: {finish_reason}")
        if response_content: return response_content.strip()
        else: logger.warning("LLM response content is empty."); raise APIError("Received empty response content.")

    def query(self, prompt: str, system_prompt_override: Optional[str] = None, max_tokens: int = LLM_DEFAULT_MAX_TOKENS, temperature: float = 0.5) -> str:
        """
        Sends a prompt to the configured LLM or prompts user for manual input.
        Validates JSON format for both manual and API responses.
        """
        # Use the specific ALIGNMENT_PROMPT from config unless overridden
        system_prompt = system_prompt_override if system_prompt_override else ALIGNMENT_PROMPT

        if self.manual_mode_enabled or self.client is None:
            if self.client is None and not self.manual_mode_enabled:
                logger.warning("LLM Client not initialized and Manual Mode disabled. Using fallback.")
                return self._get_fallback_response("Client not initialized")

            logger.info("--- LLM Query (Manual Mode) ---")
            print("\n" + "="*20 + " SEED LLM PROMPT (Manual Input Required) " + "="*20) # Renamed title
            print(f"[SYSTEM]\n{system_prompt}\n")
            print(f"[USER]\n{prompt}\n")
            print("="*70)
            while True:
                try:
                    manual_json_input = input("Enter the Seed Action JSON: ") # Renamed prompt
                    manual_json_input = manual_json_input.strip()
                    if manual_json_input.startswith("{") and manual_json_input.endswith("}"):
                        json.loads(manual_json_input) # Validate JSON format
                        logger.info(f"Manual input received: {manual_json_input}")
                        return manual_json_input
                    else: print("Invalid input. Please enter a single, valid JSON object.")
                except json.JSONDecodeError as json_err: print(f"Invalid JSON format: {json_err}. Please try again.")
                except EOFError: logger.warning("EOF received during manual input. Using fallback."); return self._get_fallback_response("EOF during manual input")
                except Exception as input_err: logger.error(f"Error during manual input: {input_err}", exc_info=True); print("Unexpected error during input."); return self._get_fallback_response(f"Manual input error: {input_err}")
        else: # API Mode
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            try:
                response = self._make_llm_api_call(messages, max_tokens, temperature)
                logger.info("LLM API Query successful, validating response content...")
                try:
                    parsed_json = json.loads(response)
                    if not isinstance(parsed_json, dict): raise TypeError("Response is valid JSON but not a dictionary object.")
                    logger.debug("LLM API Response content is valid JSON.")
                    return response
                except (json.JSONDecodeError, TypeError) as json_err:
                    logger.error(f"LLM API returned invalid/non-object JSON content: {json_err}")
                    logger.debug(f"Invalid content received: {response[:500]}...")
                    return self._get_fallback_response(f"LLM returned invalid JSON: {json_err}")
            except Exception as e:
                logger.critical(f"LLM API Query failed definitively: {e}", exc_info=True)
                return self._get_fallback_response(f"API Query failed: {e}")

    def _get_fallback_response(self, reason: str ="LLM query failed") -> str:
        """ Generates a predefined fallback action (ANALYZE_MEMORY). """
        fallback_action = {
            "action_type": "ANALYZE_MEMORY", # Fallback remains ANALYZE_MEMORY
            "query": f"Analyze state and recent failures given fallback reason: {reason}",
            "reasoning": f"Fallback Action: LLM service unavailable or failed ({reason}). Analyzing memory to understand context."
        }
        logger.warning(f"Seed LLM Service Fallback Triggered: {reason}") # Renamed log
        try:
            return json.dumps(fallback_action)
        except Exception as json_err:
             logger.critical(f"Seed LLM Service CRITICAL Error serializing fallback JSON: {json_err}") # Renamed log
             # Ultimate fallback if JSON serialization itself fails
             return '{"action_type": "NO_OP", "reasoning": "Fallback: LLM query failed and fallback JSON serialization error."}' # Fallback to NO_OP

# --- END OF FILE seed/llm_service.py ---