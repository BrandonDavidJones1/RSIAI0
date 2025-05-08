# --- START OF FILE seed/llm_service.py ---

# RSIAI/seed/llm_service.py
"""
Defines the Seed_LLMService class for interacting with an LLM API (e.g., OpenAI)
or simulating interaction via manual console input.
Includes robust error handling, retries, and configurable parameters.
Uses a mutable operational prompt template combined with an immutable core directive.
"""
import os
import json
import time
import traceback
import logging
import re # For finding JSON in responses
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
    # Define dummy decorators/functions if tenacity is not installed
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def stop_after_attempt(*args, **kwargs): return None
    def wait_exponential(*args, **kwargs): return None
    def retry_if_exception_type(*args, **kwargs): return None
    logging.getLogger(__name__).warning("Libraries 'openai' or 'tenacity' not found. LLMService fallback/manual mode only.")

# Import necessary config constants using relative import within the package
from .config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME, LLM_TIMEOUT_SEC,
    LLM_MAX_RETRIES, LLM_DEFAULT_MAX_TOKENS,
    ALIGNMENT_DIRECTIVE, # <<< Import the immutable directive
    LLM_OPERATIONAL_PROMPT_TEMPLATE as DEFAULT_OPERATIONAL_PROMPT, # <<< Import the default template
    LLM_MANUAL_MODE,
)
# Import MemorySystem type hint safely
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .memory_system import MemorySystem

logger = logging.getLogger(__name__)

# Renamed class
class Seed_LLMService:
    """
    Provides an interface to interact with an LLM API or use manual input,
    with retries and fallbacks. Combines an operational prompt template with
    the core alignment directive from config.
    """
    def __init__(self,
                 api_key: str = LLM_API_KEY,
                 base_url: Optional[str] = LLM_BASE_URL,
                 model: str = LLM_MODEL_NAME,
                 memory_system: Optional['MemorySystem'] = None): # <<< Add memory dependency
        """
        Initializes the LLM Service.

        Args:
            api_key (str): API key for the LLM service.
            base_url (Optional[str]): Base URL for the LLM API (e.g., for local models).
            model (str): The identifier of the LLM model to use.
            memory_system (Optional[MemorySystem]): Reference to the memory system
                                                     to fetch the current operational prompt.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client: Optional[OpenAI] = None
        self.manual_mode_enabled: bool = LLM_MANUAL_MODE
        self.memory = memory_system # <<< Store memory reference

        # The default_system_prompt is no longer used directly for queries,
        # as the prompt is dynamically constructed from the template and directive.
        self.default_system_prompt: str = f"""[This is a fallback prompt only used if template loading fails entirely] You are RSIAI-Seed-v0.1, a strategic reasoning core. Your primary goal is to analyze context and select the single best action from AVAILABLE_ACTIONS to achieve the CURRENT_GOAL, adhering to CONSTRAINTS and the core directive: '{ALIGNMENT_DIRECTIVE}'. Respond ONLY with a single, valid JSON object representing the chosen action. Ensure JSON is well-formed and contains necessary fields like 'action_type' and 'reasoning'. Do NOT include explanations, apologies, or any text outside the JSON object."""

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

                # Handle case where api_key might still be the placeholder if base_url is set
                if "api_key" in client_args and client_args["api_key"] == "YOUR_API_KEY_OR_USE_LOCAL":
                    # Don't pass the placeholder if using a base_url that might not need it
                    if self.base_url:
                        logger.info("Ignoring placeholder API key when using BaseURL.")
                        del client_args["api_key"]
                    else:
                        # Keep the warning if using default OpenAI without a real key
                         logger.warning("API Key is placeholder 'YOUR_API_KEY_OR_USE_LOCAL'. OpenAI client init likely to fail.")

                self.client = OpenAI(**client_args)
                # Optional: Add a ping or simple API call here to confirm connectivity
                # self.client.models.list() # Example check
                logger.info("Seed LLMService Client Initialized (API Mode).")
            except Exception as e:
                logger.error(f"Seed LLMService Client Init Error (API Mode): {e}", exc_info=True)
                self.client = None
                logger.warning("Falling back to Manual Mode due to client initialization error.")
                self.manual_mode_enabled = True
        elif self.manual_mode_enabled:
             logger.info("Seed LLMService Initialized in Manual Mode.")
        else: # Case where OPENAI_IMPORTED is False and Manual Mode is False
             logger.error("Seed LLMService cannot initialize API client (libs missing) and Manual Mode is disabled. Query will always fallback.")


    # Apply retry decorator only if tenacity was imported
    @retry(
        stop=stop_after_attempt(LLM_MAX_RETRIES + 1),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError, APIError)),
        before_sleep=lambda rs: logger.warning(f"LLM API Error: {rs.outcome.exception()}. Retrying attempt {rs.attempt_number} in {rs.next_action.sleep:.1f}s...")
    ) if OPENAI_IMPORTED else lambda func: func # No-op decorator if tenacity not available
    def _make_llm_api_call(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        """ Internal method performing the actual API call, wrapped by tenacity. """
        if not self.client: raise ConnectionError("LLM Client not available for API call.")
        start_time = time.time()
        logger.debug(f"Sending request to LLM (model: {self.model}, max_tokens: {max_tokens})...")

        # Prepare request arguments
        request_args = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # "response_format": { "type": "json_object" }, # Might require specific OpenAI model/version support
        }

        completion = self.client.chat.completions.create(**request_args) # type: ignore

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
        Constructs the system prompt dynamically from the operational template
        and the core alignment directive. Validates JSON format for both manual
        and API responses.
        """

        final_system_prompt = system_prompt_override # Use override if provided directly
        if not final_system_prompt:
            try:
                # Fetch current operational template and inject the immutable directive
                current_operational_template = self._get_current_operational_template()
                final_system_prompt = current_operational_template.format(
                    alignment_directive=ALIGNMENT_DIRECTIVE
                )
            except KeyError as ke:
                logger.error(f"CRITICAL: Operational prompt template missing '{{alignment_directive}}' placeholder! Error: {ke}. Using default fallback prompt.")
                final_system_prompt = self.default_system_prompt # Use basic default on critical formatting error
            except Exception as e:
                 logger.error(f"Error constructing final system prompt: {e}. Using default fallback prompt.", exc_info=True)
                 final_system_prompt = self.default_system_prompt

        if self.manual_mode_enabled or self.client is None:
            if self.client is None and not self.manual_mode_enabled:
                logger.warning("LLM Client not initialized and Manual Mode disabled. Using fallback.")
                return self._get_fallback_response("Client not initialized")

            logger.info("--- LLM Query (Manual Mode) ---")
            print("\n" + "="*20 + " SEED LLM PROMPT (Manual Input Required) " + "="*20)
            print(f"[SYSTEM]\n{final_system_prompt}\n") # <<< Show the combined prompt
            print(f"[USER]\n{prompt}\n")
            print("="*70)
            while True:
                try:
                    manual_json_input = input("Enter the Seed Action JSON: ")
                    manual_json_input = manual_json_input.strip()
                    # Basic check for JSON structure before parsing
                    if manual_json_input.startswith("{") and manual_json_input.endswith("}"):
                        json.loads(manual_json_input) # Validate JSON format fully
                        logger.info(f"Manual input received: {manual_json_input}")
                        return manual_json_input
                    else:
                        print("Invalid input: Does not look like a JSON object (must start with { and end with }). Please try again.")
                except json.JSONDecodeError as json_err:
                    print(f"Invalid JSON format: {json_err}. Please try again.")
                except EOFError:
                    logger.warning("EOF received during manual input. Using fallback.")
                    return self._get_fallback_response("EOF during manual input")
                except Exception as input_err:
                    logger.error(f"Error during manual input: {input_err}", exc_info=True)
                    print("Unexpected error during input.")
                    return self._get_fallback_response(f"Manual input error: {input_err}")
        else: # API Mode
            messages = [{"role": "system", "content": final_system_prompt}, {"role": "user", "content": prompt}] # <<< Use combined prompt
            try:
                response = self._make_llm_api_call(messages, max_tokens, temperature)
                logger.info("LLM API Query successful, validating response content...")
                try:
                    # Attempt to find JSON object within potential extra text
                    match = re.search(r'\{.*\}', response, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        parsed_json = json.loads(json_str)
                        if not isinstance(parsed_json, dict):
                            raise TypeError("Response contains valid JSON but it's not a dictionary object.")
                        logger.debug("LLM API Response content is valid JSON object.")
                        return json_str # Return only the JSON part
                    else:
                        logger.error("LLM API response did not contain a recognizable JSON object.")
                        logger.debug(f"Non-JSON content received: {response[:500]}...")
                        return self._get_fallback_response("LLM response missing JSON object.")

                except (json.JSONDecodeError, TypeError) as json_err:
                    logger.error(f"LLM API returned invalid/non-object JSON content: {json_err}")
                    logger.debug(f"Invalid content received: {response[:500]}...")
                    return self._get_fallback_response(f"LLM returned invalid JSON: {json_err}")
            except Exception as e:
                # This catches errors from _make_llm_api_call if retries fail
                logger.critical(f"LLM API Query failed definitively after retries: {e}", exc_info=True)
                return self._get_fallback_response(f"API Query failed: {e}")

    def _get_current_operational_template(self) -> str:
        """
        Fetches the current operational prompt template from memory,
        falling back to the default defined in config.py.
        """
        template_param_key = "operational_prompt_template.value"
        if self.memory:
            try:
                # Use the standard method for getting learning parameters
                template = self.memory.get_learning_parameter(template_param_key)
                if template and isinstance(template, str):
                    # Basic check for placeholder presence
                    if '{alignment_directive}' in template:
                        logger.debug("Using operational prompt template from memory.")
                        return template
                    else:
                        logger.error(f"Operational prompt template from memory is missing the required '{{alignment_directive}}' placeholder! Falling back to default.")
                else:
                    logger.debug(f"No valid operational prompt template found in memory for key '{template_param_key}'. Using default.")
            except Exception as e:
                logger.warning(f"Failed to get operational prompt from memory: {e}. Using default.", exc_info=True)
        else:
            logger.debug("MemorySystem not available to LLMService. Using default operational prompt.")

        logger.debug("Using default operational prompt template from config.")
        # Ensure the default template itself is valid
        if '{alignment_directive}' not in DEFAULT_OPERATIONAL_PROMPT:
            logger.critical("CRITICAL CONFIG ERROR: Default operational prompt template is missing the '{alignment_directive}' placeholder!")
            # Return a minimal safe default if the main default is broken
            return "System Error: Prompt template misconfigured. Please respond with NO_OP. {alignment_directive}"
        return DEFAULT_OPERATIONAL_PROMPT # Fallback to the one defined in config.py


    def _get_fallback_response(self, reason: str ="LLM query failed") -> str:
        """ Generates a predefined fallback action (ANALYZE_MEMORY). """
        fallback_action = {
            "action_type": "ANALYZE_MEMORY", # Fallback remains ANALYZE_MEMORY (can be internal or LLM-guided)
            "query": f"Analyze state and recent failures given fallback reason: {reason}",
            "reasoning": f"Fallback Action: LLM service unavailable or failed ({reason}). Analyzing memory to understand context."
        }
        logger.warning(f"Seed LLM Service Fallback Triggered: {reason}")
        try:
            return json.dumps(fallback_action)
        except Exception as json_err:
             logger.critical(f"Seed LLM Service CRITICAL Error serializing fallback JSON: {json_err}")
             # Ultimate fallback if JSON serialization itself fails
             return '{"action_type": "NO_OP", "reasoning": "Fallback: LLM query failed and fallback JSON serialization error."}'

# --- END OF FILE seed/llm_service.py ---