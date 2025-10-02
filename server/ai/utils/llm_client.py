import asyncio
import boto3
import json
from botocore.config import Config
from core.config import settings

class LLMClient:
    """
    This is a placeholder for the actual LLMClient that interacts with AWS Bedrock.
    It allows the server to start and provides a basic Bedrock invocation structure.
    """
    def __init__(self):
        # Configure with timeout settings
        config = Config(
            read_timeout=60,
            connect_timeout=10,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=settings.AWS_REGION, config=config)

    async def generate(self, prompt: str, response_format: dict = None, temperature: float = 0.3, max_tokens: int = 2000) -> dict:
        print(f"[LLMClient] Generating with max_tokens={max_tokens}, temperature={temperature}")
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0" # Example model ID, can be configured

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,  # Use parameter instead of hardcoded value
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        })

        try:
            print("[LLMClient] Invoking Bedrock API...")
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id, body=body, contentType="application/json", accept="application/json"
            )
            response_body = json.loads(response['body'].read().decode('utf-8'))
            print("[LLMClient] ✓ Bedrock API response received")

            # Extract text content from the response
            if 'content' in response_body and len(response_body['content']) > 0 and 'text' in response_body['content'][0]:
                llm_response_text = response_body['content'][0]['text'].strip()
            else:
                llm_response_text = "Placeholder: No text content found in LLM response."

            # If a JSON schema is expected, try to return a dummy JSON
            if response_format and response_format.get("type") == "json_schema":
                # This is a very basic attempt to return valid JSON for the schema
                # In a real scenario, you'd parse the llm_response_text into the schema
                # For now, we return a dummy structure that matches what answer_scorer expects
                return {
                    "scores": {"placeholder_score": 70},
                    "reasoning": f"Placeholder Bedrock response for: {llm_response_text[:50]}...",
                    "matched_keywords": [],
                    "missing_keywords": [],
                    "strengths": ["Placeholder strength from Bedrock"],
                    "weaknesses": ["Placeholder weakness from Bedrock"]
                }
            
            return {"text": llm_response_text}

        except Exception as e:
            print(f"✗ ERROR in LLMClient Bedrock invocation: {e}")
            print(f"   Error type: {type(e).__name__}")
            # Return a dummy error response
            if response_format and response_format.get("type") == "json_schema":
                return {
                    "scores": {"error_score": 0},
                    "reasoning": f"Error in Bedrock call: {str(e)}",
                    "matched_keywords": [],
                    "missing_keywords": [],
                    "strengths": [],
                    "weaknesses": []
                }
            return {"text": f"Error in LLM generation: {str(e)}"}

    async def chat_completion(self, messages: list, temperature: float = 0.3, max_tokens: int = 1000) -> str:
        """
        Chat completion API compatible method for Bedrock Claude

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text response
        """
        print(f"[LLMClient] Chat completion with max_tokens={max_tokens}")
        try:
            # Convert messages to single prompt (Claude Bedrock format)
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"Human: {msg['content']}\n\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n\n"
                elif msg["role"] == "system":
                    prompt = f"System: {msg['content']}\n\n" + prompt

            prompt += "Assistant:"

            model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            })

            print("[LLMClient] Invoking Bedrock API (chat_completion)...")
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read().decode('utf-8'))
            print("[LLMClient] ✓ Chat completion response received")

            # Extract text content from the response
            if 'content' in response_body and len(response_body['content']) > 0 and 'text' in response_body['content'][0]:
                return response_body['content'][0]['text'].strip()
            else:
                return "No response generated"

        except Exception as e:
            print(f"✗ ERROR in chat_completion Bedrock call: {e}")
            print(f"   Error type: {type(e).__name__}")
            return f"Error: {str(e)}"

    async def ainvoke(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """
        Async invocation for simple prompt-response interaction

        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated text response
        """
        print(f"[LLMClient] ainvoke with max_tokens={max_tokens}")
        try:
            model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            })

            print("[LLMClient] Invoking Bedrock API (ainvoke)...")
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read().decode('utf-8'))
            print("[LLMClient] ✓ ainvoke response received")

            # Extract text content from the response
            if 'content' in response_body and len(response_body['content']) > 0 and 'text' in response_body['content'][0]:
                return response_body['content'][0]['text'].strip()
            else:
                return "No response generated"

        except Exception as e:
            print(f"✗ ERROR in ainvoke Bedrock call: {e}")
            print(f"   Error type: {type(e).__name__}")
            raise
