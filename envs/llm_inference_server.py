# llm_inference_server.py - ENHANCED WITH TEMPORAL CONTEXT SUPPORT
import asyncio
import aiohttp
import json
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import subprocess
import threading
import signal
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class LLMAdvice:
    suggested_actions: List[int]
    confidence_scores: List[float] 
    reasoning_summary: str
    response_time_ms: float

class BatchInferenceRequest(BaseModel):
    observations: Dict[str, Any]
    agent_type: str  # "manager" or "worker"
    dc_ids: List[str]
    request_id: str

# NEW: Enhanced request model with temporal context
class EnhancedBatchInferenceRequest(BaseModel):
    observations: Dict[str, Any]
    agent_type: str  # "manager" or "worker"
    dc_ids: List[str]
    request_id: str
    temporal_context: Optional[Dict[str, Any]] = None  # NEW: Temporal context
    context_enhanced: Optional[bool] = False  # NEW: Flag for enhanced processing

class VLLMInferenceEngine:
    """High-performance LLM interface using external vLLM backend with OpenAI-compatible API"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = None, 
                 max_concurrent_requests: int = 64, model_name: str = None):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.max_concurrent_requests = max_concurrent_requests
        self.model_name = model_name or f"datacenter-{port}"
        
        # Connection management
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.connector = None
        self.session = None
        
        # Server status
        self.is_ready = False
        
    async def check_server_ready(self, max_wait_time: int = 60):
        """Check if vLLM OpenAI-compatible server is ready"""
        start_time = time.time()
        
        logger.info(f"Checking if vLLM OpenAI server is ready at {self.base_url}")
        
        while time.time() - start_time < max_wait_time:
            try:
                async with aiohttp.ClientSession() as session:
                    # Try /v1/models endpoint (OpenAI-compatible)
                    async with session.get(f"{self.base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            # Check if our model is available
                            model_found = any(model.get('id') == self.model_name for model in data.get('data', []))
                            if model_found:
                                logger.info(f"vLLM OpenAI server ready at {self.base_url} with model {self.model_name}")
                                self.is_ready = True
                                return True
                            else:
                                logger.warning(f"Model {self.model_name} not found, available models: {[m.get('id') for m in data.get('data', [])]}")
            except Exception as e:
                logger.debug(f"Waiting for vLLM OpenAI server at {self.base_url}... {e}")
                await asyncio.sleep(2)
        
        logger.error(f"vLLM OpenAI server not ready at {self.base_url} after {max_wait_time} seconds")
        return False
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.connector = aiohttp.TCPConnector(
            limit=self.max_concurrent_requests,
            limit_per_host=self.max_concurrent_requests,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=60, connect=10)
        self.session = aiohttp.ClientSession(
            connector=self.connector, 
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()

    def _create_json_schema(self, agent_type: str, num_agents: int = 3) -> Dict:
        """Create JSON schema for structured generation"""
        if agent_type == "manager":
            return {
                "type": "object",
                "properties": {
                    "suggested_actions": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 2},
                        "minItems": num_agents,
                        "maxItems": num_agents,
                        "description": f"Exactly {num_agents} routing actions (0=DC0, 1=DC1, 2=DC2)"
                    },
                    "confidence_scores": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "minItems": num_agents,
                        "maxItems": num_agents,
                        "description": f"Exactly {num_agents} confidence scores"
                    },
                    "reasoning_summary": {
                        "type": "string",
                        "minLength": 5,
                        "maxLength": 200,
                        "description": "Brief reasoning for routing decisions"
                    }
                },
                "required": ["suggested_actions", "confidence_scores", "reasoning_summary"],
                "additionalProperties": False
            }
        else:  # worker
            return {
                "type": "object",
                "properties": {
                    "suggested_actions": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 1},
                        "minItems": num_agents,
                        "maxItems": num_agents,
                        "description": f"Exactly {num_agents} execution actions (0=defer, 1=execute)"
                    },
                    "confidence_scores": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "minItems": num_agents,
                        "maxItems": num_agents,
                        "description": f"Exactly {num_agents} confidence scores"
                    },
                    "reasoning_summary": {
                        "type": "string",
                        "minLength": 5,
                        "maxLength": 200,
                        "description": "Brief reasoning for execution decisions"
                    }
                },
                "required": ["suggested_actions", "confidence_scores", "reasoning_summary"],
                "additionalProperties": False
            }

    def _format_observation_text(self, observations: Dict, agent_type: str, 
                            temporal_context: Optional[Dict] = None) -> str:
        """Convert RL observations to natural language with optional temporal context"""
        logger.debug(f"Received observations keys: {list(observations.keys())}")
        logger.debug(f"Agent type: {agent_type}")
        logger.debug(f"Has temporal context: {temporal_context is not None}")
        
        if agent_type == "manager":
            # Manager formatting works correctly - no changes needed
            prompt = "You are an intelligent datacenter task routing manager.\n"
            
            if temporal_context:
                prompt += self._format_temporal_context_for_manager(temporal_context)
            
            prompt += "CURRENT SYSTEM STATE:\n"
            
            # Parse manager observations (existing code works fine)
            for key, obs in observations.items():
                if key.startswith("manager_"):
                    dc_id = key.split("_")[1]
                    
                    if isinstance(obs, list):
                        mgr_obs = np.array(obs)
                    elif isinstance(obs, np.ndarray):
                        mgr_obs = obs
                    else:
                        logger.warning(f"Unexpected obs type for {key}: {type(obs)}")
                        continue
                        
                    logger.debug(f"Processing {key} with obs shape: {mgr_obs.shape}")
                    
                    if len(mgr_obs) >= 26:
                        meta_task = mgr_obs[:8] if len(mgr_obs) >= 8 else mgr_obs[:min(8, len(mgr_obs))]
                        
                        prompt += f"DC {dc_id}:\n"
                        if len(meta_task) >= 7:
                            prompt += f"  Tasks pending: {int(meta_task[0])}\n"
                            prompt += f"  CPU requirement: {meta_task[1]:.1f}\n"
                            if len(meta_task) > 3:
                                prompt += f"  Memory requirement: {meta_task[3]:.1f}GB\n"
                            if len(meta_task) > 6:
                                prompt += f"  Urgency score: {meta_task[6]:.3f}\n"
                        
                        if len(mgr_obs) > 8:
                            remaining_obs = mgr_obs[8:]
                            prompt += f"  Additional metrics: {remaining_obs[:5].tolist()}\n"
                    else:
                        prompt += f"DC {dc_id}: Incomplete observation data (length: {len(mgr_obs)})\n"
            
            # Enhanced instructions for manager
            prompt += "\n**TASK: Provide exactly 3 routing decisions for 3 datacenters.**\n"
            prompt += "Action values: 0=Route to DC0, 1=Route to DC1, 2=Route to DC2\n"
            prompt += "Confidence: 0.0=No confidence, 1.0=Full confidence\n"
            prompt += "\nConsider:\n"
            prompt += "- Task urgency and resource requirements\n"
            prompt += "- Current datacenter capacity and load\n"
            prompt += "- Load balancing across datacenters\n"
            
            if temporal_context:
                recommendations = temporal_context.get("recommendations", [])
                if recommendations:
                    prompt += f"- Recent insights: {'; '.join(recommendations[:2])}\n"
            
            prompt += "\nYou MUST respond with valid JSON only."
            
        else:  # worker - FIXED VERSION
            prompt = "You are a datacenter worker agent managing task execution.\n"
            
            if temporal_context:
                prompt += self._format_temporal_context_for_worker(temporal_context)
            
            prompt += "CURRENT LOCAL SYSTEM STATE:\n"
            
            # FIXED: Parse worker observations correctly
            for key, obs_data in observations.items():
                if key.startswith("worker_"):
                    dc_id = key.split("_")[1]
                    prompt += f"Worker {dc_id}:\n"
                    
                    # FIXED: Handle different worker observation formats
                    if isinstance(obs_data, dict):
                        # Dictionary format - parse components
                        if "obs_worker_meta_task_i" in obs_data:
                            meta = obs_data["obs_worker_meta_task_i"]
                            if isinstance(meta, list) and len(meta) >= 7:
                                prompt += f"  Queued tasks: {int(meta[0])}\n"
                                prompt += f"  Total CPU needed: {meta[1]:.1f}\n"
                                prompt += f"  Total memory needed: {meta[3]:.1f}GB\n"
                                prompt += f"  Max urgency: {meta[6]:.3f}\n"
                        
                        if "obs_local_dc_i_for_worker" in obs_data:
                            local = obs_data["obs_local_dc_i_for_worker"]
                            if isinstance(local, list) and len(local) >= 5:
                                prompt += f"  CPU available: {local[0]:.1f}%\n"
                                prompt += f"  Memory available: {local[1]:.1f}%\n"
                                prompt += f"  Current load: {local[2]:.1f}%\n"
                                prompt += f"  Current CI: {local[3]:.3f}\n"
                    
                    elif isinstance(obs_data, (list, np.ndarray)):
                        # FIXED: Handle array/list format (like what we're seeing in logs)
                        obs_array = np.array(obs_data) if isinstance(obs_data, list) else obs_data
                        
                        if len(obs_array) >= 16:  # Expected worker obs length
                            # First 7 elements: task features
                            task_features = obs_array[:7]
                            prompt += f"  Queued tasks: {int(task_features[0])}\n"
                            prompt += f"  Total CPU needed: {task_features[1]:.1f}\n"
                            if len(task_features) > 3:
                                prompt += f"  Total memory needed: {task_features[3]:.1f}GB\n"
                            if len(task_features) > 6:
                                prompt += f"  Max urgency: {task_features[6]:.3f}\n"
                            
                            # Next 5 elements: datacenter features  
                            if len(obs_array) >= 12:
                                dc_features = obs_array[7:12]
                                prompt += f"  CPU available: {dc_features[0]:.1f}%\n"
                                prompt += f"  Memory available: {dc_features[1]:.1f}%\n"
                                prompt += f"  Current load: {dc_features[2]:.1f}%\n"
                                prompt += f"  Current CI: {dc_features[3]:.3f}\n"
                        else:
                            prompt += f"  Incomplete observation data (length: {len(obs_array)})\n"
                            # Show raw data for debugging
                            prompt += f"  Raw obs: {obs_array[:10].tolist()}\n"
                    
                    else:
                        prompt += f"  Unexpected worker obs format: {type(obs_data)}\n"
            
            # Enhanced instructions for worker
            prompt += "\n**TASK: Provide exactly 3 execution decisions for 3 workers.**\n"
            prompt += "Action values: 0=Defer execution, 1=Execute now\n"
            prompt += "Confidence: 0.0=No confidence, 1.0=Full confidence\n"
            prompt += "\nConsider:\n"
            prompt += "- Available resources vs task requirements\n"
            prompt += "- Task urgency levels\n"
            prompt += "- Current system load and capacity\n"
            
            if temporal_context:
                recommendations = temporal_context.get("recommendations", [])
                if recommendations:
                    prompt += f"- Recent insights: {'; '.join(recommendations[:2])}\n"
            
            prompt += "\nYou MUST respond with valid JSON only."
        
        logger.debug(f"Generated prompt length: {len(prompt)}")
        return prompt

    # NEW: Temporal context formatting methods
    def _format_temporal_context_for_manager(self, temporal_context: Dict) -> str:
        """Format temporal context for manager agents"""
        context_text = "\nRECENT PERFORMANCE CONTEXT:\n"
        
        # Episode performance
        episode_perf = temporal_context.get("episode_performance", {})
        if episode_perf.get("status") != "no_data":
            context_text += f"Episode Performance: {episode_perf.get('reward_trend', 'stable')} trend, "
            context_text += f"{episode_perf.get('total_steps', 0)} steps, "
            context_text += f"trust: {episode_perf.get('avg_trust_score', 0.5):.2f}\n"
        
        # System trends
        trends = temporal_context.get("trends", {})
        if trends.get("status") != "insufficient_data":
            perf_status = trends.get("recent_performance", "unknown")
            context_text += f"Recent System Performance: {perf_status}\n"
        
        # Recent history summary
        recent_history = temporal_context.get("recent_history", [])
        if recent_history:
            context_text += f"Last {len(recent_history)} decisions: "
            for i, step in enumerate(recent_history[-3:]):  # Last 3 steps
                mins_ago = step.get("time_ago_minutes", 0)
                context_text += f"{mins_ago:.1f}min ago, "
            context_text = context_text.rstrip(", ") + "\n"
        
        context_text += "\n"
        return context_text
    
    def _format_temporal_context_for_worker(self, temporal_context: Dict) -> str:
        """Format temporal context for worker agents"""
        context_text = "\nRECENT EXECUTION CONTEXT:\n"
        
        # Episode performance
        episode_perf = temporal_context.get("episode_performance", {})
        if episode_perf.get("status") != "no_data":
            context_text += f"Episode Status: {episode_perf.get('reward_trend', 'stable')} performance trend\n"
        
        # Execution patterns (if available)
        if "execution_patterns" in temporal_context:
            exec_patterns = temporal_context["execution_patterns"]
            if exec_patterns.get("status") != "no_data":
                pattern = exec_patterns.get("pattern", "balanced")
                context_text += f"Recent Execution Pattern: {pattern}\n"
        
        context_text += "\n"
        return context_text

    async def get_advice(self, observations: Dict, agent_type: str, dc_ids: List[str], 
                        temporal_context: Optional[Dict] = None) -> LLMAdvice:  # NEW: Context parameter
        """Get LLM advice using external vLLM server with OpenAI-compatible API"""
        start_time = time.time()
        
        if not self.is_ready:
            raise RuntimeError(f"vLLM server is not ready at {self.base_url}")
        
        # Create prompt and schema with optional context
        prompt = self._format_observation_text(observations, agent_type, temporal_context)
        schema = self._create_json_schema(agent_type, len(dc_ids))
        
        logger.debug(f"Schema: {schema}")
        logger.debug(f"Using temporal context: {temporal_context is not None}")
        
        # Make request with structured generation
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                result = await self._make_vllm_request(prompt, schema, agent_type)
                
                if result:
                    response_time = (time.time() - start_time) * 1000
                    context_indicator = "with context" if temporal_context else "basic"
                    logger.info(f"LLM advice generated {context_indicator} in {response_time:.1f}ms: "
                               f"actions={result['suggested_actions']}, confidence={result['confidence_scores']}")
                    
                    return LLMAdvice(
                        suggested_actions=result["suggested_actions"],
                        confidence_scores=result["confidence_scores"],
                        reasoning_summary=result["reasoning_summary"],
                        response_time_ms=response_time
                    )
                
            except Exception as e:
                last_exception = e
                logger.warning(f"vLLM request attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (2 ** attempt))
        
        # Fallback on failure
        logger.error(f"All vLLM requests failed, using fallback. Last error: {last_exception}")
        return self._get_fallback_advice(agent_type, len(dc_ids))

    async def _make_vllm_request(self, prompt: str, guided_json: Dict, agent_type: str) -> Optional[Dict]:
        """Make request to vLLM OpenAI-compatible server using /v1/completions endpoint"""
        async with self.semaphore:
            # Use OpenAI-compatible completions endpoint
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": 0.9,
                "max_tokens": 500,
                "guided_json": guided_json,  # vLLM supports this parameter directly
                "stop": ["<|eot_id|>", "<|end_of_text|>", "\n\n"]
            }
            
            logger.debug(f"Making vLLM request to {self.base_url}/v1/completions")
            
            try:
                async with self.session.post(
                    f"{self.base_url}/v1/completions",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    # For OpenAI-compatible API, response is in "choices[0].text"
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["text"].strip()
                        logger.debug(f"Raw vLLM response: {content}")
                        
                        # Parse JSON response
                        try:
                            parsed = self._extract_json_from_response(content)
                            if parsed:
                                validated = self._validate_and_clamp_output(parsed, agent_type, 3)
                                return validated
                            else:
                                logger.error(f"Failed to extract valid JSON from response")
                                logger.error(f"Raw content: {content}")
                                return None
                        except Exception as e:
                            logger.error(f"Failed to parse JSON response: {e}")
                            logger.error(f"Raw content: {content}")
                            return None
                    else:
                        logger.error(f"Unexpected response format: {result}")
                        return None
                        
            except Exception as e:
                logger.error(f"vLLM OpenAI API request failed: {e}")
                return None

    def _extract_json_from_response(self, content: str) -> Optional[Dict]:
        """Extract and parse JSON from response with multiple strategies"""
        if not content or not content.strip():
            return None
            
        content = content.strip()
        
        # Strategy 1: Try direct JSON parsing
        try:
            return json.loads(content)
        except:
            pass
        
        # Strategy 2: Look for JSON within code blocks
        import re
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        # Strategy 3: Find JSON object in text
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                # Check if it looks like our expected format
                if isinstance(parsed, dict) and any(key in parsed for key in 
                    ['suggested_actions', 'confidence_scores', 'decisions', 'routing_decisions']):
                    return self._normalize_json_format(parsed)
            except:
                continue
        
        # Strategy 4: Extract from partial responses (handle truncated JSON)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                try:
                    json_text = '\n'.join(lines[i:])
                    # Try to fix common JSON issues
                    json_text = json_text.rstrip(',\n\r\t ')
                    if not json_text.endswith('}'):
                        json_text += '}'
                    return json.loads(json_text)
                except:
                    continue
        
        return None

    def _normalize_json_format(self, parsed: Dict) -> Dict:
        """Normalize different JSON formats to our expected format"""
        # If already in correct format
        if "suggested_actions" in parsed and "confidence_scores" in parsed:
            return parsed
        
        # Handle different response formats
        suggested_actions = []
        confidence_scores = []
        reasoning_summary = "Generated decision"
        
        # Format 1: routing_decisions array
        if "routing_decisions" in parsed:
            decisions = parsed["routing_decisions"]
            for i, decision in enumerate(decisions):
                if "action" in decision:
                    suggested_actions.append(decision["action"])
                else:
                    suggested_actions.append(1)  # Default
                    
                if "confidence" in decision:
                    confidence_scores.append(decision["confidence"])
                else:
                    confidence_scores.append(0.5)  # Default
        
        # Format 2: decisions array  
        elif "decisions" in parsed:
            decisions = parsed["decisions"]
            for i, decision in enumerate(decisions):
                if "action" in decision:
                    suggested_actions.append(decision["action"])
                else:
                    suggested_actions.append(1)  # Default
                    
                if "confidence" in decision:
                    confidence_scores.append(decision["confidence"])
                else:
                    confidence_scores.append(0.5)  # Default
        
        # Format 3: Array of individual decisions
        elif isinstance(parsed, list):
            for decision in parsed:
                if isinstance(decision, dict):
                    if "Action" in decision:
                        suggested_actions.append(decision["Action"])
                    elif "action" in decision:
                        suggested_actions.append(decision["action"])
                    else:
                        suggested_actions.append(1)
                        
                    if "Confidence" in decision:
                        confidence_scores.append(decision["Confidence"])
                    elif "confidence" in decision:
                        confidence_scores.append(decision["confidence"])
                    else:
                        confidence_scores.append(0.5)
        
        # Ensure we have 3 decisions
        while len(suggested_actions) < 3:
            suggested_actions.append(1)
        while len(confidence_scores) < 3:
            confidence_scores.append(0.5)
            
        return {
            "suggested_actions": suggested_actions[:3],
            "confidence_scores": confidence_scores[:3],
            "reasoning_summary": reasoning_summary
        }

    def _validate_and_clamp_output(self, output_dict: Dict, agent_type: str, num_agents: int = 3) -> Dict:
        """Validate and clamp the structured output"""
        try:
            # First, ensure output_dict is actually a dictionary
            if not isinstance(output_dict, dict):
                logger.warning(f"Expected dict but got {type(output_dict)}: {output_dict}")
                raise ValueError(f"Output is not a dictionary: {type(output_dict)}")
            
            # Extract and validate actions
            actions = output_dict.get("suggested_actions", [])
            confidences = output_dict.get("confidence_scores", [])
            reasoning = output_dict.get("reasoning_summary", "Generated decision")
            
            # Handle case where actions/confidences might not be lists
            if not isinstance(actions, list):
                logger.warning(f"actions is not a list: {type(actions)} = {actions}")
                actions = [actions] if actions is not None else []
            if not isinstance(confidences, list):
                logger.warning(f"confidences is not a list: {type(confidences)} = {confidences}")
                confidences = [confidences] if confidences is not None else []
            
            # Ensure correct lengths
            while len(actions) < num_agents:
                actions.append(1)  # Default action
            while len(confidences) < num_agents:
                confidences.append(0.5)  # Default confidence
                
            actions = actions[:num_agents]
            confidences = confidences[:num_agents]
            
            # Clamp actions based on agent type
            if agent_type == "manager":
                actions = [max(0, min(2, int(a))) for a in actions]
            else:  # worker
                actions = [max(0, min(1, int(a))) for a in actions]
            
            # Clamp confidences to [0, 1]
            confidences = [max(0.0, min(1.0, float(c))) for c in confidences]
            
            # Validate reasoning
            reasoning = str(reasoning)[:200] if reasoning else "No reasoning provided"
            if len(reasoning.strip()) < 5:
                reasoning = "Standard control decision"
            
            return {
                "suggested_actions": actions,
                "confidence_scores": confidences,
                "reasoning_summary": reasoning
            }
            
        except Exception as e:
            logger.error(f"Error validating output: {e}")
            logger.error(f"Raw output_dict type: {type(output_dict)}, value: {output_dict}")
            # Return safe fallback
            default_action = 1 if agent_type == "manager" else 1
            return {
                "suggested_actions": [default_action] * num_agents,
                "confidence_scores": [0.1] * num_agents,
                "reasoning_summary": f"Validation failed: {str(e)[:100]}"
            }

    def _get_fallback_advice(self, agent_type: str, num_agents: int) -> LLMAdvice:
        """Generate fallback advice when vLLM fails"""
        default_action = 1
        return LLMAdvice(
            suggested_actions=[default_action] * num_agents,
            confidence_scores=[0.1] * num_agents,
            reasoning_summary="Fallback advice due to vLLM service failure",
            response_time_ms=1.0
        )

class VLLMServiceManager:
    """Manages multiple external vLLM inference engines with temporal context support"""
    
    def __init__(self, manager_host: str = "127.0.0.1", manager_port: int = 9001, 
                 worker_host: str = "127.0.0.1", worker_port: int = 9002):
        
        # Create engines for external vLLM servers
        self.manager_engine = VLLMInferenceEngine(
            host=manager_host,
            port=manager_port,
            max_concurrent_requests=32,
            model_name="datacenter-9001"  # Use the exact model name from your docker run
        )
        
        self.worker_engine = VLLMInferenceEngine(
            host=worker_host,
            port=worker_port,
            max_concurrent_requests=32,
            model_name="datacenter-9002"  # Use the exact model name from your docker run
        )
        
        # Performance tracking
        self.request_count = 0
        self.context_enhanced_requests = 0  # NEW: Track context usage
        self.total_response_time = 0.0
        
    async def start_services(self):
        """Check that external vLLM servers are ready"""
        logger.info("Checking external vLLM services...")
        
        # Check both servers are ready
        manager_ready = await self.manager_engine.check_server_ready()
        worker_ready = await self.worker_engine.check_server_ready()
        
        if not manager_ready:
            raise RuntimeError(f"Manager vLLM server not ready at {self.manager_engine.base_url}")
        if not worker_ready:
            raise RuntimeError(f"Worker vLLM server not ready at {self.worker_engine.base_url}")
        
        # Initialize HTTP sessions
        await self.manager_engine.__aenter__()
        await self.worker_engine.__aenter__()
        
        logger.info("All external vLLM services ready")
    
    async def shutdown_services(self):
        """Shutdown HTTP sessions (servers are external)"""
        logger.info("Shutting down vLLM service connections...")
        
        await asyncio.gather(
            self.manager_engine.__aexit__(None, None, None),
            self.worker_engine.__aexit__(None, None, None),
            return_exceptions=True
        )
        
        logger.info("vLLM service connections closed")

    async def get_batch_advice(self, observations: Dict, agent_type: str, dc_ids: List[str], 
                              temporal_context: Optional[Dict] = None) -> Dict[str, Dict]:  # NEW: Context parameter
        """Get advice for specified agent type using appropriate external vLLM engine"""
        start_time = time.time()
        
        # Track context usage
        if temporal_context:
            self.context_enhanced_requests += 1
            logger.debug(f"Processing context-enhanced request for {agent_type}")
        
        try:
            if agent_type == "manager":
                advice = await self.manager_engine.get_advice(
                    observations, "manager", dc_ids, temporal_context
                )
                
                result = {
                    "manager": {
                        dc_id: {
                            "suggested_action": advice.suggested_actions[i],
                            "confidence": advice.confidence_scores[i],
                            "reasoning_embedding": self._encode_reasoning(advice.reasoning_summary),
                            "context_used": temporal_context is not None  # NEW: Context flag
                        } for i, dc_id in enumerate(dc_ids)
                    },
                    "metadata": {
                        "manager_response_time_ms": advice.response_time_ms,
                        "worker_response_time_ms": 0.0,
                        "avg_response_time_ms": self.total_response_time / max(1, self.request_count),
                        "service_type": "external_vllm",
                        "context_enhanced": temporal_context is not None  # NEW
                    }
                }
                
            elif agent_type == "worker":
                advice = await self.worker_engine.get_advice(
                    observations, "worker", dc_ids, temporal_context
                )
                
                result = {
                    "worker": {
                        dc_id: {
                            "suggested_action": advice.suggested_actions[i],
                            "confidence": advice.confidence_scores[i],
                            "reasoning_embedding": self._encode_reasoning(advice.reasoning_summary),
                            "context_used": temporal_context is not None  # NEW: Context flag
                        } for i, dc_id in enumerate(dc_ids)
                    },
                    "metadata": {
                        "manager_response_time_ms": 0.0,
                        "worker_response_time_ms": advice.response_time_ms,
                        "avg_response_time_ms": self.total_response_time / max(1, self.request_count),
                        "service_type": "external_vllm",
                        "context_enhanced": temporal_context is not None  # NEW
                    }
                }
                
            else:
                # Fallback for unknown agent types
                result = {
                    "manager": {
                        dc_id: {
                            "suggested_action": 1,
                            "confidence": 0.1,
                            "reasoning_embedding": [0.5] * 8,
                            "context_used": False
                        } for dc_id in dc_ids
                    },
                    "worker": {
                        dc_id: {
                            "suggested_action": 1,
                            "confidence": 0.1,
                            "reasoning_embedding": [0.5] * 8,
                            "context_used": False
                        } for dc_id in dc_ids
                    },
                    "metadata": {
                        "manager_response_time_ms": 1.0,
                        "worker_response_time_ms": 1.0,
                        "avg_response_time_ms": 1.0,
                        "service_type": "external_vllm_fallback",
                        "context_enhanced": False
                    }
                }
            
            # Update metrics
            self.request_count += 1
            response_time = (time.time() - start_time) * 1000
            self.total_response_time += response_time
            
            return result
            
        except Exception as e:
            logger.error(f"Service error in get_batch_advice: {e}")
            # Return fallback result
            return {
                "error": {
                    dc_id: {
                        "suggested_action": 1,
                        "confidence": 0.1,
                        "reasoning_embedding": [0.1] * 8,
                        "context_used": False
                    } for dc_id in dc_ids
                },
                "metadata": {
                    "manager_response_time_ms": 1.0,
                    "worker_response_time_ms": 1.0,
                    "avg_response_time_ms": 1.0,
                    "service_type": "external_vllm_error_fallback",
                    "error": str(e),
                    "context_enhanced": False
                }
            }
    
    def _encode_reasoning(self, reasoning_text: str) -> List[float]:
        """Convert reasoning text to embedding vector"""
        import hashlib
        hash_obj = hashlib.md5(reasoning_text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Ensure exactly 8 dimensions
        embedding = []
        for i in range(0, 16, 2):  # 8 iterations: 0,2,4,6,8,10,12,14
            val = int(hash_hex[i:i+2], 16) / 255.0
            embedding.append(val)
        
        assert len(embedding) == 8, f"Expected 8 dimensions, got {len(embedding)}"
        return embedding

# FastAPI Server
app = FastAPI(title="Enhanced External vLLM LLM Advice Service", version="3.2.0")
service_manager: Optional[VLLMServiceManager] = None

@app.on_event("startup")
async def startup_event():
    global service_manager
    
    # Prevent multiple startup calls
    if service_manager is not None:
        logger.warning("Service manager already exists, skipping startup")
        return
        
    logger.info("Starting enhanced external vLLM-based LLM service...")
    
    # Get configuration from environment variables
    manager_host = os.environ.get("MANAGER_HOST", "127.0.0.1")
    manager_port = int(os.environ.get("MANAGER_PORT", "9001"))
    worker_host = os.environ.get("WORKER_HOST", "127.0.0.1")
    worker_port = int(os.environ.get("WORKER_PORT", "9002"))
    
    try:
        service_manager = VLLMServiceManager(
            manager_host=manager_host,
            manager_port=manager_port,
            worker_host=worker_host,
            worker_port=worker_port
        )
        await service_manager.start_services()
        logger.info("Enhanced external vLLM service startup complete")
        
        # Log server endpoints
        logger.info(f"Manager vLLM server: http://{manager_host}:{manager_port}")
        logger.info(f"Worker vLLM server: http://{worker_host}:{worker_port}")
        logger.info("Enhanced endpoints: /batch_advice and /batch_advice_enhanced")
            
    except Exception as e:
        logger.error(f"Failed to connect to external vLLM services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    global service_manager
    if service_manager:
        await service_manager.shutdown_services()

# EXISTING: Basic endpoint (backward compatible)
@app.post("/batch_advice")
async def get_batch_advice(request: BatchInferenceRequest):
    """Get LLM advice using external vLLM backends (basic, no context)"""
    if not service_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        result = await service_manager.get_batch_advice(
            request.observations, 
            request.agent_type,
            request.dc_ids
        )
        return {"status": "success", "data": result}
        
    except Exception as e:
        logger.error(f"Service error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Enhanced endpoint with temporal context support
@app.post("/batch_advice_enhanced")
async def get_batch_advice_enhanced(request: EnhancedBatchInferenceRequest):
    """Get LLM advice using external vLLM backends with temporal context"""
    if not service_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        result = await service_manager.get_batch_advice(
            request.observations, 
            request.agent_type,
            request.dc_ids,
            request.temporal_context  # NEW: Pass temporal context
        )
        return {"status": "success", "data": result}
        
    except Exception as e:
        logger.error(f"Enhanced service error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    if not service_manager:
        return {"status": "starting", "message": "Service initializing"}
    
    manager_ready = service_manager.manager_engine.is_ready if service_manager.manager_engine else False
    worker_ready = service_manager.worker_engine.is_ready if service_manager.worker_engine else False
    
    context_rate = service_manager.context_enhanced_requests / max(1, service_manager.request_count)
    
    return {
        "status": "healthy" if (manager_ready and worker_ready) else "degraded",
        "manager_ready": manager_ready,
        "worker_ready": worker_ready,
        "manager_url": service_manager.manager_engine.base_url,
        "worker_url": service_manager.worker_engine.base_url,
        "request_count": service_manager.request_count,
        "context_enhanced_requests": service_manager.context_enhanced_requests,  # NEW
        "context_enhancement_rate": f"{context_rate:.1%}",  # NEW
        "avg_response_time_ms": service_manager.total_response_time / max(1, service_manager.request_count),
        "service_type": "external_vllm_enhanced",  # NEW
        "available_endpoints": ["/batch_advice", "/batch_advice_enhanced"]  # NEW
    }

@app.get("/models")
async def list_models():
    """List connected external models with enhanced capabilities"""
    if not service_manager:
        return {"models": [], "status": "not_ready"}
    
    return {
        "models": [
            {
                "name": "manager",
                "url": service_manager.manager_engine.base_url,
                "model_name": service_manager.manager_engine.model_name,
                "ready": service_manager.manager_engine.is_ready,
                "supports_context": True  # NEW
            },
            {
                "name": "worker", 
                "url": service_manager.worker_engine.base_url,
                "model_name": service_manager.worker_engine.model_name,
                "ready": service_manager.worker_engine.is_ready,
                "supports_context": True  # NEW
            }
        ],
        "status": "ready",
        "enhanced_features": ["temporal_context", "performance_tracking", "decision_outcomes"]  # NEW
    }

def main():
    """Run the enhanced FastAPI server"""
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        workers=1
    )

if __name__ == "__main__":
    main()