# robust_llm_client_speed_optimized.py - SPEED PRIORITIZED VERSION
# Optimized for high-performance LLM servers with reliable connectivity

import asyncio
import aiohttp
import time
import logging
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import json

logger = logging.getLogger(__name__)

@dataclass
class SpeedOptimizedRetryConfig:
    """SPEED-OPTIMIZED configuration for retry logic"""
    max_attempts: int = 2  # REDUCED: 5 → 2
    base_delay: float = 0.5  # REDUCED: 1.0 → 0.5
    max_delay: float = 10.0  # REDUCED: 30.0 → 10.0
    exponential_base: float = 1.5  # REDUCED: 2.0 → 1.5
    jitter: bool = False  # DISABLED: True → False (for consistent timing)
    timeout_multiplier: float = 1.2  # REDUCED: 1.5 → 1.2

class SpeedOptimizedTemporalContextManager:
    """SPEED-OPTIMIZED temporal context manager with faster serialization"""
    
    def __init__(self, history_window: int = 10):
        self.history_window = history_window
        
        # Store recent history for each agent type
        self.manager_history = deque(maxlen=history_window)
        self.worker_history = deque(maxlen=history_window)
        
        # Track decision outcomes
        self.decision_outcomes = deque(maxlen=history_window)
        
        # Episode-level context
        self.episode_start_time = time.time()
        self.episode_rewards = []
        self.episode_trust_scores = []
    
    def _convert_to_serializable(self, obj):
        """SPEED-OPTIMIZED: Fast conversion with reduced type checking"""
        # OPTIMIZED: Handle most common cases first for speed
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, 'tolist'):  # numpy array - fast path
            return obj.tolist()
        elif isinstance(obj, dict):
            # OPTIMIZED: Direct comprehension for speed
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # OPTIMIZED: Direct comprehension for speed
            return [self._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        else:
            return obj
        
    def add_step(self, observations: Dict, actions: Dict, rewards: Dict, 
                 trust_scores: Dict, llm_advice: Dict):
        """SPEED-OPTIMIZED: Add step with faster serialization"""
        # OPTIMIZED: Minimal serialization - only essential data
        step_data = {
            "timestamp": time.time(),
            "actions": self._convert_to_serializable(actions),
            "rewards": self._convert_to_serializable(rewards),
            "trust_scores": self._convert_to_serializable(trust_scores)
            # OPTIMIZED: Skip heavy observation serialization for speed
            # "observations": self._convert_to_serializable(observations),  # REMOVED
            # "llm_advice": self._convert_to_serializable(llm_advice),  # REMOVED
        }
        
        # OPTIMIZED: Fast agent type detection
        has_manager = any(k.startswith("manager_") for k in actions.keys())
        has_worker = any(k.startswith("worker_") for k in actions.keys())
        
        # OPTIMIZED: Direct assignment without extra checks
        if has_manager:
            self.manager_history.append(step_data)
        
        if has_worker:
            self.worker_history.append(step_data)
        
        # OPTIMIZED: Simplified episode tracking
        if rewards:
            if isinstance(rewards, dict) and rewards:
                reward_values = list(rewards.values())
                avg_reward = sum(reward_values) / len(reward_values) if reward_values else 0.0
            else:
                avg_reward = float(rewards) if rewards else 0.0
            self.episode_rewards.append(avg_reward)
        
        if trust_scores:
            if isinstance(trust_scores, dict) and trust_scores:
                trust_values = list(trust_scores.values())
                avg_trust = sum(trust_values) / len(trust_values) if trust_values else 0.0
            else:
                avg_trust = float(trust_scores) if trust_scores else 0.0
            self.episode_trust_scores.append(avg_trust)
    
    def get_context(self, agent_type: str, current_obs: Dict) -> Dict:
        """SPEED-OPTIMIZED: Get temporal context with minimal processing"""
        # OPTIMIZED: Pre-compute common values
        episode_perf = self._get_fast_episode_performance()
        
        context = {
            "episode_performance": episode_perf,
            "trends": self._get_fast_system_trends(agent_type),
            "recommendations": self._get_fast_recommendations(agent_type, episode_perf)
            # OPTIMIZED: Skip heavy processing items for speed
            # "recent_history": [],  # REMOVED for speed
            # "decision_outcomes": self._get_recent_decision_outcomes(),  # REMOVED for speed
        }
        
        return context
    
    def _get_fast_episode_performance(self) -> Dict:
        """SPEED-OPTIMIZED: Fast episode performance summary"""
        if not self.episode_rewards:
            return {"status": "no_data"}
        
        # OPTIMIZED: Simple calculations
        recent_rewards = self.episode_rewards[-5:] if len(self.episode_rewards) >= 5 else self.episode_rewards
        recent_trust = self.episode_trust_scores[-5:] if len(self.episode_trust_scores) >= 5 else self.episode_trust_scores
        
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
        avg_trust = sum(recent_trust) / len(recent_trust) if recent_trust else 0.0
        
        # OPTIMIZED: Simple trend detection
        if len(recent_rewards) > 1:
            trend = "improving" if recent_rewards[-1] > recent_rewards[0] else "stable"
        else:
            trend = "stable"
        
        return {
            "current_avg_reward": avg_reward,
            "reward_trend": trend,
            "total_steps": len(self.episode_rewards),
            "avg_trust_score": avg_trust
        }
    
    def _get_fast_system_trends(self, agent_type: str) -> Dict:
        """SPEED-OPTIMIZED: Fast system trends analysis"""
        history = self.manager_history if agent_type == "manager" else self.worker_history
        if len(history) < 2:
            return {"status": "insufficient_data"}
        
        # OPTIMIZED: Simple trend analysis
        recent_steps = list(history)[-3:]
        
        rewards = []
        for step in recent_steps:
            reward = step.get("rewards", 0)
            if isinstance(reward, dict) and reward:
                rewards.append(sum(reward.values()) / len(reward))
            else:
                rewards.append(float(reward) if reward else 0.0)
        
        if len(rewards) > 1:
            performance = "good" if rewards[-1] > rewards[0] else "stable"
        else:
            performance = "stable"
        
        return {
            "recent_performance": performance
        }
    
    def _get_fast_recommendations(self, agent_type: str, episode_perf: Dict) -> List[str]:
        """SPEED-OPTIMIZED: Fast contextual recommendations"""
        recommendations = []
        
        if episode_perf.get("reward_trend") == "declining":
            recommendations.append("Consider more conservative actions due to declining performance")
        
        if agent_type == "manager":
            recommendations.append("Focus on load balancing and CI optimization")
        else:  # worker
            recommendations.append("Consider resource constraints before executing tasks")
        
        return recommendations[:2]  # OPTIMIZED: Limit to 2 for speed
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episode_start_time = time.time()
        self.episode_rewards.clear()
        self.episode_trust_scores.clear()
        
        
class SpeedOptimizedRobustLLMClient:
    """
    SPEED-OPTIMIZED: Robust LLM client with minimal retry overhead
    Designed for reliable LLM servers where speed is critical
    """
    
    def __init__(self, 
                 service_url: str = "http://10.93.232.106:8000",
                 base_timeout: float = 2.0,  # REDUCED: 5.0 → 2.0
                 max_concurrent_requests: int = 64,  # INCREASED: 32 → 64
                 retry_config: Optional[SpeedOptimizedRetryConfig] = None,
                 history_window: int = 10):
        
        self.service_url = service_url
        self.base_timeout = base_timeout
        self.max_concurrent_requests = max_concurrent_requests
        self.retry_config = retry_config or SpeedOptimizedRetryConfig()
        
        # SPEED-OPTIMIZED: Temporal context management
        self.context_manager = SpeedOptimizedTemporalContextManager(history_window)
        
        # Connection management
        self.session = None
        self.connector = None
        self.semaphore = None
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.retry_count = 0
        self.total_response_time = 0.0
        self.fallback_cache = {}
        self.context_enhanced_requests = 0
        
        # SPEED-OPTIMIZED: Relaxed circuit breaker for faster recovery
        self.consecutive_failures = 0
        self.max_consecutive_failures = 25  # INCREASED: 10 → 25
        self.circuit_open_until = 0.0
        self.circuit_test_interval = 10.0  # REDUCED: 30.0 → 10.0
        
        logger.info(f"🚀 SpeedOptimizedRobustLLMClient initialized:")
        logger.info(f"  URL: {service_url}")
        logger.info(f"  Base timeout: {base_timeout}s (fast)")
        logger.info(f"  Max concurrent: {max_concurrent_requests} (high)")
        logger.info(f"  Max retries: {self.retry_config.max_attempts} (minimal)")
        logger.info(f"  Circuit breaker: {self.max_consecutive_failures} failures, {self.circuit_test_interval}s recovery")
        logger.info(f"  Context window: {history_window} (optimized)")
    
    async def __aenter__(self):
        """SPEED-OPTIMIZED: Async context manager entry"""
        # OPTIMIZED: Higher connection limits for throughput
        self.connector = aiohttp.TCPConnector(
            limit=self.max_concurrent_requests * 3,  # INCREASED multiplier: 2 → 3
            limit_per_host=self.max_concurrent_requests * 2,  # INCREASED
            ttl_dns_cache=600,  # INCREASED: 300 → 600 (less DNS lookups)
            use_dns_cache=True,
            keepalive_timeout=60,  # INCREASED: 30 → 60 (reuse connections more)
            enable_cleanup_closed=True,
            force_close=False
        )
        
        # OPTIMIZED: Faster timeouts
        timeout = aiohttp.ClientTimeout(
            total=self.base_timeout * self.retry_config.max_attempts,  # Total for all attempts
            connect=2.0,  # REDUCED: 5.0 → 2.0
            sock_read=self.base_timeout
        )
        
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "Connection": "keep-alive",
                "User-Agent": "SpeedOptimizedLLMClient/1.0"
            }
        )
        
        # OPTIMIZED: Higher semaphore limit
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        logger.info(f"✅ Speed-optimized HTTP session initialized")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
        logger.info("🔒 Speed-optimized HTTP session closed")

    def is_circuit_open(self) -> bool:
        """SPEED-OPTIMIZED: Relaxed circuit breaker check"""
        if self.consecutive_failures < self.max_consecutive_failures:
            return False
            
        current_time = time.time()
        if current_time > self.circuit_open_until:
            # OPTIMIZED: Faster recovery testing
            return False
            
        return True
    
    def record_success(self):
        """Record successful request"""
        self.success_count += 1
        self.consecutive_failures = 0
    
    def record_failure(self):
        """SPEED-OPTIMIZED: Record failure with relaxed circuit opening"""
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.circuit_open_until = time.time() + self.circuit_test_interval
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"⚠️ Circuit breaker OPENED - {self.consecutive_failures} failures")

    async def get_llm_advice_with_retry(self, 
                                   observations: Dict, 
                                   agent_type: str, 
                                   dc_ids: List[str],
                                   use_context: bool = True) -> Dict:
        """
        SPEED-OPTIMIZED: Get LLM advice with minimal retry overhead
        """
        self.request_count += 1
        start_time = time.time()
        
        # OPTIMIZED: Fast circuit breaker check
        if self.is_circuit_open():
            return self._get_fallback_advice(dc_ids, agent_type)
        
        # Ensure session is initialized
        if self.session is None:
            await self.initialize_session()
        
        last_exception = None
        current_timeout = self.base_timeout
        
        # SPEED-OPTIMIZED: Fewer retry attempts
        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                async with self.semaphore:
                    # OPTIMIZED: Fast request preparation
                    request_data = {
                        "observations": self._serialize_observations(observations),
                        "agent_type": agent_type,
                        "dc_ids": dc_ids,
                        "request_id": f"{agent_type}_req_{int(time.time() * 1000)}_{attempt}"
                    }
                    
                    # OPTIMIZED: Fast context check
                    should_use_context = False
                    if use_context:
                        if agent_type == "manager":
                            should_use_context = len(self.context_manager.manager_history) > 0
                        elif agent_type == "worker":
                            should_use_context = len(self.context_manager.worker_history) > 0
                    
                    # Add temporal context if available
                    if should_use_context:
                        temporal_context = self.context_manager.get_context(agent_type, observations)
                        request_data["temporal_context"] = temporal_context
                        request_data["context_enhanced"] = True
                        self.context_enhanced_requests += 1
                        endpoint = "/batch_advice_enhanced"
                    else:
                        endpoint = "/batch_advice"
                    
                    # OPTIMIZED: Make request with current timeout
                    response_data = await self._make_single_request(request_data, current_timeout, endpoint)
                    
                    if response_data:
                        # Success!
                        response_time = (time.time() - start_time) * 1000
                        self.total_response_time += response_time
                        self.record_success()
                        
                        # OPTIMIZED: Selective caching
                        self.fallback_cache[agent_type] = response_data
                        
                        if logger.isEnabledFor(logging.DEBUG):
                            context_indicator = "with context" if should_use_context else "basic"
                            logger.debug(f"✅ Fast LLM success on attempt {attempt}: {context_indicator} ({response_time:.1f}ms)")
                        
                        return response_data
                        
            except asyncio.TimeoutError as e:
                last_exception = e
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"⏱️ Fast LLM timeout on attempt {attempt}/{self.retry_config.max_attempts}")
                
            except (aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError) as e:
                last_exception = e
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"🔌 Fast LLM connection error on attempt {attempt}")
                
            except Exception as e:
                last_exception = e
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"❌ Fast LLM error on attempt {attempt}: {e}")
            
            # SPEED-OPTIMIZED: Shorter delays between attempts
            if attempt < self.retry_config.max_attempts:
                delay = self._calculate_fast_backoff_delay(attempt)
                current_timeout *= self.retry_config.timeout_multiplier
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"⏳ Fast retry in {delay:.1f}s")
                await asyncio.sleep(delay)
            
            self.retry_count += 1
        
        # All attempts failed
        self.record_failure()
        if logger.isEnabledFor(logging.ERROR):
            logger.error(f"💥 All {self.retry_config.max_attempts} fast attempts failed: {last_exception}")
        
        return self._get_fallback_advice(dc_ids, agent_type)
    
    async def _make_single_request(self, request_data: Dict, timeout: float, endpoint: str = "/batch_advice") -> Optional[Dict]:
        """SPEED-OPTIMIZED: Fast single HTTP request"""
        
        custom_timeout = aiohttp.ClientTimeout(total=timeout)
        
        async with self.session.post(
            f"{self.service_url}{endpoint}",
            json=request_data,
            timeout=custom_timeout
        ) as response:
            
            if response.status == 200:
                result = await response.json()
                advice_data = result.get("data", {})
                return advice_data
            elif response.status in (503, 429):
                # OPTIMIZED: Fast failure for overload conditions
                return None
            elif response.status == 404 and endpoint == "/batch_advice_enhanced":
                # Fallback to basic endpoint
                return await self._make_single_request(request_data, timeout, "/batch_advice")
            else:
                return None
    
    def _calculate_fast_backoff_delay(self, attempt: int) -> float:
        """SPEED-OPTIMIZED: Fast backoff calculation"""
        # OPTIMIZED: Minimal delays for speed
        delay = self.retry_config.base_delay * (self.retry_config.exponential_base ** (attempt - 1))
        delay = min(delay, self.retry_config.max_delay)
        
        # OPTIMIZED: No jitter for consistent timing
        return delay
    
    def _serialize_observations(self, obs: Dict) -> Dict:
        """SPEED-OPTIMIZED: Fast observation serialization"""
        return self.context_manager._convert_to_serializable(obs)
    
    def _get_fallback_advice(self, dc_ids: List[str], agent_type: str = "manager") -> Dict:
        """SPEED-OPTIMIZED: Fast fallback advice generation"""
        # OPTIMIZED: Direct cache lookup
        cached_advice = self.fallback_cache.get(agent_type)
        if cached_advice:
            return cached_advice
        
        # OPTIMIZED: Pre-computed fast fallback
        return {
            "manager": {
                dc_id: {
                    "suggested_action": 1,
                    "confidence": 0.3,
                    "reasoning_embedding": [0.5] * 8,
                    "context_used": False
                } for dc_id in dc_ids
            },
            "worker": {
                dc_id: {
                    "suggested_action": 1,
                    "confidence": 0.3,
                    "reasoning_embedding": [0.5] * 8,
                    "context_used": False
                } for dc_id in dc_ids
            },
            "metadata": {
                "manager_response_time_ms": 1.0,
                "worker_response_time_ms": 1.0,
                "avg_response_time_ms": 1.0,
                "service_type": "speed_optimized_fallback",
                "fallback_reason": "fast_fallback",
                "context_enhanced": False
            }
        }
    
    async def initialize_session(self):
        """Initialize session if not already done"""
        if self.session is None:
            await self.__aenter__()
    
    async def close_session(self):
        """Close session"""
        await self.__aexit__(None, None, None)
    
    def update_context(self, observations: Dict, actions: Dict, rewards: Dict, 
                      trust_scores: Dict, llm_advice: Dict):
        """Update temporal context with step results"""
        self.context_manager.add_step(observations, actions, rewards, trust_scores, llm_advice)
    
    def reset_episode(self):
        """Reset context for new episode"""
        self.context_manager.reset_episode()
    
    def get_stats(self) -> Dict:
        """Get client performance statistics"""
        success_rate = self.success_count / max(1, self.request_count)
        avg_response_time = self.total_response_time / max(1, self.success_count)
        context_rate = self.context_enhanced_requests / max(1, self.request_count)
        
        return {
            "total_requests": self.request_count,
            "successful_requests": self.success_count,
            "success_rate": success_rate,
            "retry_count": self.retry_count,
            "avg_response_time_ms": avg_response_time,
            "consecutive_failures": self.consecutive_failures,
            "circuit_open": self.is_circuit_open(),
            "concurrent_limit": self.max_concurrent_requests,
            "context_enhanced_requests": self.context_enhanced_requests,
            "context_enhancement_rate": context_rate,
            "optimization_level": "SPEED_PRIORITIZED"
        }


# Factory function for easy integration
def create_speed_optimized_robust_llm_client(
    service_url: str = "http://10.93.232.106:8000",
    timeout: float = 2.0,  # OPTIMIZED: Faster default
    max_concurrent: int = 64,  # OPTIMIZED: Higher default
    max_retries: int = 2,  # OPTIMIZED: Fewer retries
    history_window: int = 10
) -> SpeedOptimizedRobustLLMClient:
    """Create a speed-optimized robust LLM client for high-performance scenarios"""
    
    retry_config = SpeedOptimizedRetryConfig(
        max_attempts=max_retries,
        base_delay=0.5,
        max_delay=10.0,
        exponential_base=1.5,
        jitter=False,
        timeout_multiplier=1.2
    )
    
    return SpeedOptimizedRobustLLMClient(
        service_url=service_url,
        base_timeout=timeout,
        max_concurrent_requests=max_concurrent,
        retry_config=retry_config,
        history_window=history_window
    )