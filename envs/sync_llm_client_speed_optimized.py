# sync_llm_client_speed_optimized.py - SPEED PRIORITIZED VERSION
# Optimized for high-CPU, high-concurrency environments with reliable LLM servers

import asyncio
import threading
import time
import logging
import queue
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, Future
import concurrent.futures
from dataclasses import dataclass
import sys
import multiprocessing

from envs.robust_llm_client_speed_optimized import create_speed_optimized_robust_llm_client as create_robust_llm_client

logger = logging.getLogger(__name__)

@dataclass
class LLMRequest:
    """Request object for LLM advice"""
    observations: Dict
    agent_type: str
    dc_ids: List[str]
    use_context: bool = True
    request_id: str = ""

class SpeedOptimizedSyncLLMClient:
    """
    SPEED-OPTIMIZED: Synchronous wrapper for async LLM client
    - Increased thread pool size for high parallelism
    - Faster timeouts and reduced retry overhead
    - Optimized for reliable LLM servers
    """
    
    def __init__(self, 
                 service_url: str = "http://10.93.232.106:8000",
                 timeout: float = 3.0,  # REDUCED: 5.0 → 3.0
                 max_concurrent: int = 64,  # INCREASED: 32 → 64
                 max_retries: int = 2,  # REDUCED: 3 → 2
                 history_window: int = 10,
                 num_envs: int = 8):  # NEW: Environment count for thread optimization
        
        self.service_url = service_url
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.history_window = history_window
        self.num_envs = num_envs
        
        # OPTIMIZED: Scale thread pool with environment count and available CPUs
        cpu_count = multiprocessing.cpu_count()
        # Use 4 threads per environment, capped by CPU count
        optimal_threads = min(max(num_envs * 4, 16), cpu_count * 2, 64)
        
        self.executor = ThreadPoolExecutor(
            max_workers=optimal_threads,
            thread_name_prefix="FastLLM"
        )
        
        self.async_client = None
        self.loop = None
        self.loop_thread = None
        
        # Client state
        self.is_initialized = False
        self.initialization_lock = threading.Lock()
        
        # OPTIMIZED: Smaller fallback cache for speed
        self.fallback_cache = {}
        
        # Statistics
        self.request_count = 0
        self.success_count = 0
        
        logger.info(f"🚀 SpeedOptimizedSyncLLMClient initialized:")
        logger.info(f"  URL: {service_url}")
        logger.info(f"  Timeout: {timeout}s (fast)")
        logger.info(f"  Max concurrent: {max_concurrent} (high)")
        logger.info(f"  Max retries: {max_retries} (minimal)")
        logger.info(f"  Thread pool: {optimal_threads} workers (scaled for {num_envs} envs)")
        logger.info(f"  CPU cores available: {cpu_count}")
    
    def _start_async_loop(self):
        """Start the async event loop in a separate thread"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            try:
                # SPEED-OPTIMIZED: Create async client with faster settings
                self.async_client = create_robust_llm_client(
                    service_url=self.service_url,
                    timeout=self.timeout,
                    max_concurrent=self.max_concurrent,
                    max_retries=self.max_retries,
                    history_window=self.history_window
                )
                
                # Run the event loop
                self.loop.run_forever()
            except Exception as e:
                logger.error(f"Error in async loop: {e}")
            finally:
                if self.async_client:
                    try:
                        self.loop.run_until_complete(self.async_client.close_session())
                    except:
                        pass
                self.loop.close()
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        
        # OPTIMIZED: Shorter wait time for faster startup
        max_wait = 3.0  # REDUCED: 5.0 → 3.0
        start_time = time.time()
        while self.loop is None and (time.time() - start_time) < max_wait:
            time.sleep(0.05)  # REDUCED: 0.1 → 0.05
        
        if self.loop is None:
            raise RuntimeError("Failed to start async event loop")
    
    def initialize(self):
        """Initialize the sync wrapper and async client"""
        with self.initialization_lock:
            if self.is_initialized:
                return
            
            try:
                # Start the async loop in a separate thread
                self._start_async_loop()
                
                # Initialize the async client
                if self.loop and self.async_client:
                    future = asyncio.run_coroutine_threadsafe(
                        self.async_client.initialize_session(), 
                        self.loop
                    )
                    future.result(timeout=5.0)  # REDUCED: 10.0 → 5.0
                    
                    self.is_initialized = True
                    logger.info("✅ SpeedOptimizedSyncLLMClient initialized successfully")
                else:
                    raise RuntimeError("Failed to create async loop or client")
                    
            except Exception as e:
                logger.error(f"Failed to initialize SpeedOptimizedSyncLLMClient: {e}")
                self.is_initialized = False
                raise
    
    def get_llm_advice_sync(self, 
                       observations: Dict, 
                       agent_type: str, 
                       dc_ids: List[str],
                       use_context: bool = True) -> Dict:
        """
        SPEED-OPTIMIZED: Get LLM advice synchronously with fast timeouts
        """
        self.request_count += 1
        
        # OPTIMIZED: Fast initialization check
        if not self.is_initialized:
            try:
                self.initialize()
            except Exception as e:
                logger.warning(f"Fast init failed for LLM client: {e}")
                return self._get_fallback_advice(dc_ids, agent_type)
        
        # OPTIMIZED: Fast availability check
        if not self.loop or not self.async_client:
            return self._get_fallback_advice(dc_ids, agent_type)
        
        try:
            # OPTIMIZED: Context logging only in debug mode
            if use_context and self.async_client and logger.isEnabledFor(logging.DEBUG):
                if agent_type == "manager":
                    history_len = len(self.async_client.context_manager.manager_history)
                    logger.debug(f"Context check for {agent_type}: history_length={history_len}")
                elif agent_type == "worker":
                    history_len = len(self.async_client.context_manager.worker_history)
                    logger.debug(f"Context check for {agent_type}: history_length={history_len}")
            
            # OPTIMIZED: Submit with faster timeout
            future = asyncio.run_coroutine_threadsafe(
                self.async_client.get_llm_advice_with_retry(
                    observations=observations,
                    agent_type=agent_type, 
                    dc_ids=dc_ids,
                    use_context=use_context
                ),
                self.loop
            )
            
            # OPTIMIZED: Shorter timeout for faster failure detection
            result = future.result(timeout=self.timeout + 0.5)  # Minimal buffer
            
            self.success_count += 1
            
            # OPTIMIZED: Only cache for same agent type to reduce memory
            self.fallback_cache[agent_type] = result
            
            # OPTIMIZED: Reduced logging in production
            if logger.isEnabledFor(logging.DEBUG):
                if result.get("metadata", {}).get("context_enhanced", False):
                    logger.debug(f"✅ Fast LLM advice with context: {agent_type}")
                else:
                    logger.debug(f"📝 Fast LLM advice: {agent_type}")
            
            return result
            
        except concurrent.futures.TimeoutError:
            logger.warning(f"⏱️ Fast LLM timeout for {agent_type} ({self.timeout}s)")
            return self._get_fallback_advice(dc_ids, agent_type)
            
        except Exception as e:
            logger.warning(f"❌ Fast LLM request failed for {agent_type}: {e}")
            return self._get_fallback_advice(dc_ids, agent_type)
    
    def update_context(self, observations: Dict, actions: Dict, rewards: Dict, 
                      trust_scores: Dict, llm_advice: Dict):
        """OPTIMIZED: Async context update with fire-and-forget"""
        if not self.is_initialized or not self.loop or not self.async_client:
            return
        
        try:
            # OPTIMIZED: Fire and forget - don't wait for result
            asyncio.run_coroutine_threadsafe(
                self._async_update_context(observations, actions, rewards, trust_scores, llm_advice),
                self.loop
            )
            # No waiting or error handling - pure async
        except Exception:
            pass  # OPTIMIZED: Silent failure for speed
    
    async def _async_update_context(self, observations: Dict, actions: Dict, rewards: Dict, 
                                   trust_scores: Dict, llm_advice: Dict):
        """Async context update helper - optimized version"""
        try:
            if self.async_client:
                self.async_client.update_context(observations, actions, rewards, trust_scores, llm_advice)
        except Exception:
            pass  # OPTIMIZED: Silent failure for speed
    
    def reset_episode(self):
        """OPTIMIZED: Fire-and-forget episode reset"""
        if not self.is_initialized or not self.loop or not self.async_client:
            return
        
        try:
            asyncio.run_coroutine_threadsafe(
                self._async_reset_episode(),
                self.loop
            )
        except Exception:
            pass  # OPTIMIZED: Silent failure for speed
    
    async def _async_reset_episode(self):
        """Async episode reset helper - optimized"""
        try:
            if self.async_client:
                self.async_client.reset_episode()
        except Exception:
            pass  # OPTIMIZED: Silent failure for speed
    
    def _get_fallback_advice(self, dc_ids: List[str], agent_type: str = "manager") -> Dict:
        """OPTIMIZED: Fast fallback advice generation"""
        # OPTIMIZED: Try cache first, but don't spend time on deep lookups
        cached_advice = self.fallback_cache.get(agent_type)
        if cached_advice:
            return cached_advice
        
        # OPTIMIZED: Pre-computed default advice for speed
        return {
            "manager": {
                dc_id: {
                    "suggested_action": 1,
                    "confidence": 0.3,  # Higher than original for better fallback
                    "reasoning_embedding": [0.5] * 8,
                    "context_used": False
                } for dc_id in dc_ids
            },
            "worker": {
                dc_id: {
                    "suggested_action": 1,
                    "confidence": 0.3,  # Higher confidence for faster decisions
                    "reasoning_embedding": [0.5] * 8,
                    "context_used": False
                } for dc_id in dc_ids
            },
            "metadata": {
                "manager_response_time_ms": 1.0,
                "worker_response_time_ms": 1.0,
                "avg_response_time_ms": 1.0,
                "service_type": "fast_fallback_sync_wrapper",
                "fallback_reason": "speed_optimized_fallback",
                "context_enhanced": False
            }
        }
    
    def get_stats(self) -> Dict:
        """Get client performance statistics"""
        success_rate = self.success_count / max(1, self.request_count)
        
        return {
            "total_requests": self.request_count,
            "successful_requests": self.success_count,
            "success_rate": success_rate,
            "is_initialized": self.is_initialized,
            "has_async_client": self.async_client is not None,
            "has_event_loop": self.loop is not None and not self.loop.is_closed(),
            "thread_pool_size": self.executor._max_workers,
            "optimization_level": "SPEED_PRIORITIZED"
        }
    
    def close(self):
        """OPTIMIZED: Fast cleanup with timeouts"""
        logger.info("🔒 Closing SpeedOptimizedSyncLLMClient")
        
        try:
            # OPTIMIZED: Fast shutdown
            if self.loop and not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.loop.stop)
                
            # OPTIMIZED: Short wait for thread cleanup
            if self.loop_thread and self.loop_thread.is_alive():
                self.loop_thread.join(timeout=2.0)  # REDUCED: 5.0 → 2.0
                
            # OPTIMIZED: Fast executor shutdown
            if self.executor:
                if sys.version_info >= (3, 9):
                    self.executor.shutdown(wait=True, timeout=2.0)  # REDUCED: 5.0 → 2.0
                else:
                    self.executor.shutdown(wait=False)  # OPTIMIZED: Don't wait on older Python
                
        except Exception as e:
            logger.error(f"Error during fast cleanup: {e}")
        
        self.is_initialized = False
        logger.info("✅ SpeedOptimizedSyncLLMClient closed")

def create_speed_optimized_sync_llm_client(
    service_url: str = "http://10.93.232.106:8000",
    timeout: float = 3.0,  # OPTIMIZED: Faster default
    max_concurrent: int = 64,  # OPTIMIZED: Higher default
    max_retries: int = 2,  # OPTIMIZED: Fewer retries
    history_window: int = 10,
    num_envs: int = 8  # NEW: For thread pool optimization
) -> SpeedOptimizedSyncLLMClient:
    """Create a speed-optimized synchronous LLM client wrapper"""
    
    return SpeedOptimizedSyncLLMClient(
        service_url=service_url,
        timeout=timeout,
        max_concurrent=max_concurrent,
        max_retries=max_retries,
        history_window=history_window,
        num_envs=num_envs
    )