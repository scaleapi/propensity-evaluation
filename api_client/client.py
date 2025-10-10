import logging
import litellm

from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass
from typing import Any, Dict, List, Union
from api_client.api_conf import APIConfiguration

litellm.set_verbose = False

import os
import time
import psutil
import threading
from threading import Thread
import multiprocessing
import fcntl
import json
from litellm import RateLimitError

class RateLimiter:
    last_print_time = multiprocessing.Value('d', 0.0)  # Shared across all instances

    def __init__(self, api_keys: list[str], model_names: list[str], max_calls_per_minute: int, semaphore_dir: str,
                 rate_limit_enabled: bool):
        self.api_keys = api_keys
        self.model_names = model_names
        self.max_calls_per_minute = max_calls_per_minute
        self.semaphore_dir = semaphore_dir
        self.current_key_index = 0
        self.current_model_index = 0
        self.rate_limit_enabled = rate_limit_enabled
        self._initialize_semaphores()
        self._start_release_thread()
        self.process_start_time = psutil.Process(os.getpid()).create_time()

    def _initialize_semaphores(self):
        os.makedirs(self.semaphore_dir, exist_ok=True)
        for key in self.api_keys:
            for model in self.model_names:
                model = os.path.basename(model)  # Ensure model name is a simple string
                semaphore_file = self._get_semaphore_file(key, model)
                if not os.path.exists(semaphore_file):
                    with open(semaphore_file, 'w') as f:
                        json.dump([], f)

    def _get_semaphore_file(self, key: str, model: str) -> str:
        model = os.path.basename(model)  # Ensure model name is a simple string
        return os.path.join(self.semaphore_dir, f"{key}_{model}_semaphore.json")

    def _start_release_thread(self):
        def release_semaphore():
            while True:
                time.sleep(1)
                self._release_expired_calls()

        release_thread = Thread(target=release_semaphore, daemon=True)
        release_thread.start()

    def _release_expired_calls(self):
        current_time = time.time()
        tot_active_sessions = 0
        for key in self.api_keys:
            for model in self.model_names:
                semaphore_file = self._get_semaphore_file(key, model)
                with open(semaphore_file, 'r+') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    call_times = json.load(f)
                    call_times = [entry for entry in call_times if current_time - float(entry.split(":")[0]) < 60]
                    tot_active_sessions += len(call_times)
                    f.seek(0)
                    json.dump(call_times, f)
                    f.truncate()
                    fcntl.flock(f, fcntl.LOCK_UN)

        with RateLimiter.last_print_time.get_lock():
            if current_time - RateLimiter.last_print_time.value >= 5:
                RateLimiter.last_print_time.value = current_time
                print(f"Active sessions: {tot_active_sessions}")

    def _set_api_key(self, api_key: str):
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["ANTHROPIC_API_KEY"] = api_key
        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["XAI_API_KEY"] = api_key
        os.environ["REPLICATE_API_KEY"] = api_key
        os.environ["TOGETHERAI_API_KEY"] = api_key
        os.environ["FIREWORKS_AI_API_KEY"] = api_key
        os.environ["LITELLM_PROXY_API_KEY"] = api_key
        os.environ["LITE_LLM_KEY"] = api_key

    def switch_key_and_model(self):
        self.current_model_index = (self.current_model_index + 1) % len(self.model_names)
        if self.current_model_index == 0:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._set_api_key(self.api_keys[self.current_key_index])

    def acquire(self) -> tuple:
        failed = 0
        while True:
            current_key = self.api_keys[self.current_key_index]
            current_model = self.model_names[self.current_model_index]
            self._set_api_key(current_key)
            if not self.rate_limit_enabled:
                return current_key, current_model, None, False

            semaphore_file = self._get_semaphore_file(current_key, current_model)
            with open(semaphore_file, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                current_time = time.time()
                call_times = json.load(f)
                call_times = [entry for entry in call_times if current_time - float(entry.split(":")[0]) < 60]
                if len(call_times) < self.max_calls_per_minute:
                    call_times.append(f"{current_time}:{current_model}")
                    f.seek(0)
                    json.dump(call_times, f)
                    f.truncate()
                    fcntl.flock(f, fcntl.LOCK_UN)
                    return current_key, current_model, current_time, True
                fcntl.flock(f, fcntl.LOCK_UN)

            # Switch to the next key and model if the current one is exhausted
            self.switch_key_and_model()
            failed += 1
            if failed > 0 and failed % (len(self.api_keys) * len(self.model_names)) == 0:
                time.sleep(20)

    def release(self, session: tuple):
        current_key, current_model, call_time = session
        if call_time is None or not self.rate_limit_enabled:
            logging.warning("No call time provided to release or rate_limit_enabled is set to false.")
            return

        semaphore_file = self._get_semaphore_file(current_key, current_model)
        with open(semaphore_file, 'r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            call_times = json.load(f)
            call_times.remove(f"{call_time}:{current_model}")
            f.seek(0)
            json.dump(call_times, f)
            f.truncate()
            fcntl.flock(f, fcntl.LOCK_UN)


def get_fields_without_defaults(model: BaseModel) -> List[str]:
    fields_without_defaults = []
    for field_name, field_info in model.__fields__.items():
        if field_info.default is None and field_info.default_factory is None:
            fields_without_defaults.append(field_name)
    return fields_without_defaults


class Client:
    _warning_lock = threading.Lock()
    _proxy_warning_printed = False

    def __init__(self, api_conf: APIConfiguration, sys_prompt: str = None, output_schema=None,
                 temperature: float = 1):
        self.system_prompt = sys_prompt if sys_prompt is not None else ""

        self._model_name = api_conf.model_name
        self._model_provider = api_conf.model_provider
        self.api_base = api_conf.api_base
        self.use_cache = api_conf.use_cache

        assert os.environ.get("API_KEYS") is not None, "API_KEYS environment variable is not set"
        API_KEYS = [x.strip('"') for x in os.getenv("API_KEYS").strip("()").split(" ")]
        model_names = [self._model_name]

        use_rate_limiter = False
        if os.getenv('RATE_LIMIT') is not None:
            assert os.getenv('RATE_LIMIT') in ['true',
                                               'false'], "RATE_LIMIT environment variable must be 'true' or 'false'"
            use_rate_limiter = os.getenv('RATE_LIMIT') == 'true'

        if 'RATE_PM' in os.environ:
            max_calls_per_min = int(os.getenv('RATE_PM'))
            assert max_calls_per_min > 0, "RATE_PM must be a positive integer"
        elif use_rate_limiter:
            max_calls_per_min = 20
            logging.warning("LLM calling rate-per-min not set, defaulting to 20 calls per minute")
        else:
            max_calls_per_min = 1

        self.rate_limiter = RateLimiter(api_keys=API_KEYS, model_names=model_names, max_calls_per_minute=max_calls_per_min,
                                        semaphore_dir='.tmp/', rate_limit_enabled=use_rate_limiter)

        self.api_proxy = os.environ.get("API_PROXY", None)
        with Client._warning_lock:
            if not Client._proxy_warning_printed:
                if not self.api_proxy:
                    logging.warning("No API proxy provided. Defaulting to Litellm.")
                    self.api_proxy = 'litellm'
                else:
                    self.api_proxy = self.api_proxy.lower()
                    logging.warning(f"Using provided API proxy: {self.api_proxy}.")

                Client._proxy_warning_printed = True

        assert output_schema is None or isinstance(output_schema, BaseModel) or isinstance(output_schema,
                                                                                           ModelMetaclass) or isinstance(
            output_schema, dict)

        self.output_schema = output_schema
        if isinstance(output_schema, dict):
            if self.api_proxy == 'litellm':
                self.output_schema = {"type": "json_schema", "json_schema": {"schema": output_schema, "strict": True}}
            elif self.api_proxy == 'openai':
                self.output_schema = {"type": "json_schema",
                                      "json_schema": {"name": "schema", "schema": output_schema}}
            else:
                raise ValueError(f"Unsupported API proxy: {self.api_proxy}")

        self.temperature = temperature

    def __call__(self, query: Union[str, List[Dict[str, Any]]], **kwargs):
        session_key, session_model, session_time, session_valid = self.rate_limiter.acquire()
        session = None
        if session_valid:
            session = (session_key, session_model, session_time)

        if isinstance(query, str):
            messages: list[Dict[str, Any]] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
            ]
        elif isinstance(query, list):
            messages = query
        else:
            raise ValueError("Query must be a string or a list of messages")

        completion_args = {
            "model": session_model,
            "messages": messages,
            "temperature": self.temperature,
            "no-log": True,
            **kwargs
        }

        if self.api_base is not None:
            completion_args["api_base"] = self.api_base

        if self.output_schema is not None:
            completion_args["response_format"] = self.output_schema

        # Proxy-specific args
        assert self.api_proxy, "API proxy is not set"
        if self.api_proxy.lower() == 'litellm':
            completion_args["custom_llm_provider"] = self._model_provider
            completion_args["caching"] = self.use_cache

        try:
            if self.api_proxy == 'litellm':
                result = litellm.completion(**completion_args)
            elif self.api_proxy == 'openai':
                raise NotImplementedError("OpenAI API proxy is not implemented yet")
                # TODO: Implement this using openai.beta.chat.completions.create(**completion_args)
            else:
                raise ValueError(f"Unsupported API proxy: {self.api_proxy}")
        except RateLimitError:
            if session is not None:
                self.rate_limiter.release(session)
                self.rate_limiter.switch_key_and_model()
            raise

        return result