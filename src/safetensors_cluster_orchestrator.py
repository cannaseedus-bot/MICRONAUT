"""
SafeTensors Cluster Orchestrator.

This is an example leader bot that coordinates model loader, tokenizer,
cluster manager, and inference workers. It is designed as a helper layer
for fold orchestrators and Micronauts, not a replacement.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SafeTensorsCluster")


class ModelFormat(Enum):
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    GGUF = "gguf"


@dataclass
class ModelShard:
    shard_id: str
    model_path: str
    format: ModelFormat
    device: str = "cpu"
    loaded: bool = False
    shard_range: Tuple[int, int] = (0, 0)
    memory_usage: int = 0
    process_pid: Optional[int] = None


@dataclass
class ModelConfig:
    name: str
    model_path: str
    tokenizer_path: str
    config_path: str
    format: ModelFormat = ModelFormat.SAFETENSORS
    num_shards: int = 1
    max_seq_length: int = 2048
    dtype: str = "float16"
    quantization: Optional[str] = None


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stream: bool = False


@dataclass
class InferenceResponse:
    request_id: str
    text: str
    tokens_generated: int
    time_taken: float
    model_shard: str
    finish_reason: str = "length"


class SafeTensorsClusterOrchestrator:
    def __init__(self, cluster_port: int = 8080, webui_port: int = 7860) -> None:
        self.cluster_port = cluster_port
        self.webui_port = webui_port
        self.model_shards: Dict[str, ModelShard] = {}
        self.running = True
        self.task_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.result_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.inference_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.model_configs: Dict[str, ModelConfig] = {}

        self.workers: Dict[str, subprocess.Popen[str]] = {}
        self.worker_types = {
            "model_loader": "model_loader_bot.py",
            "tokenizer_manager": "tokenizer_manager_bot.py",
            "inference_worker": "inference_worker_bot.py",
            "cluster_manager": "cluster_manager_bot.py",
        }

        self._create_worker_scripts()

        self.websocket_thread = threading.Thread(target=self._start_websocket_server, daemon=True)
        self.task_processor_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self.inference_processor_thread = threading.Thread(target=self._process_inference, daemon=True)

        self.loaded_models: Dict[str, List[str]] = {}

        logger.info("SafeTensors Cluster Orchestrator initialized on port %s", cluster_port)

    def _create_worker_scripts(self) -> None:
        scripts = {
            "model_loader_bot.py": MODEL_LOADER_SCRIPT,
            "tokenizer_manager_bot.py": TOKENIZER_MANAGER_SCRIPT,
            "inference_worker_bot.py": INFERENCE_WORKER_SCRIPT,
            "cluster_manager_bot.py": CLUSTER_MANAGER_SCRIPT,
        }

        for filename, content in scripts.items():
            if not os.path.exists(filename):
                with open(filename, "w", encoding="utf-8") as file_handle:
                    file_handle.write(content)
                logger.info("Created %s", filename)

        self._create_example_configs()

    def _create_example_configs(self) -> None:
        example_config = {
            "name": "tiny-llama-example",
            "model_type": "llama",
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,
        }

        os.makedirs("models", exist_ok=True)
        with open("models/tiny-llama-config.json", "w", encoding="utf-8") as file_handle:
            json.dump(example_config, file_handle, indent=2)

    def register_model(self, config: ModelConfig) -> str:
        model_id = f"model_{len(self.model_configs)}"
        self.model_configs[model_id] = config
        logger.info("Registered model: %s as %s", config.name, model_id)

        self.task_queue.put(
            {
                "type": "load_model",
                "model_id": model_id,
                "config": asdict(config),
            }
        )

        return model_id

    def load_model_shard(self, model_id: str, shard_index: int = 0) -> str:
        if model_id not in self.model_configs:
            raise ValueError(f"Model {model_id} not registered")

        config = self.model_configs[model_id]

        shard_id = f"{model_id}_shard_{shard_index}"
        shard = ModelShard(
            shard_id=shard_id,
            model_path=config.model_path,
            format=config.format,
        )

        worker_script = self.worker_types["model_loader"]
        process = subprocess.Popen(
            [sys.executable, worker_script, shard_id, config.model_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        shard.process_pid = process.pid
        self.model_shards[shard_id] = shard
        self.workers[shard_id] = process

        logger.info("Loading model shard %s", shard_id)

        threading.Thread(
            target=self._monitor_model_loading,
            args=(process, shard_id),
            daemon=True,
        ).start()

        return shard_id

    def _monitor_model_loading(self, process: subprocess.Popen[str], shard_id: str) -> None:
        try:
            stdout, stderr = process.communicate(timeout=30)

            if process.returncode == 0:
                self.model_shards[shard_id].loaded = True
                logger.info("Model shard %s loaded successfully", shard_id)

                model_id = shard_id.split("_shard", 1)[0]
                self.loaded_models.setdefault(model_id, []).append(shard_id)

                if stdout:
                    logger.info("Loader output: %s", stdout.strip())
            else:
                logger.error("Failed to load shard %s: %s", shard_id, stderr)

        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("Timeout loading shard %s", shard_id)

    def unload_model(self, model_id: str) -> None:
        if model_id in self.loaded_models:
            for shard_id in self.loaded_models[model_id]:
                if shard_id in self.workers:
                    self.workers[shard_id].terminate()
                if shard_id in self.model_shards:
                    del self.model_shards[shard_id]
            del self.loaded_models[model_id]
            logger.info("Unloaded model %s", model_id)

    def run_inference(self, model_id: str, request: InferenceRequest) -> str:
        if model_id not in self.loaded_models:
            raise ValueError(f"Model {model_id} not loaded")

        self.inference_queue.put({"model_id": model_id, "request": asdict(request)})

        return request.request_id

    def _process_inference(self) -> None:
        while self.running:
            try:
                inference_data = self.inference_queue.get(timeout=0.5)
                model_id = inference_data["model_id"]
                request_dict = inference_data["request"]

                if model_id in self.loaded_models:
                    shards = self.loaded_models[model_id]
                    if shards:
                        shard_id = shards[0]
                        self._spawn_inference_worker(shard_id, request_dict)

            except queue.Empty:
                continue
            except Exception as exc:
                logger.error("Error processing inference: %s", exc)

    def _spawn_inference_worker(self, shard_id: str, request_dict: Dict[str, Any]) -> None:
        worker_script = self.worker_types["inference_worker"]

        request_data = json.dumps({"shard_id": shard_id, "request": request_dict})
        request_file = f"temp_request_{request_dict['request_id']}.json"
        with open(request_file, "w", encoding="utf-8") as file_handle:
            file_handle.write(request_data)

        process = subprocess.Popen(
            [sys.executable, worker_script, request_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        threading.Thread(
            target=self._monitor_inference,
            args=(process, request_dict["request_id"], request_file),
            daemon=True,
        ).start()

    def _monitor_inference(
        self, process: subprocess.Popen[str], request_id: str, request_file: str
    ) -> None:
        try:
            stdout, stderr = process.communicate(timeout=60)

            if process.returncode == 0:
                response_lines = stdout.strip().split("\n") if stdout else []
                for line in response_lines:
                    if line.startswith("RESPONSE:"):
                        response_data = json.loads(line[9:])
                        logger.info("Inference completed for %s", request_id)
                        self.result_queue.put({"request_id": request_id, "response": response_data})
                        break
            else:
                logger.error("Inference failed for %s: %s", request_id, stderr)

        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("Inference timeout for %s", request_id)
        finally:
            if os.path.exists(request_file):
                os.remove(request_file)

    def _process_tasks(self) -> None:
        while self.running:
            try:
                task = self.task_queue.get(timeout=0.5)
                task_type = task.get("type")

                if task_type == "load_model":
                    self._handle_load_model(task)
                elif task_type == "unload_model":
                    self._handle_unload_model(task)
                elif task_type == "status":
                    self._handle_status_request(task)

            except queue.Empty:
                continue
            except Exception as exc:
                logger.error("Error processing task: %s", exc)

    def _handle_load_model(self, task: Dict[str, Any]) -> None:
        model_id = task["model_id"]
        config_dict = task["config"]

        config = ModelConfig(**config_dict)

        for shard_index in range(config.num_shards):
            shard_id = self.load_model_shard(model_id, shard_index)
            logger.info("Loading shard %s", shard_id)

    def _handle_unload_model(self, task: Dict[str, Any]) -> None:
        model_id = task["model_id"]
        self.unload_model(model_id)

    def _handle_status_request(self, task: Dict[str, Any]) -> None:
        response = {
            "status": "running",
            "loaded_models": list(self.loaded_models.keys()),
            "total_shards": len(self.model_shards),
            "loaded_shards": [
                shard.shard_id for shard in self.model_shards.values() if shard.loaded
            ],
        }

        response_queue = task.get("response_queue")
        if response_queue is not None:
            response_queue.put(response)

    def _start_websocket_server(self) -> None:
        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            while self.running:
                data = await reader.readline()
                if not data:
                    break
                message = data.decode().strip()
                try:
                    payload = json.loads(message)
                    command = payload.get("command")

                    if command == "load_model":
                        config = ModelConfig(**payload["config"])
                        model_id = self.register_model(config)
                        response = {"status": "loading", "model_id": model_id}
                    elif command == "inference":
                        request = InferenceRequest(**payload["request"])
                        request_id = self.run_inference(payload["model_id"], request)
                        response = {"status": "processing", "request_id": request_id}
                    elif command == "status":
                        response_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
                        self.task_queue.put({"type": "status", "response_queue": response_queue})
                        response = response_queue.get(timeout=5)
                    else:
                        response = {"status": "error", "error": "Unknown command"}

                except Exception as exc:
                    response = {"status": "error", "error": str(exc)}

                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()

            writer.close()
            await writer.wait_closed()

        async def main() -> None:
            server = await asyncio.start_server(handler, "127.0.0.1", self.cluster_port)
            async with server:
                await server.serve_forever()

        asyncio.run(main())

    def start(self) -> None:
        logger.info("Starting SafeTensors Cluster...")

        self.websocket_thread.start()
        self.task_processor_thread.start()
        self.inference_processor_thread.start()

        cluster_manager = subprocess.Popen(
            [sys.executable, self.worker_types["cluster_manager"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.workers["cluster_manager"] = cluster_manager

        logger.info("Cluster started on tcp://127.0.0.1:%s", self.cluster_port)
        logger.info("WebUI available on http://localhost:%s", self.webui_port)

    def stop(self) -> None:
        logger.info("Stopping SafeTensors Cluster...")
        self.running = False

        for _, process in self.workers.items():
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                process.kill()

        logger.info("Cluster stopped")


MODEL_LOADER_SCRIPT = '''#!/usr/bin/env python3
"""
Model Loader Bot - Loads safetensors models (simulated).
"""
import json
import sys
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelLoader")

class ModelLoaderBot:
    def __init__(self, shard_id: str, model_path: str):
        self.shard_id = shard_id
        self.model_path = model_path
        self.device = "cpu"

    def load_safetensors(self) -> bool:
        try:
            logger.info("Loading safetensors from %s", self.model_path)

            model_file = Path(self.model_path)
            if not model_file.exists():
                logger.error("Model file not found: %s", self.model_path)
                return False

            file_size = model_file.stat().st_size
            logger.info("Loaded safetensors file size: %s bytes", file_size)

            self.model = {
                "loaded": True,
                "device": self.device,
                "shard_id": self.shard_id,
                "memory_usage": max(file_size, 1),
            }
            return True

        except Exception as exc:
            logger.error("Error loading model: %s", exc)
            return False

    def run(self) -> None:
        logger.info("ModelLoader Bot started for shard %s", self.shard_id)
        time.sleep(1)

        success = self.load_safetensors()

        if success:
            print(f"SHARD_LOADED:{self.shard_id}", flush=True)
            print(f"STATUS:Model shard {self.shard_id} loaded successfully", flush=True)
            try:
                while True:
                    line = sys.stdin.readline()
                    if not line:
                        break

                    command = line.strip()
                    if command == "PING":
                        print("PONG", flush=True)
                    elif command == "STATUS":
                        print(json.dumps(self.model), flush=True)
                    elif command.startswith("INFERENCE:"):
                        print(f"RESPONSE:Inference simulated for shard {self.shard_id}", flush=True)
            except KeyboardInterrupt:
                pass
        else:
            print(f"ERROR:Failed to load shard {self.shard_id}", flush=True)
            sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python model_loader_bot.py <shard_id> <model_path>")
        sys.exit(1)

    shard_id = sys.argv[1]
    model_path = sys.argv[2]

    bot = ModelLoaderBot(shard_id, model_path)
    bot.run()
'''

TOKENIZER_MANAGER_SCRIPT = '''#!/usr/bin/env python3
"""
Tokenizer Manager Bot - Manages tokenizer files.
"""
import json
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TokenizerManager")

class TokenizerManagerBot:
    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = Path(tokenizer_path)
        self.vocab = {}
        self.merges = []

    def load_tokenizer_files(self) -> bool:
        try:
            tokenizer_files = {
                "vocab": self.tokenizer_path / "vocab.json",
                "merges": self.tokenizer_path / "merges.txt",
            }

            if tokenizer_files["vocab"].exists():
                with open(tokenizer_files["vocab"], "r", encoding="utf-8") as file_handle:
                    self.vocab = json.load(file_handle)
                logger.info("Loaded vocab with %s tokens", len(self.vocab))

            if tokenizer_files["merges"].exists():
                with open(tokenizer_files["merges"], "r", encoding="utf-8") as file_handle:
                    self.merges = [line.strip() for line in file_handle if line.strip()]
                logger.info("Loaded %s merges", len(self.merges))

            return True

        except Exception as exc:
            logger.error("Error loading tokenizer: %s", exc)
            return False

    def encode(self, text: str) -> list:
        tokens = []
        words = text.split()
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(abs(hash(word)) % 32000)

        return tokens

    def decode(self, tokens: list) -> str:
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        words = []
        for token in tokens:
            if token in reverse_vocab:
                words.append(reverse_vocab[token])
            else:
                words.append(f"[UNK:{token}]")

        return " ".join(words)

    def run(self) -> None:
        logger.info("Tokenizer Manager Bot started")

        if self.load_tokenizer_files():
            print("TOKENIZER_LOADED:OK", flush=True)

            try:
                while True:
                    line = sys.stdin.readline()
                    if not line:
                        break

                    command = line.strip()
                    if command.startswith("ENCODE:"):
                        text = command[7:]
                        tokens = self.encode(text)
                        print(f"TOKENS:{json.dumps(tokens)}", flush=True)

                    elif command.startswith("DECODE:"):
                        tokens_json = command[7:]
                        tokens = json.loads(tokens_json)
                        text = self.decode(tokens)
                        print(f"TEXT:{text}", flush=True)

                    elif command == "STATUS":
                        print(
                            json.dumps(
                                {
                                    "vocab_size": len(self.vocab),
                                    "merges": len(self.merges),
                                    "loaded": True,
                                }
                            ),
                            flush=True,
                        )

            except KeyboardInterrupt:
                pass
        else:
            print("TOKENIZER_LOADED:FAILED", flush=True)
            sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tokenizer_manager_bot.py <tokenizer_path>")
        sys.exit(1)

    tokenizer_path = sys.argv[1]
    bot = TokenizerManagerBot(tokenizer_path)
    bot.run()
'''

INFERENCE_WORKER_SCRIPT = '''#!/usr/bin/env python3
"""
Inference Worker Bot - Runs model inference (simulated).
"""
import json
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InferenceWorker")

class InferenceWorkerBot:
    def __init__(self, shard_id: str):
        self.shard_id = shard_id
        self.generation_count = 0

    def simulate_inference(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        time.sleep(0.2)

        tokens_generated = min(max_tokens, 50)
        words = prompt.split()[:5]
        generated_words = []
        for index in range(max(tokens_generated // 3, 1)):
            next_word = (words[index % len(words)] if words else "generated") + str(index)
            generated_words.append(next_word)

        generated_text = " ".join(generated_words)

        return {
            "text": generated_text,
            "tokens": tokens_generated,
            "time_taken": 0.2 + (tokens_generated * 0.01),
            "finish_reason": "length" if tokens_generated >= max_tokens else "stop",
        }

    def run_inference(self, request: dict) -> dict:
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 0.7)

        logger.info("Running inference for request %s", request.get("request_id"))

        result = self.simulate_inference(prompt, max_tokens, temperature)

        self.generation_count += 1

        return {
            "request_id": request.get("request_id"),
            "shard_id": self.shard_id,
            "generation": self.generation_count,
            **result,
        }

    def run(self, request_file: str) -> None:
        logger.info("Inference Worker Bot started for shard %s", self.shard_id)

        with open(request_file, "r", encoding="utf-8") as file_handle:
            request_data = json.load(file_handle)

        request = request_data.get("request")

        result = self.run_inference(request)

        print(f"RESPONSE:{json.dumps(result)}", flush=True)

        os.remove(request_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_worker_bot.py <request_file>")
        sys.exit(1)

    request_file = sys.argv[1]

    with open(request_file, "r", encoding="utf-8") as file_handle:
        request_data = json.load(file_handle)

    shard_id = request_data.get("shard_id")

    bot = InferenceWorkerBot(shard_id)
    bot.run(request_file)
'''

CLUSTER_MANAGER_SCRIPT = '''#!/usr/bin/env python3
"""
Cluster Manager Bot - Manages the model cluster.
"""
import json
import sys
import time
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClusterManager")

class ClusterManagerBot:
    def __init__(self) -> None:
        self.shard_registry: Dict[str, Dict] = {}
        self.load_balancer_index = 0

    def register_shard(self, shard_id: str, shard_info: Dict) -> None:
        self.shard_registry[shard_id] = {
            **shard_info,
            "registered_at": time.time(),
            "inference_count": 0,
            "last_used": time.time(),
        }
        logger.info("Registered shard %s", shard_id)

    def get_next_shard(self, model_id: str) -> str:
        model_shards = [shard for shard in self.shard_registry.keys() if shard.startswith(model_id)]

        if not model_shards:
            return ""

        self.load_balancer_index = (self.load_balancer_index + 1) % len(model_shards)
        selected_shard = model_shards[self.load_balancer_index]

        self.shard_registry[selected_shard]["inference_count"] += 1
        self.shard_registry[selected_shard]["last_used"] = time.time()

        return selected_shard

    def get_cluster_status(self) -> Dict:
        return {
            "total_shards": len(self.shard_registry),
            "active_shards": len(self.shard_registry),
            "shards": list(self.shard_registry.keys()),
            "total_inferences": sum(
                shard["inference_count"] for shard in self.shard_registry.values()
            ),
        }

    def run(self) -> None:
        logger.info("Cluster Manager Bot started")

        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break

                command = line.strip()

                if command.startswith("REGISTER:"):
                    data = json.loads(command[9:])
                    self.register_shard(data["shard_id"], data["shard_info"])
                    print(f"REGISTERED:{data['shard_id']}", flush=True)

                elif command.startswith("GET_SHARD:"):
                    model_id = command[10:]
                    shard = self.get_next_shard(model_id)
                    print(f"SHARD:{json.dumps({'shard_id': shard})}", flush=True)

                elif command == "STATUS":
                    status = self.get_cluster_status()
                    print(f"CLUSTER_STATUS:{json.dumps(status)}", flush=True)

                elif command.startswith("INFERENCE_RESULT:"):
                    data = json.loads(command[17:])
                    shard_id = data.get("shard_id")
                    if shard_id in self.shard_registry:
                        self.shard_registry[shard_id]["last_result"] = data
                    print(f"RESULT_RECORDED:{shard_id}", flush=True)

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Cluster Manager shutting down")

if __name__ == "__main__":
    bot = ClusterManagerBot()
    bot.run()
'''


def create_demo_files() -> None:
    scripts = {
        "model_loader_bot.py": MODEL_LOADER_SCRIPT,
        "tokenizer_manager_bot.py": TOKENIZER_MANAGER_SCRIPT,
        "inference_worker_bot.py": INFERENCE_WORKER_SCRIPT,
        "cluster_manager_bot.py": CLUSTER_MANAGER_SCRIPT,
    }

    for filename, content in scripts.items():
        with open(filename, "w", encoding="utf-8") as file_handle:
            file_handle.write(content)
        print(f"Created {filename}")

    os.makedirs("demo_models/tiny-llama", exist_ok=True)
    with open("demo_models/tiny-llama/model.safetensors", "w", encoding="utf-8") as file_handle:
        file_handle.write("demo")

    vocab = {f"token_{i}": i for i in range(1000)}
    vocab.update({"<|endoftext|>": 1000, "<|pad|>": 1001})

    with open("demo_models/tiny-llama/vocab.json", "w", encoding="utf-8") as file_handle:
        json.dump(vocab, file_handle, indent=2)

    with open("demo_models/tiny-llama/merges.txt", "w", encoding="utf-8") as file_handle:
        file_handle.write("# Example merges file\n")
        file_handle.write("t o\n")
        file_handle.write("h e\n")
        file_handle.write("l l\n")

    print("\nDemo files created in 'demo_models/' directory")


async def run_demo() -> None:
    print("=" * 60)
    print("SAFETENSORS MODEL CLUSTER DEMO")
    print("=" * 60)

    create_demo_files()

    orchestrator = SafeTensorsClusterOrchestrator()

    try:
        orchestrator.start()

        print("\n1. Cluster started successfully!")
        print(f"   TCP: tcp://127.0.0.1:{orchestrator.cluster_port}")
        print(f"   WebUI: http://localhost:{orchestrator.webui_port}")

        await asyncio.sleep(1)

        print("\n2. Registering demo model...")

        config = ModelConfig(
            name="tiny-llama-demo",
            model_path="demo_models/tiny-llama/model.safetensors",
            tokenizer_path="demo_models/tiny-llama",
            config_path="models/tiny-llama-config.json",
            format=ModelFormat.SAFETENSORS,
            num_shards=2,
        )

        model_id = orchestrator.register_model(config)
        print(f"   Model registered with ID: {model_id}")

        print("\n3. Loading model shards...")
        await asyncio.sleep(2)

        print("\n4. Testing inference...")

        request = InferenceRequest(
            request_id="demo_1",
            prompt="Hello, how are you today?",
            max_tokens=50,
            temperature=0.7,
        )

        request_id = orchestrator.run_inference(model_id, request)
        print(f"   Inference request sent: {request_id}")

        await asyncio.sleep(1)

        print("\nCluster is running. Press Ctrl+C to stop.")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(run_demo())
