"""CLI command: edgeshard infer — remote distributed inference.

Connects to deployed shards via Master's endpoint registry,
creates remote shard proxies, and runs pipeline decoding.

Usage:
    edgeshard infer "Hello, world!" --master 192.168.1.10:10500
    edgeshard infer "Hello" --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import asyncio
import time

from rich.console import Console

console = Console()


async def _run_inference(
    prompt: str,
    shard_addresses: list[str] | None,
    master: str,
    service_name: str | None,
    model_name: str | None,
    max_tokens: int,
) -> None:
    """Async implementation of inference."""
    from transformers import AutoTokenizer

    from edgeshard._grpc import edgeshard_pb2, edgeshard_pb2_grpc
    from edgeshard.runtime.decoder import GenerationConfig
    from edgeshard.transport.remote_shard import RemoteModelShard

    # ------------------------------------------------------------------
    # 1. Discover shard endpoints from Master
    # ------------------------------------------------------------------
    if shard_addresses is None:
        console.print(f"[cyan]Discovering shards from Master at {master}...[/cyan]")
        try:
            import grpc

            channel = grpc.insecure_channel(master)
            stub = edgeshard_pb2_grpc.WorkerServiceStub(channel)

            response = stub.GetShardEndpoints(
                edgeshard_pb2.GetShardEndpointsRequest(
                    service_name=service_name or "",
                )
            )
            channel.close()

            if not response.endpoints:
                console.print("[red]No deployed shards found.[/red]")
                console.print("[yellow]Run 'edgeshard deploy' first.[/yellow]")
                raise SystemExit(1)

            # Sort by layer_start for pipeline order
            endpoints = sorted(response.endpoints, key=lambda ep: ep.layer_start)

            console.print(
                f"[green]Found {len(endpoints)} shard(s) "
                f"in service '{response.service_name}'[/green]"
            )
            for ep in endpoints:
                flags = []
                if ep.is_first_shard:
                    flags.append("first")
                if ep.is_last_shard:
                    flags.append("last")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                console.print(
                    f"  [dim]{ep.shard_id}: {ep.data_address} "
                    f"(layers {ep.layer_start}:{ep.layer_end}, "
                    f"{ep.device}){flag_str}[/dim]"
                )
            console.print()

            # Build shard addresses and metadata
            shard_addresses = [ep.data_address for ep in endpoints]
            discovered_model = endpoints[0].model_name if endpoints else None
            shard_meta = [
                {
                    "shard_id": ep.shard_id,
                    "is_first": ep.is_first_shard,
                    "is_last": ep.is_last_shard,
                }
                for ep in endpoints
            ]

        except SystemExit:
            raise
        except Exception as e:
            console.print(f"[red]Error discovering shards: {e}[/red]")
            console.print("[yellow]Is the Master running?[/yellow]")
            raise SystemExit(1)
    else:
        # Manual mode — try to discover model from local plan/service files
        discovered_model = None
        shard_meta = [
            {"shard_id": f"shard-{i}", "is_first": (i == 0), "is_last": (i == len(shard_addresses) - 1)}
            for i in range(len(shard_addresses))
        ]

        # Try to read model name from local .edgeshard/plan.yaml or service.yaml
        try:
            from pathlib import Path
            plan_path = Path(".edgeshard/plan.yaml")
            service_path = Path(".edgeshard/service.yaml")

            if plan_path.exists():
                import yaml
                with open(plan_path) as f:
                    plan_data = yaml.safe_load(f)
                    discovered_model = plan_data.get("model")
            elif service_path.exists():
                import yaml
                with open(service_path) as f:
                    service_data = yaml.safe_load(f)
                    model_info = service_data.get("model", {})
                    discovered_model = model_info.get("name") if isinstance(model_info, dict) else model_info
        except Exception:
            pass  # Ignore errors reading local files

    # ------------------------------------------------------------------
    # 2. Resolve model name for tokenizer
    # ------------------------------------------------------------------
    resolved_model = model_name or discovered_model
    if not resolved_model:
        console.print("[red]Cannot determine model for tokenizer.[/red]")
        console.print("[yellow]Use --model to specify the model name/path.[/yellow]")
        raise SystemExit(1)

    console.print(f"[cyan]Loading tokenizer for {resolved_model}...[/cyan]")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_model, trust_remote_code=True
        )
    except Exception as e:
        console.print(f"[red]Failed to load tokenizer: {e}[/red]")
        console.print(
            "[yellow]Make sure the model is accessible locally or on HuggingFace.[/yellow]"
        )
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 3. Connect to remote shards
    # ------------------------------------------------------------------
    console.print(f"[cyan]Connecting to {len(shard_addresses)} remote shard(s)...[/cyan]")
    remote_shards = []
    for i, addr in enumerate(shard_addresses):
        meta = shard_meta[i] if i < len(shard_meta) else {}
        remote = RemoteModelShard(
            shard_id=meta.get("shard_id", f"shard-{i}"),
            address=addr,
            is_first_shard=meta.get("is_first", i == 0),
            is_last_shard=meta.get("is_last", i == len(shard_addresses) - 1),
        )
        remote_shards.append(remote)

    # ------------------------------------------------------------------
    # 4. Run pipeline inference
    # ------------------------------------------------------------------
    from edgeshard.common.identifiers import SessionId

    console.print(f"\n[bold]Generating...[/bold]")
    console.print(f"  Prompt: {prompt}")
    console.print(f"  Max tokens: {max_tokens}")
    console.print()

    # Encode prompt
    # Check if tokenizer has chat template (for chat models like Qwen2.5-Instruct)
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        # Use chat template for chat models
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        logger.info(f"Using chat template, input_ids shape: {input_ids.shape}")
    else:
        # Direct encode for base models
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
    prompt_len = input_ids.shape[1]

    # Create session on all shards
    session_id = SessionId.generate()
    max_seq_len = prompt_len + max_tokens
    for shard in remote_shards:
        shard.create_session(session_id, batch_size=1, max_seq_len=max_seq_len)

    try:
        # Prefill: process prompt through all shards
        t0 = time.time()
        tensor = input_ids
        for i, shard in enumerate(remote_shards):
            if shard.is_first_shard:
                tensor = await shard.prefill(session_id, input_ids=tensor)
            else:
                tensor = await shard.prefill(session_id, hidden_states_input=tensor)

        prefill_ms = (time.time() - t0) * 1000

        # Get first token
        logits = tensor  # Output from last shard
        next_token_logits = logits[0, -1, :]
        next_token = int(next_token_logits.argmax().item())

        generated_tokens = [next_token]
        eos_id = tokenizer.eos_token_id

        # Decode loop
        t1 = time.time()
        for step in range(max_tokens - 1):
            if eos_id is not None and next_token == eos_id:
                break

            # Run decode through all shards
            tensor_in = next_token
            for j, shard in enumerate(remote_shards):
                if shard.is_first_shard:
                    out = await shard.decode(session_id, token_id=tensor_in)
                else:
                    out = await shard.decode(session_id, hidden_states_input=tensor_in)
                tensor_in = out  # hidden_states or logits

            logits = tensor_in
            next_token_logits = logits[0, -1, :]
            next_token = int(next_token_logits.argmax().item())
            generated_tokens.append(next_token)

        decode_ms = (time.time() - t1) * 1000
        total_ms = (time.time() - t0) * 1000

        # Decode to text
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_text = prompt + generated_text

        # Print results
        console.print(f"[bold green]Output:[/bold green]")
        console.print(f"  {full_text}")
        console.print()
        console.print(f"[dim]  Tokens: {len(generated_tokens)}[/dim]")
        console.print(f"[dim]  Prefill: {prefill_ms:.0f} ms[/dim]")
        console.print(f"[dim]  Decode: {decode_ms:.0f} ms "
                       f"({decode_ms / max(len(generated_tokens), 1):.1f} ms/token)[/dim]")
        console.print(f"[dim]  Total: {total_ms:.0f} ms[/dim]")

    finally:
        # Release sessions
        for shard in remote_shards:
            try:
                shard.release_session(session_id)
            except Exception:
                pass
        # Close channels
        for shard in remote_shards:
            try:
                await shard.close()
            except Exception:
                pass


def run_inference(
    prompt: str,
    shard_addresses: list[str] | None = None,
    master: str = "localhost:10500",
    service_name: str | None = None,
    model_name: str | None = None,
    max_tokens: int = 100,
) -> None:
    """Run distributed inference across remote shards.

    Args:
        prompt: Input prompt text.
        shard_addresses: List of shard gRPC addresses. If None, auto-discover from Master.
        master: Master gRPC address for auto-discovery.
        service_name: Service name to query. If None, uses any deployed service.
        model_name: Model name/path for tokenizer. If None, auto-detect from endpoints.
        max_tokens: Maximum tokens to generate.
    """
    from edgeshard.common.logging import setup_logging

    setup_logging(level="INFO", component="infer")

    asyncio.run(
        _run_inference(
            prompt=prompt,
            shard_addresses=shard_addresses,
            master=master,
            service_name=service_name,
            model_name=model_name,
            max_tokens=max_tokens,
        )
    )
