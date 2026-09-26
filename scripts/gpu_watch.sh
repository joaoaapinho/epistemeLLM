# Track and log gpu stats every 5s.
nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,clocks.sm,power.draw --format=csv -l 5 >> logs/gpu.log