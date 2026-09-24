# Run on Swing

Use the `swing` compute configuration to submit embedding workers from an
Argonne LCRC Swing login node through PBS Pro. Each Parsl block requests one
GPU and 16 CPU cores and runs one embedding task.

```yaml
compute_config:
  name: swing
  account: PROJECT
  queue: gpu
  walltime: '04:00:00'
  init_blocks: 1
  min_blocks: 0
  max_blocks: 4
  worker_init: |
    cd "$PBS_O_WORKDIR"
    source .venv/bin/activate
```

Replace `PROJECT` with the LCRC project to charge. Create the environment from
the repository root before launching:

```bash
uv sync --locked
source .venv/bin/activate
python -m genslm_embeddings.distributed_embeddings \
  --config examples/swing/esm2-8M.yaml
```

The login-node hostname is advertised to workers by default. Set `address` to
a compute-node-reachable login-node address when that hostname is not routable
from allocated nodes.

Swing permits up to four running jobs per user, so `max_blocks` is limited to
four. The `gpu` and `gpu-large` queues accept walltimes up to 24 hours, while
`backfill` accepts up to four hours. Use the shortest accurate walltime because
job duration affects queue priority.

Parsl metadata and logs are written below `<output_dir>/parsl`. Use `pbsq` or
`qstat` to inspect jobs and `qdel <jobid>` to cancel one.

See the LCRC [Swing job guide](https://argonne-lcrc.github.io/user-guides/swing/running-jobs-swing/)
and [PBS Pro guide](https://argonne-lcrc.github.io/user-guides/running-jobs-at-lcrc/pbs-pro/)
for current scheduler policies and job-management commands.
