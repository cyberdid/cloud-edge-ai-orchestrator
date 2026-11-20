# Laptop-independent Colab operation

The Paper II feasibility pilot is controlled by a persistent e2-micro instance named `paper2-colab-watchdog`. The e2-micro starts and monitors the Colab L4 session; the laptop does not host the runner, watchdog, or checkpoints. Turning off the laptop therefore does not stop the active remote run.

## Persistence and recovery

- The supervisor is a systemd service on the e2-micro and starts automatically after VM boot.
- The Colab kernel runs the whole study driver. The e2-micro owns the Colab CLI OAuth state and can reconnect or recreate the L4 session.
- Each logged model response is uploaded to GCS immediately after append and before it is returned to OpenEvolve. OpenEvolve checkpoints and state are synchronized every 30 seconds, after each condition, and at clean exit.
- If the L4 runtime disappears or an execution is interrupted, the B2 entrypoint restores the run tree from `gs://oe-colab-scale-0903767759/paper2-v2/b2-estimation-2026-11-20/` before resuming. It validates the saved protocol/source identity, completed-arm inventory, manifests, and call ledgers, then resumes an incomplete arm from its OpenEvolve checkpoint and recorded call index.
- For the frozen B2 cohort, the watchdog stops only the `paper2-b2-estimation` L4 session if pooled account draw reaches 70.4 CU from the launch balance or the account balance reaches the 100-CU reserve. The account meter can include other Colab assignments, so the draw is pooled rather than attributable to B2 alone.
- A model call is committed to GCS before its response is returned to OpenEvolve. If a runtime ends while a candidate is being evaluated, that candidate evaluation may need to resume; the model call itself is already recorded and is not silently lost.

## Remote status

From any machine with the account's gcloud configuration:

```bash
gcloud compute ssh paper2-colab-watchdog \
  --project=gen-lang-client-0903767759 --zone=us-west4-a \
  --tunnel-through-iap --command='systemctl status paper2-b2-estimation.service --no-pager'

gcloud storage cat gs://oe-colab-scale-0903767759/paper2-v2/b2-estimation-2026-11-20/state.json
```

The `colab exec` call has a six-hour timeout. If it exits before the cohort reaches a terminal state, the systemd service restarts the watchdog; the watchdog checks whether the B2 session is still busy, avoids queuing a duplicate, and resumes from the GCS mirror when the session is available. On successful completion or a budget stop, the watchdog stops only the B2 L4 session. The e2-micro remains available for the next preregistered stage; stop it after all planned cloud work is complete.

## Credential boundaries

- The e2-micro has no inbound public SSH access. SSH is allowed through IAP only. Its ephemeral public address is used for outbound package and GitHub downloads.
- The e2-micro uses the bucket-scoped `oe-colab-sync` service account attached through VM metadata.
- The Colab runtime receives the same service-account key in `/content/.oe/sa-key.json`, with mode `0600` and parent directory mode `0700`. The subprocess candidate sandbox runs as uid 65534 with a cleared environment and cannot traverse that directory.
- The Colab OAuth refresh token is stored only under `/root/.config/colab-cli/` on the e2-micro. Do not copy it into the repository, GCS bucket, or Colab filesystem.
- The GCS prefix for this study is distinct from the previous scale study. Raw logs are append-oriented and are not overwritten by a different protocol.
