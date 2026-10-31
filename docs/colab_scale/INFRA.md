# Colab scale study — infrastructure

Created by the owner with `scripts/colab/create_bucket.sh` (Claude Code's safety
policy does not let the agent mint service-account keys itself).

| item | value |
|---|---|
| Project | `gen-lang-client-0903767759` (billing account `01B53F-…`) |
| Bucket | `gs://oe-colab-scale-0903767759`, region `us-west4` (same region as the Colab L4 VMs seen so far), uniform bucket-level access |
| Service account | `oe-colab-sync@gen-lang-client-0903767759.iam.gserviceaccount.com` |
| Permission | `roles/storage.objectAdmin` on this bucket only; no project-level roles |
| Key | laptop `~/.config/oe-colab/sa-key.json` (600); copied to each VM as `/root/.oe/sa-key.json` (600, `/root` is 700); never committed, never readable by candidate code (uid 65534) |

Cost: a few hundred MB of run logs — well under one US cent per month for storage;
uploads are free; same-region traffic from the VMs is free.

## Always-on watchdog (2026-10-31)

The controller runs as user `cyberdid` on GCE instance `oe-watchdog`, zone
`us-central1-a`. Its user timer checks once per minute; the laptop timer is inactive.
`oe-colab-watchdog.service` performs one recovery/check cycle, while the separate
`oe-colab-client.service` holds the long-running notebook connection with an explicit
48-hour CLI execution timeout. This keeps the client outside the watchdog's oneshot
cgroup. The timeout does not extend the provider's VM lifetime or guarantee that
Colab will keep an assignment available.

A recent `running` heartbeat does not suppress recovery when the server reports no
study VM. A cross-host live assignment (`?`) and a recent `starting` phase retain
duplicate protection. Results remain in GCS; scientific settings and the spending
cap are unchanged.

Inspect on the GCE host:

```bash
systemctl --user status oe-colab-watchdog.timer oe-colab-watchdog.service oe-colab-client.service
journalctl --user -u oe-colab-client.service -n 30
tail -10 ~/.cache/oe-colab/watchdog.log
```

## Revoke after the study

```bash
gcloud iam service-accounts keys list --iam-account oe-colab-sync@gen-lang-client-0903767759.iam.gserviceaccount.com
gcloud iam service-accounts keys delete <KEY_ID> --iam-account oe-colab-sync@gen-lang-client-0903767759.iam.gserviceaccount.com
gcloud iam service-accounts delete oe-colab-sync@gen-lang-client-0903767759.iam.gserviceaccount.com
gcloud storage rm -r gs://oe-colab-scale-0903767759   # only after the data is downloaded
rm ~/.config/oe-colab/sa-key.json
```
