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

## Revoke after the study

```bash
gcloud iam service-accounts keys list --iam-account oe-colab-sync@gen-lang-client-0903767759.iam.gserviceaccount.com
gcloud iam service-accounts keys delete <KEY_ID> --iam-account oe-colab-sync@gen-lang-client-0903767759.iam.gserviceaccount.com
gcloud iam service-accounts delete oe-colab-sync@gen-lang-client-0903767759.iam.gserviceaccount.com
gcloud storage rm -r gs://oe-colab-scale-0903767759   # only after the data is downloaded
rm ~/.config/oe-colab/sa-key.json
```
