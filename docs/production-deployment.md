# Production deployment and rollback (repository packaging)

**Scope:** repository production packaging only.  
**Not approved:** live production cutover, DNS/TLS/ingress changes, cloud deploy, or traffic switch.

## 1. Packaging artifacts

| Artifact | Role |
|----------|------|
| `.env.production.example` | Secret-free template (`APP_ENV=production`, `MOCK_LLM=false`) |
| `docker-compose.production.yml` | Override: external `APP_IMAGE`, restart, stop grace, healthcheck |
| `app/config.py` | Settings init: reject production + mock; require provider secret |
| `app/main.py` lifespan | Startup re-assert of production mock rejection |

## 2. Required production configuration

```bash
APP_ENV=production
MOCK_LLM=false
APP_IMAGE=<registry>/<image>:<immutable-tag>
OPENAI_API_KEY=<from secret manager — never commit>
# Prefer paired BGE mode for production FAQ quality:
QDRANT_COLLECTION=onlybook_faq_bge_m3_v1
PREFER_BGE=true
```

Hard rejection at process start if:

- `APP_ENV` is `production` or `prod` **and** `MOCK_LLM` is enabled
- production + secret-backed `LLM_PROVIDER` without `OPENAI_API_KEY` / `LLM_API_KEY`

## 3. Static Compose validation (no deploy)

```bash
docker compose --env-file .env.production.example \
  -f docker-compose.yml -f docker-compose.production.yml config
```

## 4. Deploy outline (platform-owned steps)

1. Build and push an immutable image tag (CI/registry — platform decision).
2. Materialize `.env.production` from the example via secret manager injection.
3. Confirm Qdrant collection + `PREFER_BGE` pairing and network reachability.
4. `docker compose --env-file .env.production -f docker-compose.yml -f docker-compose.production.yml up -d`
5. Verify `GET /health` (liveness) and `GET /ready` (readiness snapshot).
6. Run a bounded multilingual smoke against the production candidate **before** traffic switch.

## 5. Rollback

**Config rollback (hash baseline):**

```bash
QDRANT_COLLECTION=onlybook_faq
PREFER_BGE=false
```

Redeploy/restart the API with the previous known-good image tag (`APP_IMAGE`).

**Do not** delete `onlybook_faq` while it remains the rollback target.

## 6. Platform decisions still required

- TLS termination / ingress / DNS
- Secret manager and key rotation
- Registry, image signing, SBOM, vulnerability policy
- Qdrant HA, auth, and backups
- Autoscaling, multi-AZ, and production monitoring alerts
- Formal production cutover approval and traffic switch

## 7. Decision labels

| Decision | Status |
|----------|--------|
| Repository production packaging | Implemented in-repo (M9) |
| Actual production cutover | **NOT APPROVED** |
| Platform / infra decisions | **REQUIRED** before live traffic |
