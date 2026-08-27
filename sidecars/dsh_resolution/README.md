# DSH Resolution Sidecar

This independent long-lived Node process owns Kazusa's DSH sessions. It uses
Node `^22.19.0 || >=24`, pnpm `11.7.0`, and exact DSH `0.1.1-rc.2` packages.

```powershell
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution install --frozen-lockfile
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution test
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution typecheck
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution build
```

Set `KAZUSA_DSH_SIDECAR_URL` to a loopback `/rpc` URL and provide
`KAZUSA_DSH_RPC_TOKEN`, `KAZUSA_DSH_DATA_ROOT`, and `KAZUSA_DSH_MODEL`.
Start `node sidecars/dsh_resolution/dist/src/main.js`. Send authenticated
JSON-RPC method `system.health` to check it. Stop with SIGTERM or Ctrl+C; the
Brain has an independent lifecycle.

The store is `<data-root>/dsh/0.1.1-rc.2/sessions.sqlite`. An incompatible DSH
release or store epoch rotates the segment rather than opening this store.
