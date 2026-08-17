# compile

The Docker image (see [../Dockerfile](../Dockerfile)) builds this
automatically into `/usr/local/bin/move_proxy` — no manual step needed for
Docker-based deployment.

To build a standalone binary outside Docker:
```
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o move_proxy .
```
