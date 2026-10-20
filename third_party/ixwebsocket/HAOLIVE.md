IXWebSocket v11.4.6, commit 2efe037c9cc96fd536774f17bdb5215161ee5087.
Source: https://github.com/machinezone/IXWebSocket . MIT license in LICENSE.
Vendored source builds statically, with TLS and compression disabled for the
owned loopback connection. Local patches: record port 0 via getsockname, cap
incoming frames and accumulated fragmented messages at 1 MiB.
