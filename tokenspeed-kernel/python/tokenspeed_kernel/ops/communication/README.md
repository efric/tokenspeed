# Communication operations

## Iris token-sharded MoE prefill

The CDNA4 MoE reduce-scatter and push gather kernels share the existing
`communication/iris.py` module with their Iris state and synchronization
helpers. The MoE host operation imports those kernels after validating its
prepared eight-rank group and storage.
