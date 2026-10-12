# Cluster examples

Only the names differ between clusters: partition, GPU type, account. Find yours with `sinfo -o "%P %G %l"` (partitions,
GPUs, time limits) and your cluster's documentation. Copy the lines you need into `env.sh`, or `source` a file from
here. Files here are examples, not defaults: nothing in the scripts depends on them.
