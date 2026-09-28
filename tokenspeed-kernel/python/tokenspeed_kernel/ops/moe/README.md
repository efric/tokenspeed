# Token-sharded MoE tail

`token_sharded_moe_tail` is an optional semantic operation. On gfx950, its
Iris solution reduces the routed and shared BF16 producer partials by token
row, normalizes and projects each rank's local rows, then pushes the final
residual to every rank. Other vendors and unsupported inputs return `None` so
the caller retains its ordinary reduction and projection path.

The prepared domain is TP8, PP1, equal attention/MoE groups, widths
3584/7168, and 512–8192 rows in multiples of eight. The producer partials
must be consecutive views of the same Iris input. A disjoint prefix is read
without mutation; the exact borrowed result view may instead be the prefix
for an in-place update. Shifted overlap and weights overlapping collective
storage are rejected before launch.

The result is borrowed from a symmetric Iris workspace until the next tail on
that group. Clone it if it must survive later calls. The producer input and
ordinary producer-direct result use separate storage. The reduction shares
producer flags with other collectives, while the gather has its own flags and
epoch domain. All producers and consumers must be ordered on the calling
stream, including graph replay.
