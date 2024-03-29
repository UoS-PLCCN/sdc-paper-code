import gym


def _create_pbcn(
    _type: str, n: int, m: int, funcs, target_attr, all_attr, name: str, T: int = None
):
    assert m > 0 and n > 0

    if _type not in ["self-triggering", "sampled-data", "normal"]:
        raise ValueError(f"Invalid type {_type}")

    kwargs = {
        "logic_func_data": (
            ([f"u{i}" for i in range(1, m + 1)] if m > 1 else ["u"])
            + [f"x{i}" for i in range(1, n + 1)],
            [[] for _ in range(1, m + 1)] + funcs,
        ),
        "goal_config": {
            "all_attractors": all_attr,
            "target": target_attr,
        },
        "name": name,
    }

    if _type != "normal":
        kwargs["T"] = T
        kwargs["gamma"] = 0.9

    return gym.make(
        f"gym_PBN:PBCN-{_type + '-' if _type != 'normal' else ''}v0", **kwargs
    )


# _PBCN_4_1 = {
#     "n": 4,
#     "m": 1,
#     "target_attr": {(0, 0, 0, 1)},
#     "all_attr": [{(0, 0, 0, 1)}, {(0, 1, 0, 0)}],
#     "funcs": [
#         [("not x2 and not x4", 1)],
#         [("not x4 and not u and (x2 or x3)", 1)],
#         [("not x2 and not x4 and x1", 0.7), ("False", 0.3)],
#         [("not x2 and not x3", 1)],
#     ],
# }
# PBCN_4_1 = _create_pbcn("sampled-data", name="PBCN_4_1_SDC", **_PBCN_4_1)
# PBCN_4_1_STC = _create_pbcn("self-triggering", name="PBCN_4_1_STC", T=5, **_PBCN_4_1)
# PBCN_4_1_NORMAL = _create_pbcn("normal", name="PBCN_4_1_SHORTEST_PATH", **_PBCN_4_1)

# _PBCN_9_2 = {
#     "name": "PBCN_9_2",
#     "n": 9,
#     "m": 2,
#     "target_attr": {(1, 1, 1, 1, 1, 1, 0, 1, 1)},
#     "all_attr": [
#         {(0, 0, 1, 0, 0, 0, 1, 0, 0)},
#         {(0, 0, 0, 0, 0, 0, 1, 0, 0)},
#         {(0, 0, 0, 0, 0, 0, 1, 0, 0)},
#         {(1, 1, 1, 1, 1, 1, 0, 1, 1)},
#     ],
#     "funcs": [
#         [("not x7 and x3", 1)],
#         [("x1", 1)],
#         [("not u1", 1)],
#         [("x5 and x6", 1)],
#         [("not u1 and x2 and u2", 0.7), ("x5", 0.3)],
#         [("x1", 1)],
#         [("not x4 and not x8", 1)],
#         [("x4 or x5 or x9", 1)],
#         [("not u1 and (x5 or u2)", 0.6), ("x9", 0.4)],
#     ],
# }
# PBCN_9_2 = _create_pbcn("sampled-data", T=10, **_PBCN_9_2)
# PBCN_9_2_STC = _create_pbcn("self-triggering", T=5, **_PBCN_9_2)

# _PBCN_16_3 = {
#     "name": "PBCN_16_3",
#     "n": 16,
#     "m": 3,
#     "target_attr": {(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)},
#     "all_attr": [
#         {(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)},
#         {(0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0)},
#         {(0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0)},
#         {(0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0)},
#         {(0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0)},
#     ],
#     "funcs": [
#         [("x2 and not x16", 1)],
#         [("not (x5 or x3 or x16)", 1)],
#         [("(x2 or x3) and (not x16)", 1)],
#         [("x15 and not x16", 1)],
#         [("x4 and not x16", 1)],
#         [("not (x7 or x16)", 1)],
#         [("x15 and not x16 and u1", 1)],
#         [("x6 and (not (x15 or x16)) and u2", 1)],
#         [("(x8 or (x6 and not x11)) and not x16", 1)],
#         [("((x12 and not x13) or x9) and not x16", 1)],
#         [("not (x9 or x16)", 1)],
#         [("not (x14 or x16)", 1)],
#         [("not (x12 or x16)", 1)],
#         [("not (x9 or x16) and u3", 0.5), ("x14", 0.5)],
#         [("not (x9 or x16)", 1)],
#         [("x10 or x16", 1)],
#     ],
# }
# PBCN_16_3 = _create_pbcn("sampled-data", T=10, **_PBCN_16_3)
# PBCN_16_3_STC = _create_pbcn("self-triggering", T=5, **_PBCN_16_3)

# _PBCN_28_3 = {
#     "n": 28,
#     "m": 3,
#     # fmt: off
#     "target_attr": {(0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0)},
#     "all_attr": [{(0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0)}],
#     # fmt: on
#     "funcs": [
#         [("x6 and x3", 1)],
#         [("x25", 1)],
#         [("x2", 1)],
#         [("x28", 1)],
#         [("x21", 1)],
#         [("x5", 1)],
#         [("(x15 and u2) or (x26 and u2)", 1)],
#         [("x14", 1)],
#         [("x18", 1)],
#         [("x25 and x28", 1)],
#         [("not x9", 1)],
#         [("x24", 1)],
#         [("x12", 1)],
#         [("x28", 1)],
#         [("(not x20) and u1 and u2", 1)],
#         [("x3", 1)],
#         [("not x11", 1)],
#         [("x2", 1)],
#         [("(x10 and x11 and x25 and x28) or (x11 and x23 and x25 and x28)", 1)],
#         [("x7 or not x26", 1)],
#         [("x11 or x22", 1)],
#         [("x2 and x18", 1)],
#         [("x15", 1)],
#         [("x18", 1)],
#         [("x8", 1)],
#         [("not x4 and u3", 0.5), ("x26", 0.5)],
#         [("x7 or (x15 and x26)", 1)],
#         [("not x4 and x15 and x24", 1)],
#     ],
# }
# PBCN_28_3_5 = _create_pbcn("sampled-data", T=5, name="PBCN_28_3_SDC_5", **_PBCN_28_3)
# PBCN_28_3 = _create_pbcn("sampled-data", T=10, name="PBCN_28_3_SDC_10", **_PBCN_28_3)
# PBCN_28_3_15 = _create_pbcn("sampled-data", T=15, name="PBCN_28_3_SDC_15", **_PBCN_28_3)
# PBCN_28_3_STC = _create_pbcn("self-triggering", T=5, name="PBCN_28_3_STC", **_PBCN_28_3)
# PBCN_28_3_NORMAL = _create_pbcn("normal", name="PBCN_28_3_SHORTEST_PATH", **_PBCN_28_3)

# # Highly Probabilistic Networks
# _PBCN_7_1_HIGH_PROB = {
#     "n": 7,
#     "m": 1,
#     "target_attr": {(0, 0, 1, 0, 1, 1, 0)},
#     "all_attr": [{(0, 0, 1, 0, 1, 1, 0)}],
#     "funcs": [
#         [("u and not x6", 0.7), ("x1", 0.3)],
#         [("x1", 1)],
#         [("not x4", 0.8), ("x3", 0.2)],
#         [("x3 and not x5", 1)],
#         [("not x2 or not x3", 1)],
#         [("x5 and not x4 and u", 0.7), ("x6", 0.3)],
#         [("u and x7 and not x6", 0.8), ("x7", 0.2)],
#     ],
# }
# PBCN_7_1_HIGH_PROB_5 = _create_pbcn(
#     "sampled-data", T=5, name="PBCN_7_1_HIGH_PROB_SDC_5", **_PBCN_7_1_HIGH_PROB
# )
# PBCN_7_1_HIGH_PROB = _create_pbcn(
#     "sampled-data", T=10, name="PBCN_7_1_HIGH_PROB_SDC_10", **_PBCN_7_1_HIGH_PROB
# )
# PBCN_7_1_HIGH_PROB_15 = _create_pbcn(
#     "sampled-data", T=15, name="PBCN_7_1_HIGH_PROB_SDC_15", **_PBCN_7_1_HIGH_PROB
# )
# PBCN_7_1_HIGH_PROB_STC = _create_pbcn(
#     "self-triggering", T=5, name="PBCN_7_1_HIGH_PROB_STC", **_PBCN_7_1_HIGH_PROB
# )
# PBCN_7_1_HIGH_PROB_STC_10 = _create_pbcn(
#     "self-triggering", T=10, name="PBCN_7_1_HIGH_PROB_STC_10", **_PBCN_7_1_HIGH_PROB
# )
# PBCN_7_1_HIGH_PROB_NORMAL = _create_pbcn(
#     "normal", name="PBCN_7_1_HIGH_PROB_SHORTEST_PATH", **_PBCN_7_1_HIGH_PROB
# )


_PBCN_9_4_HIGH_PROB = {
    "n": 9,
    "m": 4,
    "target_attr": {(0, 1, 1, 1, 1, 0, 1, 1, 1)},
    "all_attr": [
        {(0, 1, 1, 1, 1, 0, 1, 1, 1)},
        {(0, 0, 0, 1, 0, 1, 0, 0, 0)},
    ],
    "funcs": [
        [("u1 and x8", 1)],
        [("(u2 and x9) or u1", 1)],
        [("(u2 or x1) and u3", 0.8), ("x3", 0.2)],
        [("not u4", 1)],
        [("x7", 0.8), ("not x7", 0.2)],
        [("not x3 and u3", 0.7), ("x6", 0.3)],
        [("x3 and x4 and not x6", 0.8), ("x3 and x4 and x6", 0.2)],
        [("x3 and x4", 1)],
        [("x8", 1)],
    ],
}
PBCN_9_4_HIGH_PROB = _create_pbcn(
    "sampled-data", T=10, name="PBCN_9_4_HIGH_PROB_SDC_10", **_PBCN_9_4_HIGH_PROB
)
# PBCN_9_4_HIGH_PROB_STC = _create_pbcn(
#     "self-triggering", T=5, name="PBCN_9_4_HIGH_PROB_STC", **_PBCN_9_4_HIGH_PROB
# )
# PBCN_9_4_HIGH_PROB_NORMAL = _create_pbcn(
#     "normal", name="PBCN_9_4_HIGH_PROB_SHORTEST_PATH", **_PBCN_9_4_HIGH_PROB
# )
