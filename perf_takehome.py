"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    Instruction,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

from scheduler import (ScratchRegPool, Scheduler)

def vcompare_keys(round, i, *rest):
    ret = tuple((round, k, *rest) for k in range(i, i+VLEN))
    return ret


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.pool = ScratchRegPool()
        self.const_map = {}
        self.v_const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.pool.scratch_debug)

    def scratch_const(self, val):
        assert val in self.const_map, f"Unreserved const: {val}"
        return self.const_map[val]

    def scratch_v_const(self, val):
        assert val in self.v_const_map, f"Unreserved v_const: {val}"
        return self.v_const_map[val]

    def preload_const(self, val):
        if val in self.const_map:
            return []
        addr = self.pool.alloc(f"CONST[0x{val:X}]")
        self.const_map[val] = addr
        return [("load", ("const", addr, val))]

    def preload_v_const(self, val):
        if val in self.v_const_map:
            return []
        addr = self.pool.alloc(f"V_CONST[0x{val:X}]", length=VLEN)
        self.v_const_map[val] = addr
        return self.preload_const(val) + [("valu", ("vbroadcast", addr, self.scratch_const(val)))]

    #Robert Jenkins, jenkins32: https://gist.github.com/badboy/6267743#robert-jenkins-32-bit-integer-hash-function
    def build_hash(self, val_hash_reg, tmp1, tmp2, round, i):
        slots = []

        # alu(self, core, op, dest, a1, a2)
        slots.append(("alu", ("+", tmp1, val_hash_reg, self.scratch_const(0x7ED55D16)))) # parity=P
        slots.append(("alu", ("<<", tmp2, val_hash_reg, self.scratch_const(12)))) # parity=0
        slots.append(("alu", ("+", val_hash_reg, tmp1, tmp2))) # reg = parity=P
        slots.append(("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 0))))

        slots.append(("alu", ("^", tmp1, val_hash_reg, self.scratch_const(0xC761C23C)))) # parity=P
        slots.append(("alu", (">>", tmp2, val_hash_reg, self.scratch_const(19)))) # parity=reg&(1<<19)
        slots.append(("alu", ("^", val_hash_reg, tmp1, tmp2))) # regparity = P ^ (reg&(1<<19))
        slots.append(("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 1))))

        slots.append(("alu", ("+", tmp1, val_hash_reg, self.scratch_const(0x165667B1)))) # parity=P^1
        slots.append(("alu", ("<<", tmp2, val_hash_reg, self.scratch_const(5)))) #parity = 0
        slots.append(("alu", ("+", val_hash_reg, tmp1, tmp2))) # parity = P^1
        slots.append(("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 2))))

        slots.append(("alu", ("+", tmp1, val_hash_reg, self.scratch_const(0xD3A2646C)))) # parity=P
        slots.append(("alu", ("<<", tmp2, val_hash_reg, self.scratch_const(9)))) # parity=0
        slots.append(("alu", ("^", val_hash_reg, tmp1, tmp2))) # regparity = P
        slots.append(("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 3))))

        slots.append(("alu", ("+", tmp1, val_hash_reg, self.scratch_const(0xFD7046C5)))) #parity=P^1
        slots.append(("alu", ("<<", tmp2, val_hash_reg, self.scratch_const(3)))) #parity=0
        slots.append(("alu", ("+", val_hash_reg, tmp1, tmp2))) #regparity = P^1
        slots.append(("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 4))))

        slots.append(("alu", ("^", tmp1, val_hash_reg, self.scratch_const(0xB55A4F09))))
        slots.append(("alu", (">>", tmp2, val_hash_reg, self.scratch_const(16))))
        slots.append(("alu", ("^", val_hash_reg, tmp1, tmp2)))
        slots.append(("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 5))))

        return slots

    def build_v_hash(self, reg, tmp1, tmp2, round, i):
        slots = []
        # valu has "multiply_add", we should exploit this
        # alu(self, core, op, dest, a1, a2)
        slots.append(("valu", ("+", tmp1, reg, self.scratch_v_const(0x7ED55D16))))
        slots.append(("valu", ("multiply_add", reg, reg, self.scratch_v_const(2**12), tmp1)))
        # slots.append(("valu", ("<<", tmp2, reg, self.scratch_v_const(12))))
        # slots.append(("valu", ("+", reg, tmp1, tmp2)))
        slots.append(("debug", ("vcompare", reg, vcompare_keys(round, i, "hash_stage", 0))))

        slots.append(("valu", ("^", tmp1, reg, self.scratch_v_const(0xC761C23C))))
        slots.append(("valu", (">>", tmp2, reg, self.scratch_v_const(19))))
        slots.append(("valu", ("^", reg, tmp1, tmp2)))
        slots.append(("debug", ("vcompare", reg, vcompare_keys(round, i, "hash_stage", 1))))

        slots.append(("valu", ("+", tmp1, reg, self.scratch_v_const(0x165667B1))))
        slots.append(("valu", ("multiply_add", reg, reg, self.scratch_v_const(2**5), tmp1)))
        # slots.append(("valu", ("<<", tmp2, reg, self.scratch_v_const(5)))) #parity = 0
        # slots.append(("valu", ("+", reg, tmp1, tmp2)))
        slots.append(("debug", ("vcompare", reg, vcompare_keys(round, i, "hash_stage", 2))))

        slots.append(("valu", ("+", tmp1, reg, self.scratch_v_const(0xD3A2646C))))
        slots.append(("valu", ("<<", tmp2, reg, self.scratch_v_const(9))))
        slots.append(("valu", ("^", reg, tmp1, tmp2)))
        slots.append(("debug", ("vcompare", reg, vcompare_keys(round, i, "hash_stage", 3))))

        slots.append(("valu", ("+", tmp1, reg, self.scratch_v_const(0xFD7046C5))))
        slots.append(("valu", ("multiply_add", reg, reg, self.scratch_v_const(2**3), tmp1)))
        # slots.append(("valu", ("<<", tmp2, reg, self.scratch_v_const(3))))
        # slots.append(("valu", ("+", reg, tmp1, tmp2)))
        slots.append(("debug", ("vcompare", reg, vcompare_keys(round, i, "hash_stage", 4))))

        slots.append(("valu", ("^", tmp1, reg, self.scratch_v_const(0xB55A4F09))))
        slots.append(("valu", (">>", tmp2, reg, self.scratch_v_const(16))))
        slots.append(("valu", ("^", reg, tmp1, tmp2)))
        slots.append(("debug", ("vcompare", reg, vcompare_keys(round, i, "hash_stage", 5))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        """
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]

        body = []  # array of slots

        # Const preloads
        body += self.preload_const(0x7ED55D16)
        body += self.preload_const(12)
        body += self.preload_const(0xC761C23C)
        body += self.preload_const(19)
        body += self.preload_const(0x165667B1)
        body += self.preload_const(5)
        body += self.preload_const(0xD3A2646C)
        body += self.preload_const(9)
        body += self.preload_const(0xFD7046C5)
        body += self.preload_const(3)
        body += self.preload_const(0xB55A4F09)
        body += self.preload_const(16)
        body += self.preload_const(0)
        body += self.preload_const(1)
        body += self.preload_const(2)
        body += self.preload_const(VLEN)

        body += self.preload_v_const(0x7ED55D16)
        body += self.preload_v_const(2**12) # useful for multiply_add
        body += self.preload_v_const(0xC761C23C)
        body += self.preload_v_const(19)
        body += self.preload_v_const(0x165667B1)
        body += self.preload_v_const(2**5) # useful for multiply_add
        body += self.preload_v_const(0xD3A2646C)
        body += self.preload_v_const(9)
        body += self.preload_v_const(0xFD7046C5)
        body += self.preload_v_const(2**3) # useful for multiply_add
        body += self.preload_v_const(0xB55A4F09)
        body += self.preload_v_const(16)
        body += self.preload_v_const(0)
        body += self.preload_v_const(1)
        body += self.preload_v_const(2)

        # Scratch space addresses

        input_reg = {}
        for i, v in enumerate(init_vars):
            # consider: replacing const register with tmp <- add_imm ZERO #i
            body += self.preload_const(i);
            input_reg[v] = self.pool.alloc(v, 1)
            body.append(("load", ("load", input_reg[v], self.scratch_const(i))))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        vlen_const = self.scratch_const(VLEN)

        zero_v_const = self.scratch_v_const(0)
        one_v_const = self.scratch_v_const(1)
        two_v_const = self.scratch_v_const(2)

        input_forest_p_v_const = self.pool.alloc(name="V_CONST[FOREST_P]", length=VLEN)
        body.append(("valu", ("vbroadcast", input_forest_p_v_const, input_reg["forest_values_p"])))
        input_n_v_const = self.pool.alloc(name="V_CONST[N]", length=VLEN)
        body.append(("valu", ("vbroadcast", input_n_v_const, input_reg["n_nodes"])))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.instrs.append({"flow": [("pause",)]})
        # Any debug engine instruction is ignored by the submission simulator
        self.instrs.append({"debug": [("comment", "Starting loop")]})

        # this code only works with small batch sizes! bcs it is not smart enough ;)
        # instead of working in memory, let's put into register
        idx_regstore = [-1] * batch_size
        val_regstore = [-1] * batch_size
        idx_v_regstore = [-1] * batch_size
        val_v_regstore = [-1] * batch_size
        SIMD_LIMIT = batch_size

        # so much things to prepare for simd, ew
        tmp_addr = self.pool.alloc()
        tmp_addr2 = self.pool.alloc()
        body.append(("alu", ("+", tmp_addr, zero_const, input_reg["inp_indices_p"])))
        body.append(("alu", ("+", tmp_addr2, zero_const, input_reg["inp_values_p"])))

        i = 0
        while i + VLEN - 1 < SIMD_LIMIT:
            idx_v_regstore[i] = self.pool.alloc(f"idx[{i}:{i+VLEN-1}]", length=VLEN)
            val_v_regstore[i] = self.pool.alloc(f"val[{i}:{i+VLEN-1}]", length=VLEN)
            body.append(("load", ("vload", idx_v_regstore[i], tmp_addr)))
            body.append(("load", ("vload", val_v_regstore[i], tmp_addr2)))
            body.append(("debug", ("vcompare", idx_v_regstore[i], vcompare_keys(0, i, "idx"))))
            body.append(("debug", ("vcompare", val_v_regstore[i], vcompare_keys(0, i, "val"))))

            i += VLEN
            body.append(("alu", ("+", tmp_addr, tmp_addr, vlen_const)))
            body.append(("alu", ("+", tmp_addr2, tmp_addr2, vlen_const)))

        while i < batch_size:
            idx_regstore[i] = self.pool.alloc(f"idx[{i}]")
            val_regstore[i] = self.pool.alloc(f"val[{i}]")
            body.append(("load", ("load", idx_regstore[i], tmp_addr)))
            body.append(("load", ("load", val_regstore[i], tmp_addr2)))

            i += 1
            body.append(("alu", ("+", tmp_addr, tmp_addr, one_const)))
            body.append(("alu", ("+", tmp_addr2, tmp_addr2, one_const)))
        self.pool.free(tmp_addr)
        self.pool.free(tmp_addr2)

        for round in range(rounds):
            i = 0
            while i + VLEN - 1 < SIMD_LIMIT:
                # Vector scratch registers
                tmp_addr = self.pool.alloc(length=VLEN)
                tmp1 = self.pool.alloc(length=VLEN)
                tmp2 = self.pool.alloc(length=VLEN)
                tmp_node_val = self.pool.alloc(length=VLEN)
                tmp_idx_move = self.pool.alloc(length=VLEN)
                assert idx_v_regstore[i] != -1, f"Unaligned access {i}"
                assert val_v_regstore[i] != -1, f"Unaligned access {i}"

                # idx = mem[inp_indices_p + i]
                body.append(("debug", ("vcompare", idx_v_regstore[i], vcompare_keys(round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("debug", ("vcompare", val_v_regstore[i], vcompare_keys(round, i, "val"))))

                # node_val = mem[forest_values_p + idx]
                body.append(("valu", ("+", tmp_addr, input_forest_p_v_const, idx_v_regstore[i])))
                for offset in range(VLEN):
                    body.append(("load", ("load_offset", tmp_node_val, tmp_addr, offset)))
                body.append(("debug", ("vcompare", tmp_node_val, vcompare_keys(round, i, "node_val"))))

                # val = myhash(val ^ node_val)
                # mem[inp_values_p + i] = val
                body.append(("valu", ("^", val_v_regstore[i], val_v_regstore[i], tmp_node_val)))
                body.extend(self.build_v_hash(val_v_regstore[i], tmp1, tmp2, round, i))
                body.append(("debug", ("vcompare", val_v_regstore[i], vcompare_keys(round, i, "hashed_val"))))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                # idx --> 2*idx + (val&1) + 1
                body.append(("valu", ("<<", idx_v_regstore[i], idx_v_regstore[i], one_v_const)))
                body.append(("valu", ("&", tmp_idx_move, val_v_regstore[i], one_v_const)))
                body.append(("valu", ("+", tmp_idx_move, tmp_idx_move, one_v_const)))
                body.append(("valu", ("+", idx_v_regstore[i], idx_v_regstore[i], tmp_idx_move)))
                body.append(("debug", ("vcompare", idx_v_regstore[i], vcompare_keys(round, i, "next_idx"))))

                # idx = 0 if idx >= n_nodes else idx
                # ---> idx = cond*idx, cond = idx < n_nodes
                # mem[inp_indices_p + i] = idx
                body.append(("valu", ("<", tmp1, idx_v_regstore[i], input_n_v_const)))
                body.append(("valu", ("*", idx_v_regstore[i], idx_v_regstore[i], tmp1)))
                body.append(("debug", ("vcompare", idx_v_regstore[i], vcompare_keys(round, i, "wrapped_idx"))))

                self.pool.free(tmp1)
                self.pool.free(tmp2)
                self.pool.free(tmp_node_val)
                self.pool.free(tmp_addr)
                self.pool.free(tmp_idx_move)
                i += VLEN

            while i < batch_size:
                # Scalar scratch registers
                tmp1 = self.pool.alloc()
                tmp2 = self.pool.alloc()
                tmp_node_val = self.pool.alloc()
                tmp_addr = self.pool.alloc()
                tmp_idx_move = self.pool.alloc()

                assert idx_regstore[i] != -1, f"Unexpected scalar access {i}"
                assert val_regstore[i] != -1, f"Unexpected scalar access {i}"

                # idx = mem[inp_indices_p + i]
                body.append(("debug", ("compare", idx_regstore[i], (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("debug", ("compare", val_regstore[i], (round, i, "val"))))

                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, input_reg["forest_values_p"], idx_regstore[i])))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))

                # val = myhash(val ^ node_val)
                # mem[inp_values_p + i] = val
                body.append(("alu", ("^", val_regstore[i], val_regstore[i], tmp_node_val)))
                body.extend(self.build_hash(val_regstore[i], tmp1, tmp2, round, i))
                body.append(("debug", ("compare", val_regstore[i], (round, i, "hashed_val"))))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                # idx --> 2*idx + (val&1) + 1
                body.append(("alu", ("<<", idx_regstore[i], idx_regstore[i], one_const)))
                body.append(("alu", ("&", tmp_idx_move, val_regstore[i], one_const)))
                body.append(("alu", ("+", tmp_idx_move, tmp_idx_move, one_const)))
                body.append(("alu", ("+", idx_regstore[i], idx_regstore[i], tmp_idx_move)))
                body.append(("debug", ("compare", idx_regstore[i], (round, i, "next_idx"))))

                # idx = 0 if idx >= n_nodes else idx
                # ---> idx = cond*idx, cond = idx < n_nodes
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("<", tmp1, idx_regstore[i], input_reg["n_nodes"])))
                body.append(("alu", ("*", idx_regstore[i], idx_regstore[i], tmp1)))
                body.append(("debug", ("compare", idx_regstore[i], (round, i, "wrapped_idx"))))

                self.pool.free(tmp1)
                self.pool.free(tmp2)
                self.pool.free(tmp_node_val)
                self.pool.free(tmp_addr)
                self.pool.free(tmp_idx_move)
                i += 1

        # Write into memory
        i = 0
        tmp_addr = self.pool.alloc()
        tmp_addr2 = self.pool.alloc()
        body.append(("alu", ("+", tmp_addr, zero_const, input_reg["inp_indices_p"])))
        body.append(("alu", ("+", tmp_addr2, zero_const, input_reg["inp_values_p"])))

        while i + VLEN - 1 < SIMD_LIMIT:
            assert idx_v_regstore[i] != -1, f"Unaligned access {i}"
            assert val_v_regstore[i] != -1, f"Unaligned access {i}"
            tmp_offset = self.pool.alloc()
            body.append(("store", ("vstore", tmp_addr, idx_v_regstore[i])))
            body.append(("store", ("vstore", tmp_addr2, val_v_regstore[i])))

            i += VLEN
            body.append(("alu", ("+", tmp_addr, tmp_addr, vlen_const)))
            body.append(("alu", ("+", tmp_addr2, tmp_addr2, vlen_const)))

        while i < batch_size:
            assert idx_regstore[i] != -1, f"Unexpected scalar access {i}"
            assert val_regstore[i] != -1, f"Unexpected scalar access {i}"
            body.append(("store", ("store", tmp_addr, idx_regstore[i])))
            body.append(("store", ("store", tmp_addr2, val_regstore[i])))

            i += 1
            body.append(("alu", ("+", tmp_addr, tmp_addr, one_const)))
            body.append(("alu", ("+", tmp_addr2, tmp_addr2, one_const)))
        self.pool.free(tmp_addr)
        self.pool.free(tmp_addr2)

        self.instrs.extend(Scheduler(self.pool).build(body, vliw=True))
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256, prints=False)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
