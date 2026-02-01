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
        self.scheduler = Scheduler(self.pool, vliw=True)

    def debug_info(self):
        return DebugInfo(scratch_map=self.pool.scratch_debug)

    def scratch_const(self, val):
        if val in self.v_const_map:
            return self.v_const_map[val]

        assert val in self.const_map, f"Unreserved const: {val}"
        return self.const_map[val]

    def scratch_v_const(self, val):
        assert val in self.v_const_map, f"Unreserved v_const: {val}"
        return self.v_const_map[val]

    def preload_const(self, val):
        if val in self.const_map:
            return
        if val in self.v_const_map:
            return
        addr = self.pool.alloc(f"CONST[0x{val:X}]")
        self.const_map[val] = addr
        self.scheduler.add("load", ("const", addr, val))

    def preload_v_const(self, val):
        if val in self.v_const_map:
            return []
        addr = self.pool.alloc(f"V_CONST[0x{val:X}]", length=VLEN)
        self.v_const_map[val] = addr

        tmp = self.pool.alloc()
        self.scheduler.add("load", ("const", tmp, val))
        self.scheduler.add("valu", ("vbroadcast", addr, tmp))
        self.pool.free(tmp)

    #Robert Jenkins, jenkins32: https://gist.github.com/badboy/6267743#robert-jenkins-32-bit-integer-hash-function
    def hash(self, val_hash_reg, tmp1, tmp2, round, i):
        # alu(self, core, op, dest, a1, a2)
        self.scheduler.add("alu", ("+", tmp1, val_hash_reg, self.scratch_const(0x7ED55D16)))
        self.scheduler.add("alu", ("<<", tmp2, val_hash_reg, self.scratch_const(12)))
        self.scheduler.add("alu", ("+", val_hash_reg, tmp1, tmp2))
        self.scheduler.add("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 0)))

        self.scheduler.add("alu", ("^", tmp1, val_hash_reg, self.scratch_const(0xC761C23C)))
        self.scheduler.add("alu", (">>", tmp2, val_hash_reg, self.scratch_const(19)))
        self.scheduler.add("alu", ("^", val_hash_reg, tmp1, tmp2))
        self.scheduler.add("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 1)))

        self.scheduler.add("alu", ("+", tmp1, val_hash_reg, self.scratch_const(0x165667B1)))
        self.scheduler.add("alu", ("<<", tmp2, val_hash_reg, self.scratch_const(5)))
        self.scheduler.add("alu", ("+", val_hash_reg, tmp1, tmp2))
        self.scheduler.add("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 2)))

        self.scheduler.add("alu", ("+", tmp1, val_hash_reg, self.scratch_const(0xD3A2646C)))
        self.scheduler.add("alu", ("<<", tmp2, val_hash_reg, self.scratch_const(9)))
        self.scheduler.add("alu", ("^", val_hash_reg, tmp1, tmp2))
        self.scheduler.add("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 3)))

        self.scheduler.add("alu", ("+", tmp1, val_hash_reg, self.scratch_const(0xFD7046C5)))
        self.scheduler.add("alu", ("<<", tmp2, val_hash_reg, self.scratch_const(3)))
        self.scheduler.add("alu", ("+", val_hash_reg, tmp1, tmp2))
        self.scheduler.add("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 4)))

        self.scheduler.add("alu", ("^", tmp1, val_hash_reg, self.scratch_const(0xB55A4F09)))
        self.scheduler.add("alu", (">>", tmp2, val_hash_reg, self.scratch_const(16)))
        self.scheduler.add("alu", ("^", val_hash_reg, tmp1, tmp2))
        self.scheduler.add("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 5)))

    def v_hash(self, reg, tmp1, tmp2, round, i):
        # valu has "multiply_add", we should exploit this
        # alu(self, core, op, dest, a1, a2)
        self.scheduler.add("valu", ("+", tmp1, reg, self.scratch_v_const(0x7ED55D16)))
        self.scheduler.add("valu", ("multiply_add", reg, reg, self.scratch_v_const(2**12), tmp1))
        # self.scheduler.add("valu", ("<<", tmp2, reg, self.scratch_v_const(12)))
        # self.scheduler.add("valu", ("+", reg, tmp1, tmp2))
        self.scheduler.add("debug", ("vcompare", reg, vcompare_keys(round, i, "hash_stage", 0)))

        self.scheduler.add("valu", ("^", tmp1, reg, self.scratch_v_const(0xC761C23C)))
        self.scheduler.add("valu", (">>", tmp2, reg, self.scratch_v_const(19)))
        self.scheduler.add("valu", ("^", reg, tmp1, tmp2))
        self.scheduler.add("debug", ("vcompare", reg, vcompare_keys(round, i, "hash_stage", 1)))

        self.scheduler.add("valu", ("+", tmp1, reg, self.scratch_v_const(0x165667B1)))
        self.scheduler.add("valu", ("multiply_add", reg, reg, self.scratch_v_const(2**5), tmp1))
        # self.scheduler.add("valu", ("<<", tmp2, reg, self.scratch_v_const(5)))) #parity = 
        # self.scheduler.add("valu", ("+", reg, tmp1, tmp2))
        self.scheduler.add("debug", ("vcompare", reg, vcompare_keys(round, i, "hash_stage", 2)))

        self.scheduler.add("valu", ("+", tmp1, reg, self.scratch_v_const(0xD3A2646C)))
        self.scheduler.add("valu", ("<<", tmp2, reg, self.scratch_v_const(9)))
        self.scheduler.add("valu", ("^", reg, tmp1, tmp2))
        self.scheduler.add("debug", ("vcompare", reg, vcompare_keys(round, i, "hash_stage", 3)))

        self.scheduler.add("valu", ("+", tmp1, reg, self.scratch_v_const(0xFD7046C5)))
        self.scheduler.add("valu", ("multiply_add", reg, reg, self.scratch_v_const(2**3), tmp1))
        # self.scheduler.add("valu", ("<<", tmp2, reg, self.scratch_v_const(3)))
        # self.scheduler.add("valu", ("+", reg, tmp1, tmp2))
        self.scheduler.add("debug", ("vcompare", reg, vcompare_keys(round, i, "hash_stage", 4)))

        self.scheduler.add("valu", ("^", tmp1, reg, self.scratch_v_const(0xB55A4F09)))
        self.scheduler.add("valu", (">>", tmp2, reg, self.scratch_v_const(16)))
        self.scheduler.add("valu", ("^", reg, tmp1, tmp2))
        self.scheduler.add("debug", ("vcompare", reg, vcompare_keys(round, i, "hash_stage", 5)))

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        
        # this code only works with small batch sizes! bcs it is not smart enough ;)
        # instead of working in memory, let's put into register
        idx_reg = self.pool.alloc("idx", length=batch_size)
        val_reg = self.pool.alloc("val", length=batch_size)
        node_val_reg = self.pool.alloc("NODE_VAL", length=batch_size)
        precache_max_height = 9
        node_val_cache = self.pool.alloc("NODE_VAL_CACHE", length=2**precache_max_height)

        # Const preloads
        self.preload_const(12)
        self.preload_const(5)
        self.preload_const(3)
        self.preload_const(VLEN)
        for i in range(0, precache_max_height+1):
            self.preload_const(2**i - 1)

        self.preload_v_const(0x7ED55D16)
        self.preload_v_const(2**12) # useful for multiply_add
        self.preload_v_const(0xC761C23C)
        self.preload_v_const(19)
        self.preload_v_const(0x165667B1)
        self.preload_v_const(2**5) # useful for multiply_add
        self.preload_v_const(0xD3A2646C)
        self.preload_v_const(9)
        self.preload_v_const(0xFD7046C5)
        self.preload_v_const(2**3) # useful for multiply_add
        self.preload_v_const(0xB55A4F09)
        self.preload_v_const(16)
        self.preload_v_const(0)
        self.preload_v_const(1)
        self.preload_v_const(2)

        # Scratch space addresses

        input_reg = {}
        for i, v in enumerate(init_vars):
            # consider: replacing const register with tmp <- add_imm ZERO #i
            self.preload_const(i);
            input_reg[v] = self.pool.alloc(v, 1)
            self.scheduler.add("load", ("load", input_reg[v], self.scratch_const(i)))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        vlen_const = self.scratch_const(VLEN)

        zero_v_const = self.scratch_v_const(0)
        one_v_const = self.scratch_v_const(1)
        two_v_const = self.scratch_v_const(2)

        input_forest_p_v_const = self.pool.alloc(name="V_CONST[FOREST_P]", length=VLEN)
        self.scheduler.add("valu", ("vbroadcast", input_forest_p_v_const, input_reg["forest_values_p"]))
        input_n_v_const = self.pool.alloc(name="V_CONST[N]", length=VLEN)
        self.scheduler.add("valu", ("vbroadcast", input_n_v_const, input_reg["n_nodes"]))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.scheduler.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.scheduler.add("debug", ("comment", "Starting loop"))

        idx_regstore = [idx_reg+i for i in range(batch_size)]
        val_regstore = [val_reg+i for i in range(batch_size)]
        node_val_regstore = [node_val_reg+i for i in range(batch_size)]
        node_val_cache_regstore = [node_val_cache+i for i in range(2**precache_max_height)]
        SIMD_LIMIT = batch_size

        # so much things to prepare for simd, ew
        tmp_addr = self.pool.alloc()
        tmp_addr2 = self.pool.alloc()
        self.scheduler.add("alu", ("+", tmp_addr, zero_const, input_reg["inp_indices_p"]))
        self.scheduler.add("alu", ("+", tmp_addr2, zero_const, input_reg["inp_values_p"]))

        i = 0
        while i + VLEN - 1 < SIMD_LIMIT:
            self.scheduler.add("load", ("vload", idx_regstore[i], tmp_addr))
            self.scheduler.add("load", ("vload", val_regstore[i], tmp_addr2))

            i += VLEN
            self.scheduler.add("alu", ("+", tmp_addr, tmp_addr, vlen_const))
            self.scheduler.add("alu", ("+", tmp_addr2, tmp_addr2, vlen_const))

        while i < batch_size:
            self.scheduler.add("load", ("load", idx_regstore[i], tmp_addr))
            self.scheduler.add("load", ("load", val_regstore[i], tmp_addr2))

            i += 1
            self.scheduler.add("alu", ("+", tmp_addr, tmp_addr, one_const))
            self.scheduler.add("alu", ("+", tmp_addr2, tmp_addr2, one_const))
        self.pool.free(tmp_addr)
        self.pool.free(tmp_addr2)

        
        tmp_addr = self.pool.alloc(length=VLEN)
        tmp1 = self.pool.alloc(length=VLEN)
        tmp2 = self.pool.alloc(length=VLEN)
        tmp_idx_move = self.pool.alloc(length=VLEN)
        self.pool.free(tmp_addr)
        self.pool.free(tmp1)
        self.pool.free(tmp2)
        self.pool.free(tmp_idx_move)

        tree_level = -1
        for round in range(rounds):
            tree_level += 1
            node_val_cached = tree_level <= precache_max_height
            if node_val_cached:
                # preload data into register
                le = 2**tree_level - 1
                num_nodes_h = 2**tree_level

                tmp_addr = self.pool.alloc()
                self.scheduler.add("alu", ("+", tmp_addr, input_reg["forest_values_p"], self.scratch_const(le)))
                i = 0
                while i + VLEN - 1 < num_nodes_h:
                    self.scheduler.add("load", ("vload", node_val_cache_regstore[i], tmp_addr))
                    self.scheduler.add("alu", ("+", tmp_addr, tmp_addr, self.scratch_const(VLEN)))
                    i += VLEN
                while i < num_nodes_h:
                    self.scheduler.add("load", ("load", node_val_cache_regstore[i], tmp_addr))
                    self.scheduler.add("alu", ("+", tmp_addr, tmp_addr, one_const))
                    i += 1
                self.pool.free(tmp_addr)

                for k in range(batch_size):
                    # print(f"firmanhp round{round} cache for {i}")
                    cur_idx = self.pool.alloc()
                    cond = self.pool.alloc()
                    self.scheduler.add("alu", ("+", cur_idx, zero_const, self.scratch_const(le)))
                    self.scheduler.add("alu", ("+", node_val_regstore[k], zero_const, zero_const))
                    for j in range(num_nodes_h):
                        self.scheduler.add("alu", ("==", cond, idx_regstore[k], cur_idx))
                        self.scheduler.add("alu", ("*", cond, cond, node_val_cache_regstore[j]))
                        self.scheduler.add("alu", ("+", node_val_regstore[k], node_val_regstore[k], cond))
                        self.scheduler.add("alu", ("+", cur_idx, cur_idx, one_const))

                    self.pool.free(cur_idx)
                    self.pool.free(cond)
            
            i = 0
            while i + VLEN - 1 < SIMD_LIMIT:
                # Vector scratch registers
                tmp_addr = self.pool.alloc(length=VLEN)
                tmp1 = self.pool.alloc(length=VLEN)
                tmp2 = self.pool.alloc(length=VLEN)
                tmp_idx_move = self.pool.alloc(length=VLEN)
                
                # idx = mem[inp_indices_p + i]
                self.scheduler.add("debug", ("vcompare", idx_regstore[i], vcompare_keys(round, i, "idx")))
                # val = mem[inp_values_p + i]
                self.scheduler.add("debug", ("vcompare", val_regstore[i], vcompare_keys(round, i, "val")))

                # node_val = mem[forest_values_p + idx]
                if not node_val_cached:
                    self.scheduler.add("valu", ("+", tmp_addr, input_forest_p_v_const, idx_regstore[i]))
                    for offset in range(VLEN):
                        self.scheduler.add("load", ("load_offset", node_val_regstore[i], tmp_addr, offset))
                self.scheduler.add("debug", ("vcompare", node_val_regstore[i], vcompare_keys(round, i, "node_val")))

                # val = myhash(val ^ node_val)
                # mem[inp_values_p + i] = val
                self.scheduler.add("valu", ("^", val_regstore[i], val_regstore[i], node_val_regstore[i]))
                self.v_hash(val_regstore[i], tmp1, tmp2, round, i)
                self.scheduler.add("debug", ("vcompare", val_regstore[i], vcompare_keys(round, i, "hashed_val")))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                # idx --> 2*idx + (val&1) + 1
                self.scheduler.add("valu", ("multiply_add", idx_regstore[i], idx_regstore[i], two_v_const, one_v_const))
                self.scheduler.add("valu", ("&", tmp_idx_move, val_regstore[i], one_v_const))
                self.scheduler.add("valu", ("+", idx_regstore[i], idx_regstore[i], tmp_idx_move))
                self.scheduler.add("debug", ("vcompare", idx_regstore[i], vcompare_keys(round, i, "next_idx")))

                # idx = 0 if idx >= n_nodes else idx
                # ---> idx = cond*idx, cond = idx < n_nodes
                # mem[inp_indices_p + i] = idx
                self.scheduler.add("valu", ("<", tmp1, idx_regstore[i], input_n_v_const))
                self.scheduler.add("valu", ("*", idx_regstore[i], idx_regstore[i], tmp1))
                self.scheduler.add("debug", ("vcompare", idx_regstore[i], vcompare_keys(round, i, "wrapped_idx")))

                self.pool.free(tmp1)
                self.pool.free(tmp2)
                self.pool.free(tmp_addr)
                self.pool.free(tmp_idx_move)

                i += VLEN

            while i < batch_size:
                # Scalar scratch registers
                tmp1 = self.pool.alloc()
                tmp2 = self.pool.alloc()
                tmp_addr = self.pool.alloc()
                tmp_idx_move = self.pool.alloc()

                assert idx_regstore[i] != -1, f"Unexpected scalar access {i}"
                assert val_regstore[i] != -1, f"Unexpected scalar access {i}"

                # idx = mem[inp_indices_p + i]
                self.scheduler.add("debug", ("compare", idx_regstore[i], (round, i, "idx")))
                # val = mem[inp_values_p + i]
                self.scheduler.add("debug", ("compare", val_regstore[i], (round, i, "val")))

                # node_val = mem[forest_values_p + idx]
                if not node_val_cached:
                    self.scheduler.add("alu", ("+", tmp_addr, input_reg["forest_values_p"], idx_regstore[i]))
                    self.scheduler.add("load", ("load", node_val_regstore[i], tmp_addr))
                self.scheduler.add("debug", ("compare", node_val_regstore[i], (round, i, "node_val")))

                # val = myhash(val ^ node_val)
                # mem[inp_values_p + i] = val
                self.scheduler.add("alu", ("^", val_regstore[i], val_regstore[i], node_val_regstore[i]))
                self.hash(val_regstore[i], tmp1, tmp2, round, i)
                self.scheduler.add("debug", ("compare", val_regstore[i], (round, i, "hashed_val")))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                # idx --> 2*idx + (val&1) + 1
                self.scheduler.add("alu", ("<<", idx_regstore[i], idx_regstore[i], one_const))
                self.scheduler.add("alu", ("&", tmp_idx_move, val_regstore[i], one_const))
                self.scheduler.add("alu", ("+", tmp_idx_move, tmp_idx_move, one_const))
                self.scheduler.add("alu", ("+", idx_regstore[i], idx_regstore[i], tmp_idx_move))
                self.scheduler.add("debug", ("compare", idx_regstore[i], (round, i, "next_idx")))

                # idx = 0 if idx >= n_nodes else idx
                # ---> idx = cond*idx, cond = idx < n_nodes
                # mem[inp_indices_p + i] = idx
                self.scheduler.add("alu", ("<", tmp1, idx_regstore[i], input_reg["n_nodes"]))
                self.scheduler.add("alu", ("*", idx_regstore[i], idx_regstore[i], tmp1))
                self.scheduler.add("debug", ("compare", idx_regstore[i], (round, i, "wrapped_idx")))

                self.pool.free(tmp1)
                self.pool.free(tmp2)
                self.pool.free(tmp_addr)
                self.pool.free(tmp_idx_move)
                i += 1

        # Write into memory
        i = 0
        tmp_addr = self.pool.alloc()
        tmp_addr2 = self.pool.alloc()
        self.scheduler.add("alu", ("+", tmp_addr, zero_const, input_reg["inp_indices_p"]))
        self.scheduler.add("alu", ("+", tmp_addr2, zero_const, input_reg["inp_values_p"]))

        while i + VLEN - 1 < SIMD_LIMIT:
            tmp_offset = self.pool.alloc()
            self.scheduler.add("store", ("vstore", tmp_addr, idx_regstore[i]))
            self.scheduler.add("store", ("vstore", tmp_addr2, val_regstore[i]))

            i += VLEN
            self.scheduler.add("alu", ("+", tmp_addr, tmp_addr, vlen_const))
            self.scheduler.add("alu", ("+", tmp_addr2, tmp_addr2, vlen_const))

        while i < batch_size:
            assert idx_regstore[i] != -1, f"Unexpected scalar access {i}"
            assert val_regstore[i] != -1, f"Unexpected scalar access {i}"
            self.scheduler.add("store", ("store", tmp_addr, idx_regstore[i]))
            self.scheduler.add("store", ("store", tmp_addr2, val_regstore[i]))

            i += 1
            self.scheduler.add("alu", ("+", tmp_addr, tmp_addr, one_const))
            self.scheduler.add("alu", ("+", tmp_addr2, tmp_addr2, one_const))
        self.pool.free(tmp_addr)
        self.pool.free(tmp_addr2)

        # Required to match with the yield in reference_kernel2
        self.scheduler.add("flow", ("pause",))
        self.instrs = self.scheduler.program

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
        do_kernel_test(10, 16, 256, prints=True)


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
