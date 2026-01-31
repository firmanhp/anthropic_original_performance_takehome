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

from scheduler import (ScratchRegPool)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.pool = ScratchRegPool()
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.pool.scratch_debug)

    def append_alu(self, cycles: [Instruction], single_instruction):
        if not cycles:
            cycles.append({'alu': [single_instruction]})
            return
        
        last_cyc = cycles[-1]
        if 'alu' not in last_cyc or len(last_cyc['alu']) >= SLOT_LIMITS['alu']:
            cycles.append({'alu': [single_instruction]})
            return
        
        # op dest a1 a2
        # dest <- last write, last read
        # a1 <- last write, last read
        # a2 <- last write, last read
        reads = set()
        writes = set()
        for uop in last_cyc['alu']:
            op, dest, a1, a2 = uop
            writes.add(dest)
            reads.add(dest)
            reads.add(a1)
            reads.add(a2)
        
        op, dest, a1, a2 = single_instruction
        if dest not in reads and a1 not in writes and a2 not in writes:
            cycles[-1]["alu"].append(single_instruction)
        else:
            cycles.append({'alu': [single_instruction]})

    def build(self, single_instructions: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        cycles = []
        for engine, instruction in single_instructions:
            # consider alu
            if engine == 'alu':
                self.append_alu(cycles, instruction)
                continue
            
            cycles.append({engine: [instruction]})

        return cycles

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def scratch_const(self, val, name=None):
        assert val in self.const_map, f"Unreserved const: {val}"
        return self.const_map[val]

    def preload_const(self, val, name=None):
        if val in self.const_map:
            return
        addr = self.pool.alloc(name)
        self.add("load", ("const", addr, val))
        self.const_map[val] = addr

    def build_hash(self, val_hash_reg, tmp1, tmp2, round, i):
        slots = []

        # alu(self, core, op, dest, a1, a2)
        slots.append(("alu", ("+", tmp1, val_hash_reg, self.scratch_const(0x7ED55D16))))
        slots.append(("alu", ("<<", tmp2, val_hash_reg, self.scratch_const(12))))
        slots.append(("alu", ("+", val_hash_reg, tmp1, tmp2)))
        slots.append(("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 0))))
        
        slots.append(("alu", ("^", tmp1, val_hash_reg, self.scratch_const(0xC761C23C))))
        slots.append(("alu", (">>", tmp2, val_hash_reg, self.scratch_const(19))))
        slots.append(("alu", ("^", val_hash_reg, tmp1, tmp2)))
        slots.append(("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 1))))

        slots.append(("alu", ("+", tmp1, val_hash_reg, self.scratch_const(0x165667B1))))
        slots.append(("alu", ("<<", tmp2, val_hash_reg, self.scratch_const(5))))
        slots.append(("alu", ("+", val_hash_reg, tmp1, tmp2)))
        slots.append(("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 2))))

        slots.append(("alu", ("+", tmp1, val_hash_reg, self.scratch_const(0xD3A2646C))))
        slots.append(("alu", ("<<", tmp2, val_hash_reg, self.scratch_const(9))))
        slots.append(("alu", ("^", val_hash_reg, tmp1, tmp2)))
        slots.append(("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 3))))

        slots.append(("alu", ("+", tmp1, val_hash_reg, self.scratch_const(0xFD7046C5))))
        slots.append(("alu", ("<<", tmp2, val_hash_reg, self.scratch_const(3))))
        slots.append(("alu", ("+", val_hash_reg, tmp1, tmp2)))
        slots.append(("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 4))))

        slots.append(("alu", ("^", tmp1, val_hash_reg, self.scratch_const(0xB55A4F09))))
        slots.append(("alu", (">>", tmp2, val_hash_reg, self.scratch_const(16))))
        slots.append(("alu", ("^", val_hash_reg, tmp1, tmp2)))
        slots.append(("debug", ("compare", val_hash_reg, (round, i, "hash_stage", 5))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
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

        # Const preloads
        self.preload_const(0x7ED55D16)
        self.preload_const(12)
        self.preload_const(0xC761C23C)
        self.preload_const(19)
        self.preload_const(0x165667B1)
        self.preload_const(5)
        self.preload_const(0xD3A2646C)
        self.preload_const(9)
        self.preload_const(0xFD7046C5)
        self.preload_const(3)
        self.preload_const(0xB55A4F09)
        self.preload_const(16)
        self.preload_const(0)
        self.preload_const(1)
        self.preload_const(2)
        for i in range(batch_size):
            self.preload_const(i)

        # Scratch space addresses

        input_reg = {}
        for i, v in enumerate(init_vars):
            input_reg[v] = self.pool.alloc(v, 1)
            self.add("load", ("load", input_reg[v], self.scratch_const(i)))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # assumption only works with small batch sizes!
        # instead of working in memory, let's put into register
        idx_regstore = []
        val_regstore = []
        tmp_addr = self.pool.alloc()
        for i in range(batch_size):
            idx_regstore.append(self.pool.alloc())
            val_regstore.append(self.pool.alloc())

            i_const = self.scratch_const(i)
            body.append(("alu", ("+", tmp_addr, input_reg["inp_indices_p"], i_const)))
            body.append(("load", ("load", idx_regstore[i], tmp_addr)))

            body.append(("alu", ("+", tmp_addr, input_reg["inp_values_p"], i_const)))
            body.append(("load", ("load", val_regstore[i], tmp_addr)))
        self.pool.free(tmp_addr)

        for round in range(rounds):
            for i in range(batch_size):
                # Scalar scratch registers
                tmp1 = self.pool.alloc()
                tmp2 = self.pool.alloc()
                tmp3 = self.pool.alloc()
                tmp_idx = self.pool.alloc()
                tmp_val = self.pool.alloc()
                tmp_node_val = self.pool.alloc()
                tmp_addr = self.pool.alloc()
                tmp_idx_move = self.pool.alloc()

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
                body.append(("alu", ("<<", tmp_idx, idx_regstore[i], one_const)))
                body.append(("alu", ("&", tmp_idx_move, val_regstore[i], one_const)))
                body.append(("alu", ("+", tmp_idx_move, tmp_idx_move, one_const)))
                body.append(("alu", ("+", idx_regstore[i], tmp_idx, tmp_idx_move)))

                body.append(("debug", ("compare", idx_regstore[i], (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("<", tmp1, idx_regstore[i], input_reg["n_nodes"])))
                body.append(("flow", ("select", idx_regstore[i], tmp1, idx_regstore[i], zero_const)))
                body.append(("debug", ("compare", idx_regstore[i], (round, i, "wrapped_idx"))))

                self.pool.free(tmp1)
                self.pool.free(tmp2)
                self.pool.free(tmp3)
                self.pool.free(tmp_idx)
                self.pool.free(tmp_val)
                self.pool.free(tmp_node_val)
                self.pool.free(tmp_addr)
                self.pool.free(tmp_idx_move)

        # Write into memory
        tmp_addr = self.pool.alloc()
        for i in range(batch_size):
            idx_regstore.append(self.pool.alloc())
            val_regstore.append(self.pool.alloc())

            i_const = self.scratch_const(i)
            body.append(("alu", ("+", tmp_addr, input_reg["inp_indices_p"], i_const)))
            body.append(("store", ("store", tmp_addr, idx_regstore[i])))

            body.append(("alu", ("+", tmp_addr, input_reg["inp_values_p"], i_const)))
            body.append(("store", ("store", tmp_addr, val_regstore[i])))
        self.pool.free(tmp_addr)

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
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
        do_kernel_test(10, 16, 256)


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
