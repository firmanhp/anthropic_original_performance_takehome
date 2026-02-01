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
        assert val in self.const_map, f"Unreserved const: {val}"
        return self.const_map[val]

    def scratch_v_const(self, val):
        assert val in self.v_const_map, f"Unreserved v_const: {val}"
        return self.v_const_map[val]

    def preload_const(self, val):
        if val in self.const_map:
            return
        addr = self.pool.alloc(f"CONST[0x{val:X}]")
        self.const_map[val] = addr
        self.scheduler.add("load", ("const", addr, val))

    def preload_v_const(self, val):
        if val in self.v_const_map:
            return []
        addr = self.pool.alloc(f"V_CONST[0x{val:X}]", length=VLEN)
        self.v_const_map[val] = addr
        self.preload_const(val)
        self.scheduler.add("valu", ("vbroadcast", addr, self.scratch_const(val)))

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

    def schedule_work_vlen_valu(self, round, i, **vec_regs):
        assert 'idx_v_regstore' in vec_regs
        assert 'val_v_regstore' in vec_regs
        assert 'input_forest_p_v_const' in vec_regs
        assert 'input_n_v_const' in vec_regs
        assert 'one_v_const' in vec_regs
        assert 'two_v_const' in vec_regs

        tmp_addr = self.pool.alloc(length=VLEN)
        tmp1 = self.pool.alloc(length=VLEN)
        tmp2 = self.pool.alloc(length=VLEN)
        tmp_node_val = self.pool.alloc(length=VLEN)
        tmp_idx_move = self.pool.alloc(length=VLEN)
        
        idx_v_regstore = vec_regs['idx_v_regstore']
        val_v_regstore = vec_regs['val_v_regstore']
        input_forest_p_v_const = vec_regs['input_forest_p_v_const']
        input_n_v_const = vec_regs['input_n_v_const']
        one_v_const = vec_regs['one_v_const']
        two_v_const = vec_regs['two_v_const']
        
        # idx = mem[inp_indices_p + i]
        self.scheduler.add("debug", ("vcompare", idx_v_regstore, vcompare_keys(round, i, "idx")))
        # val = mem[inp_values_p + i]
        self.scheduler.add("debug", ("vcompare", val_v_regstore, vcompare_keys(round, i, "val")))

        # node_val = mem[forest_values_p + idx]
        self.scheduler.add("valu", ("+", tmp_addr, input_forest_p_v_const, idx_v_regstore))
        for offset in range(VLEN):
            self.scheduler.add("load", ("load_offset", tmp_node_val, tmp_addr, offset))
        self.scheduler.add("debug", ("vcompare", tmp_node_val, vcompare_keys(round, i, "node_val")))

        # val = myhash(val ^ node_val)
        # mem[inp_values_p + i] = val
        self.scheduler.add("valu", ("^", val_v_regstore, val_v_regstore, tmp_node_val))
        self.v_hash(val_v_regstore, tmp1, tmp2, round, i)
        self.scheduler.add("debug", ("vcompare", val_v_regstore, vcompare_keys(round, i, "hashed_val")))

        # idx = 2*idx + (1 if val % 2 == 0 else 2)
        # idx --> 2*idx + (val&1) + 1
        self.scheduler.add("valu", ("multiply_add", idx_v_regstore, idx_v_regstore, two_v_const, one_v_const))
        self.scheduler.add("valu", ("&", tmp_idx_move, val_v_regstore, one_v_const))
        self.scheduler.add("valu", ("+", idx_v_regstore, idx_v_regstore, tmp_idx_move))
        self.scheduler.add("debug", ("vcompare", idx_v_regstore, vcompare_keys(round, i, "next_idx")))

        # idx = 0 if idx >= n_nodes else idx
        # ---> idx = cond*idx, cond = idx < n_nodes
        # mem[inp_indices_p + i] = idx
        self.scheduler.add("valu", ("<", tmp1, idx_v_regstore, input_n_v_const))
        self.scheduler.add("valu", ("*", idx_v_regstore, idx_v_regstore, tmp1))
        self.scheduler.add("debug", ("vcompare", idx_v_regstore, vcompare_keys(round, i, "wrapped_idx")))

        self.pool.free(tmp1)
        self.pool.free(tmp2)
        self.pool.free(tmp_node_val)
        self.pool.free(tmp_addr)
        self.pool.free(tmp_idx_move)

    def schedule_work_vlen_alu(self, round, i, **vec_regs):
        assert 'idx_v_regstore' in vec_regs
        assert 'val_v_regstore' in vec_regs
        assert 'input_forest_p_v_const' in vec_regs
        assert 'input_n_v_const' in vec_regs
        assert 'one_v_const' in vec_regs
        assert 'two_v_const' in vec_regs

        v_tmp_addr = self.pool.alloc(length=VLEN)
        v_tmp1 = self.pool.alloc(length=VLEN)
        v_tmp2 = self.pool.alloc(length=VLEN)
        v_tmp_node_val = self.pool.alloc(length=VLEN)
        v_tmp_idx_move = self.pool.alloc(length=VLEN)
        for offset in range(VLEN):
            tmp1 = v_tmp1 + offset
            tmp2 = v_tmp2 + offset
            tmp_node_val = v_tmp_node_val + offset
            tmp_addr = v_tmp_addr + offset
            tmp_idx_move = v_tmp_idx_move + offset
            idx_v_regstore = vec_regs['idx_v_regstore'] + offset
            val_v_regstore = vec_regs['val_v_regstore'] + offset
            input_forest_p_v_const = vec_regs['input_forest_p_v_const'] + offset
            input_n_v_const = vec_regs['input_n_v_const'] + offset
            one_v_const = vec_regs['one_v_const'] + offset
            two_v_const = vec_regs['two_v_const'] + offset

            # idx = mem[inp_indices_p + i]
            self.scheduler.add("debug", ("compare", idx_v_regstore, (round, i+offset, "idx")))
            # val = mem[inp_values_p + i]
            self.scheduler.add("debug", ("compare", val_v_regstore, (round, i+offset, "val")))

            # node_val = mem[forest_values_p + idx]
            self.scheduler.add("alu", ("+", tmp_addr, input_forest_p_v_const, idx_v_regstore))
            self.scheduler.add("load", ("load", tmp_node_val, tmp_addr))
            self.scheduler.add("debug", ("compare", tmp_node_val, (round, i+offset, "node_val")))

            # val = myhash(val ^ node_val)
            # mem[inp_values_p + i] = val
            self.scheduler.add("alu", ("^", val_v_regstore, val_v_regstore, tmp_node_val))
            self.hash(val_v_regstore, tmp1, tmp2, round, i+offset)
            self.scheduler.add("debug", ("compare", val_v_regstore, (round, i+offset, "hashed_val")))

            # idx = 2*idx + (1 if val % 2 == 0 else 2)
            # idx --> 2*idx + (val&1) + 1
            self.scheduler.add("alu", ("<<", idx_v_regstore, idx_v_regstore, one_v_const))
            self.scheduler.add("alu", ("&", tmp_idx_move, val_v_regstore, one_v_const))
            self.scheduler.add("alu", ("+", tmp_idx_move, tmp_idx_move, one_v_const))
            self.scheduler.add("alu", ("+", idx_v_regstore, idx_v_regstore, tmp_idx_move))
            self.scheduler.add("debug", ("compare", idx_v_regstore, (round, i+offset, "next_idx")))

            # idx = 0 if idx >= n_nodes else idx
            # ---> idx = cond*idx, cond = idx < n_nodes
            # mem[inp_indices_p + i] = idx
            self.scheduler.add("alu", ("<", tmp1, idx_v_regstore, input_n_v_const))
            self.scheduler.add("alu", ("*", idx_v_regstore, idx_v_regstore, tmp1))
            self.scheduler.add("debug", ("compare", idx_v_regstore, (round, i+offset, "wrapped_idx")))
        self.pool.free(v_tmp1)
        self.pool.free(v_tmp2)
        self.pool.free(v_tmp_node_val)
        self.pool.free(v_tmp_addr)
        self.pool.free(v_tmp_idx_move)

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
        self.preload_const(VLEN)

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
        self.scheduler.add("alu", ("+", tmp_addr, zero_const, input_reg["inp_indices_p"]))
        self.scheduler.add("alu", ("+", tmp_addr2, zero_const, input_reg["inp_values_p"]))

        i = 0
        while i + VLEN - 1 < SIMD_LIMIT:
            idx_v_regstore[i] = self.pool.alloc(f"idx[{i}:{i+VLEN-1}]", length=VLEN)
            val_v_regstore[i] = self.pool.alloc(f"val[{i}:{i+VLEN-1}]", length=VLEN)
            self.scheduler.add("load", ("vload", idx_v_regstore[i], tmp_addr))
            self.scheduler.add("load", ("vload", val_v_regstore[i], tmp_addr2))

            i += VLEN
            self.scheduler.add("alu", ("+", tmp_addr, tmp_addr, vlen_const))
            self.scheduler.add("alu", ("+", tmp_addr2, tmp_addr2, vlen_const))

        while i < batch_size:
            idx_regstore[i] = self.pool.alloc(f"idx[{i}]")
            val_regstore[i] = self.pool.alloc(f"val[{i}]")
            self.scheduler.add("load", ("load", idx_regstore[i], tmp_addr))
            self.scheduler.add("load", ("load", val_regstore[i], tmp_addr2))

            i += 1
            self.scheduler.add("alu", ("+", tmp_addr, tmp_addr, one_const))
            self.scheduler.add("alu", ("+", tmp_addr2, tmp_addr2, one_const))
        self.pool.free(tmp_addr)
        self.pool.free(tmp_addr2)

        for round in range(rounds):
            i = 0
            while i + VLEN - 1 < SIMD_LIMIT:
                # Vector scratch registers
                assert idx_v_regstore[i] != -1, f"Unaligned access {i}"
                assert val_v_regstore[i] != -1, f"Unaligned access {i}"

                sched_pre_snap, pool_pre_snap = self.scheduler.snapshot(), self.pool.snapshot()
                self.schedule_work_vlen_valu(round, i,
                                             idx_v_regstore = idx_v_regstore[i],
                                             val_v_regstore = val_v_regstore[i],
                                             input_forest_p_v_const = input_forest_p_v_const,
                                             input_n_v_const = input_n_v_const,
                                             one_v_const = one_v_const,
                                             two_v_const = two_v_const)
                sched_valu_snap, pool_valu_snap = self.scheduler.snapshot(), self.pool.snapshot()
                cycles_valu = len(self.scheduler.program)
                self.scheduler.load_snapshot(sched_pre_snap)
                self.pool.load_snapshot(pool_pre_snap)
                self.schedule_work_vlen_alu(round, i,
                                            idx_v_regstore = idx_v_regstore[i],
                                            val_v_regstore = val_v_regstore[i],
                                            input_forest_p_v_const = input_forest_p_v_const,
                                            input_n_v_const = input_n_v_const,
                                            one_v_const = one_v_const,
                                            two_v_const = two_v_const)
                cycles_alu = len(self.scheduler.program)
                print(f"firmanhp round {round} {i}:{i+VLEN-1} (valu,alu) ({cycles_valu}, {cycles_alu})")
                if cycles_valu < cycles_alu:
                    print(f"firmanhp round {round} {i}:{i+VLEN-1} take valu")
                    self.scheduler.load_snapshot(sched_valu_snap)
                    self.pool.load_snapshot(pool_valu_snap)
                else:
                    print(f"firmanhp round {round} {i}:{i+VLEN-1} take alu")

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
                self.scheduler.add("debug", ("compare", idx_regstore[i], (round, i, "idx")))
                # val = mem[inp_values_p + i]
                self.scheduler.add("debug", ("compare", val_regstore[i], (round, i, "val")))

                # node_val = mem[forest_values_p + idx]
                self.scheduler.add("alu", ("+", tmp_addr, input_reg["forest_values_p"], idx_regstore[i]))
                self.scheduler.add("load", ("load", tmp_node_val, tmp_addr))
                self.scheduler.add("debug", ("compare", tmp_node_val, (round, i, "node_val")))

                # val = myhash(val ^ node_val)
                # mem[inp_values_p + i] = val
                self.scheduler.add("alu", ("^", val_regstore[i], val_regstore[i], tmp_node_val))
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
                self.pool.free(tmp_node_val)
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
            assert idx_v_regstore[i] != -1, f"Unaligned access {i}"
            assert val_v_regstore[i] != -1, f"Unaligned access {i}"
            tmp_offset = self.pool.alloc()
            self.scheduler.add("store", ("vstore", tmp_addr, idx_v_regstore[i]))
            self.scheduler.add("store", ("vstore", tmp_addr2, val_v_regstore[i]))

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
