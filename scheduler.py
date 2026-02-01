from problem import (
    SCRATCH_SIZE,
    SLOT_LIMITS,
    Instruction,
    VLEN,
    Engine
)
from copy import deepcopy

# I didn't like their definition of "Instruction"
CycleAssignment = Instruction

class ScratchRegPool:
    def __init__(self):
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        # For scalar registers, then parent[i] = i
        # For vector registers, can access the components individually.
        # If address = k, and len = VLEN, then parent[k..k+VLEN-1] = k
        # This will be useful for the scheduler later on.
        self.scratch_parent = [i for i in range(SCRATCH_SIZE)]
        self.scratch_free = []

    def alloc(self, name=None, length=1):
        if self.scratch_ptr + length >= SCRATCH_SIZE:
            # Cannot extend more
            addr = self.__pop_from_free(length)
            assert addr is not None, f"Cannot allocate/recycle anymore for reg of len {length}"
            return addr

        addr = self.scratch_ptr
        if name is None:
            name = f"ANON[{addr}]"
        self.scratch[name] = addr
        self.scratch_debug[addr] = (name, length)
        self.scratch_parent[addr] = addr
        for i in range(length):
            # if i > 0:
            #     comp_name = f"{name}+{i}"
            #     self.scratch[comp_name] = addr + i
            #     self.scratch_debug[addr + i] = (comp_name, 1)
            self.scratch_parent[addr + i] = addr

        self.scratch_ptr += length
        return addr

    def free(self, addr):
        assert addr not in self.scratch_free, f"Double free: {addr}"
        assert self.scratch_parent[addr] == addr, f"Attempted to free component: {addr}, parent: {self.scratch_parent[addr]}"
        self.scratch_free.append(addr)

    def __pop_from_free(self, length):
        idx = None
        for (i, addr) in enumerate(self.scratch_free):
            (_, l) = self.scratch_debug[addr]
            # Can reuse this address if it fits
            if l >= length:
                idx = i
                break
        if idx is None:
            return None
        return self.scratch_free.pop(i)

    def snapshot(self):
        return {
            'scratch': deepcopy(self.scratch),
            'scratch_debug': deepcopy(self.scratch_debug),
            'scratch_ptr': self.scratch_ptr,
            'scratch_parent': deepcopy(self.scratch_parent),
            'scratch_free': deepcopy(self.scratch_free)}

    def load_snapshot(self, snapshot: dict):
        self.scratch = snapshot['scratch']
        self.scratch_debug = snapshot['scratch_debug']
        self.scratch_ptr = snapshot['scratch_ptr']
        self.scratch_parent = snapshot['scratch_parent']
        self.scratch_free = snapshot['scratch_free']


class Scheduler:
    def __init__(self, reg_pool: ScratchRegPool, vliw: bool = False):
        self.pool = reg_pool
        self.vliw = vliw
        # (last read, last write)
        self.scratch_info = [(-1, -1) for i in range(SCRATCH_SIZE)]
        # [CycleAssignment]
        self.program = []

    def add(self, engine: Engine, instruction: tuple):
        if not self.vliw:
            self.program.append({engine: [instruction]})
            return
        
        match engine:
            case "debug":
                self.__sched_debug(instruction)
            case "alu":
                self.__sched_alu(instruction)
            case "valu":
                self.__sched_valu(instruction)
            case "flow":
                self.__sched_flow(instruction)
            case "store":
                self.__sched_store(instruction)
            case "load":
                self.__sched_load(instruction)
            case _:
                raise NotImplementedError(f"Unhandled engine {engine}")

    def snapshot(self):
        return {'scratch_info': deepcopy(self.scratch_info),
                'program': deepcopy(self.program)}

    def load_snapshot(self, snapshot):
        self.scratch_info = snapshot['scratch_info']
        self.program = snapshot['program']
    
        
    def __sched_alu(self, instruction: tuple):
        _, dest, a1, a2 = instruction
        dest = self.pool.scratch_parent[dest]
        a1 = self.pool.scratch_parent[a1]
        a2 = self.pool.scratch_parent[a2]
        dest_read, dest_write = self.scratch_info[dest]
        a1_read, a1_write = self.scratch_info[a1]
        a2_read, a2_write = self.scratch_info[a2]

        best_cyc = max(
            # write, must happen after read
            dest_read + 1, dest_write + 1,
            # read must happen after write
            a1_write + 1,
            a2_write + 1)

        pick_cyc = self.__sched('alu', instruction, best_cyc)

        assert pick_cyc > dest_write
        self.scratch_info[dest] = (dest_read, pick_cyc)
        if a1 != dest:
            self.scratch_info[a1] = (pick_cyc, a1_write)
        if a2 != dest:
            self.scratch_info[a2] = (pick_cyc, a2_write)

    def __sched_valu(self, instruction: tuple):
        best_cyc = None
        on_pick_cyc_fn = None
        match instruction:
            case ("vbroadcast", dest, src):
                dest = self.pool.scratch_parent[dest]
                src = self.pool.scratch_parent[src]
                dest_read, dest_write = self.scratch_info[dest]
                src_read, src_write = self.scratch_info[src]
                best_cyc = max(
                    # write, must happen after read
                    dest_read + 1, dest_write + 1,
                    # read must happen after write
                    src_write + 1
                )
                def on_pick_cyc(cyc):
                    self.scratch_info[dest] = (dest_read, cyc)
                    if src != dest:
                        self.scratch_info[src] = (cyc, src_write)
                on_pick_cyc_fn = on_pick_cyc
            case ("multiply_add", dest, a, b, c):
                dest = self.pool.scratch_parent[dest]
                a = self.pool.scratch_parent[a]
                b = self.pool.scratch_parent[b]
                c = self.pool.scratch_parent[c]
                dest_read, dest_write = self.scratch_info[dest]
                a_read, a_write = self.scratch_info[a]
                b_read, b_write = self.scratch_info[b]
                c_read, c_write = self.scratch_info[c]
                best_cyc = max(
                    # write, must happen after read
                    dest_read + 1, dest_write + 1,
                    # read must happen after write
                    a_write + 1, b_write + 1, c_write + 1
                )
                def on_pick_cyc(cyc):
                    self.scratch_info[dest] = (dest_read, cyc)
                    if a != dest:
                        self.scratch_info[a] = (cyc, a_write)
                    if b != dest:
                        self.scratch_info[b] = (cyc, b_write)
                    if c != dest:
                        self.scratch_info[c] = (cyc, c_write)
                on_pick_cyc_fn = on_pick_cyc
            case (op, dest, a, b):
                dest = self.pool.scratch_parent[dest]
                a = self.pool.scratch_parent[a]
                b = self.pool.scratch_parent[b]
                dest_read, dest_write = self.scratch_info[dest]
                a_read, a_write = self.scratch_info[a]
                b_read, b_write = self.scratch_info[b]
                best_cyc = max(
                    # write, must happen after read
                    dest_read + 1, dest_write + 1,
                    # read must happen after write
                    a_write + 1, b_write + 1
                )
                def on_pick_cyc(cyc):
                    self.scratch_info[dest] = (dest_read, cyc)
                    if a != dest:
                        self.scratch_info[a] = (cyc, a_write)
                    if b != dest:
                        self.scratch_info[b] = (cyc, b_write)
                on_pick_cyc_fn = on_pick_cyc
            case _:
                raise NotImplementedError(f"Unhandled valu {instruction}")
        assert best_cyc is not None
        assert on_pick_cyc_fn is not None

        pick_cyc = self.__sched('valu', instruction, best_cyc)
        on_pick_cyc_fn(pick_cyc)

    def __sched_flow(self, instruction: tuple):
        best_cyc = None
        # Helper lambda to update
        on_pick_cyc_fn = None
        match instruction:
            case ("select", dest, cond, a, b):
                dest = self.pool.scratch_parent[dest]
                cond = self.pool.scratch_parent[cond]
                a = self.pool.scratch_parent[a]
                b = self.pool.scratch_parent[b]
                dest_r, dest_w = self.scratch_info[dest]
                cond_r, cond_w = self.scratch_info[cond]
                a_r, a_w = self.scratch_info[a]
                b_r, b_w = self.scratch_info[b]

                best_cyc = max(
                    # write must happen after read
                    dest_r + 1, dest_w + 1,
                    # read must happen after write
                    cond_w + 1,
                    a_w + 1,
                    b_w + 1
                )
                def on_pick_cyc(cyc):
                    self.scratch_info[dest] = (dest_r, cyc)
                    if cond != dest:
                        self.scratch_info[cond] = (cyc, cond_w)
                    if a != dest:
                        self.scratch_info[a] = (cyc, a_w)
                    if b != dest:
                        self.scratch_info[b] = (cyc, b_w)
                on_pick_cyc_fn = on_pick_cyc

            case ("add_imm", dest, a, imm):
                dest = self.pool.scratch_parent[dest]
                a = self.pool.scratch_parent[a]
                dest_r, dest_w = self.scratch_info[dest]
                a_r, a_w = self.scratch_info[a]

                best_cyc = max(
                    # write must happen after read
                    dest_r + 1, dest_w + 1,
                    a_w + 1,
                )
                def on_pick_cyc(cyc):
                    self.scratch_info[dest] = (dest_r, cyc)
                    if a != dest:
                        self.scratch_info[a] = (cyc, a_w)
                on_pick_cyc_fn = on_pick_cyc

            case ("pause",):
                best_cyc = len(self.program) - 1
                def on_pick_cyc(cyc):
                    pass
                on_pick_cyc_fn = on_pick_cyc

            case _:
                raise NotImplementedError(f"Unhandled flow {instruction}")
        assert best_cyc is not None
        assert on_pick_cyc_fn is not None

        pick_cyc = self.__sched('flow', instruction, best_cyc)
        on_pick_cyc_fn(pick_cyc)

    def __sched_load(self, instruction: tuple):
        best_cyc = None
        on_pick_cyc_fn = None
        match instruction:
            case ("load", dest, addr):
                dest = self.pool.scratch_parent[dest]
                addr = self.pool.scratch_parent[addr]
                dest_r, dest_w = self.scratch_info[dest]
                addr_r, addr_w = self.scratch_info[addr]

                best_cyc = max(
                    # write must happen after read
                    dest_r + 1, dest_w + 1,
                    # read must happen after write
                    addr_w + 1
                )
                def on_pick_cyc(cyc):
                    self.scratch_info[dest] = (dest_r, cyc)
                    if addr != dest:
                        self.scratch_info[addr] = (cyc, addr_w)
                on_pick_cyc_fn = on_pick_cyc
            case ("load_offset", dest, addr, offset):
                dest = self.pool.scratch_parent[dest]
                addr = self.pool.scratch_parent[addr]
                dest_r, dest_w = self.scratch_info[dest]
                addr_r, addr_w = self.scratch_info[addr]

                best_cyc = max(
                    # write must happen after read
                    dest_r + 1, dest_w + 1,
                    # read must happen after write
                    addr_w + 1
                )
                def on_pick_cyc(cyc):
                    self.scratch_info[dest] = (dest_r, cyc)
                    if addr != dest:
                        self.scratch_info[addr] = (cyc, addr_w)
                on_pick_cyc_fn = on_pick_cyc
            case ("vload", dest, addr):
                dest = self.pool.scratch_parent[dest]
                addr = self.pool.scratch_parent[addr]
                dest_r, dest_w = self.scratch_info[dest]
                addr_r, addr_w = self.scratch_info[addr]

                best_cyc = max(
                    # write must happen after read
                    dest_r + 1, dest_w + 1,
                    # read must happen after write
                    addr_w + 1
                )
                def on_pick_cyc(cyc):
                    self.scratch_info[dest] = (dest_r, cyc)
                    if addr != dest:
                        self.scratch_info[addr] = (cyc, addr_w)
                on_pick_cyc_fn = on_pick_cyc
            case ("const", dest, val):
                dest = self.pool.scratch_parent[dest]
                dest_r, dest_w = self.scratch_info[dest]

                best_cyc = max(
                    # write must happen after read
                    dest_r + 1, dest_w + 1,
                )
                def on_pick_cyc(cyc):
                    self.scratch_info[dest] = (dest_r, cyc)
                on_pick_cyc_fn = on_pick_cyc
            case _:
                raise NotImplementedError(f"Unhandled load {instruction}")
        assert best_cyc is not None
        assert on_pick_cyc_fn is not None

        pick_cyc = self.__sched('load', instruction, best_cyc)
        on_pick_cyc_fn(pick_cyc)

    def __sched_store(self, instruction: tuple):
        best_cyc = None
        on_pick_cyc_fn = None
        match instruction:
            case ("store", dest, addr):
                dest = self.pool.scratch_parent[dest]
                addr = self.pool.scratch_parent[addr]
                dest_r, dest_w = self.scratch_info[dest]
                addr_r, addr_w = self.scratch_info[addr]

                best_cyc = max(
                    # read must happen after write
                    dest_w + 1,
                    addr_w + 1
                )
                def on_pick_cyc(cyc):
                    self.scratch_info[dest] = (cyc, dest_w)
                    self.scratch_info[addr] = (cyc, addr_w)
                on_pick_cyc_fn = on_pick_cyc
            case ("vstore", dest, addr):
                dest = self.pool.scratch_parent[dest]
                addr = self.pool.scratch_parent[addr]
                dest_r, dest_w = self.scratch_info[dest]
                addr_r, addr_w = self.scratch_info[addr]

                best_cyc = max(
                    # read must happen after write
                    dest_w + 1,
                    addr_w + 1
                )
                def on_pick_cyc(cyc):
                    self.scratch_info[dest] = (cyc, dest_w)
                    self.scratch_info[addr] = (cyc, addr_w)
                on_pick_cyc_fn = on_pick_cyc
            case _:
                raise NotImplementedError(f"Unhandled store {instruction}")
        assert best_cyc is not None
        assert on_pick_cyc is not None

        pick_cyc = self.__sched('store', instruction, best_cyc)
        on_pick_cyc_fn(pick_cyc)

    def __sched_debug(self, instruction: tuple):
        # Due to how debug works, it needs to be placed right after the register is written
        # at point of assignment
        cyc_to_place = None
        on_pick_cyc_fn = None
        match instruction:
            case ("compare", reg, _):
                reg = self.pool.scratch_parent[reg]
                reg_r, reg_w = self.scratch_info[reg]
                cyc_to_place = reg_w + 1
                def on_pick_cyc(cyc):
                    self.scratch_info[reg] = (cyc_to_place, reg_w)
                on_pick_cyc_fn = on_pick_cyc
            case ("vcompare", reg, _):
                # print('firmanhp vcompare', reg, '/', self.pool.scratch_debug[reg][0], '->', self.pool.scratch_parent[reg], '->', self.scratch_info[reg], instruction)
                reg = self.pool.scratch_parent[reg]
                reg_r, reg_w = self.scratch_info[reg]
                cyc_to_place = reg_w + 1
                # print('firmanhp put in', cyc_to_place)
                def on_pick_cyc(cyc):
                    self.scratch_info[reg] = (cyc_to_place, reg_w)
                on_pick_cyc_fn = on_pick_cyc
            case ("comment", _):
                cyc_to_place = len(self.program) - 1
                def on_pick_cyc(cyc):
                    pass
                on_pick_cyc_fn = on_pick_cyc
            case _:
                raise NotImplementedError(f"Unhandled debug {instruction}")
        assert cyc_to_place is not None
        assert on_pick_cyc_fn is not None

        pick_cyc = self.__sched('debug', instruction, cyc_to_place)
        assert pick_cyc == cyc_to_place, f"Could not designate debug for {instruction} at cyc {cyc_to_place}, got {pick_cyc} instead (full?)"
        on_pick_cyc_fn(pick_cyc)

    def __sched(self, engine: Engine, instruction: tuple, cyc_start: int):
        assert engine in SLOT_LIMITS, f"Unknown eng {engine}"
        # print("scheduling", engine, instruction, "cyc_start:", cyc_start)

        pick_cyc = None
        for offset, cyc in enumerate(self.program[cyc_start:]):
            if engine not in cyc:
                pick_cyc = cyc_start + offset
                break
            if len(cyc[engine]) < SLOT_LIMITS[engine]:
                pick_cyc = cyc_start + offset
                break
            assert engine != "debug", f"debug full??? {len(cyc[engine])}"

        if pick_cyc is None:
            # New cycle
            self.program.append({})
            pick_cyc = len(self.program) - 1

        if engine not in self.program[pick_cyc]:
            self.program[pick_cyc][engine] = [instruction]
        else:
            self.program[pick_cyc][engine].append(instruction)

        assert pick_cyc is not None
        return pick_cyc




