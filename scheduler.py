from problem import (
    SCRATCH_SIZE,
    Instruction
)

# I didn't like their definition of "Instruction"
CycleAssignment = Instruction

class ScratchRegPool:
    def __init__(self):
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        
        self.scratch_free = []

    def alloc(self, name=None, length=1):
        if self.scratch_ptr + length >= SCRATCH_SIZE:
            # Cannot extend more
            addr = self.__pop_from_free(length)
            assert addr is not None, f"Cannot allocate/recycle anymore for reg of len {length}"
            return addr

        # TODO: use length
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
        self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        return addr

    def free(self, scratch_addr):
        assert scratch_addr not in self.scratch_free, f"Double free: {scratch_addr}"
        self.scratch_free.append(scratch_addr)

    def __pop_from_free(self, length):
        idx = None
        for (i, addr) in enumerate(self.scratch_free):
            (_, l) = self.scratch_debug[addr]
            if l == length:
                idx = i
                break
        if idx is None:
            return None
        return self.scratch_free.pop(i)



    
