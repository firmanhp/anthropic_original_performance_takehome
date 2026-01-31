from problem import (
    SCRATCH_SIZE
)

class ScratchRegPool:
    def __init__(self):
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def alloc(self, name, length=1):
        # TODO: use length
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    # def const(self, val, name=None):
    #     if val not in self.const_map:
    #         addr = self.alloc_scratch(name)
    #         self.add("load", ("const", addr, val))
    #         self.const_map[val] = addr
    #     return self.const_map[val]

    
