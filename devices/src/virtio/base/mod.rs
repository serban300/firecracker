pub mod virtq_avail;
pub mod virtq_desc;
pub mod virtq_used;

use memory_model::{DataInit, GuestAddress, GuestMemory, GuestMemoryError};
use std::mem::size_of;

pub struct VirtqRingBase {
    mem: GuestMemory,
    addr: GuestAddress,
    size: u16,
}

impl VirtqRingBase {
    fn new(mem: GuestMemory, addr: GuestAddress, size: u16) -> VirtqRingBase {
        VirtqRingBase { mem, addr, size }
    }
}

pub trait VirtqRing {
    fn base(&self) -> &VirtqRingBase;

    fn mem(&self) -> &GuestMemory {
        &self.base().mem
    }

    fn addr(&self) -> GuestAddress {
        self.base().addr
    }

    fn size(&self) -> u16 {
        self.base().size
    }

    fn normalize_index(&self, index: u16) -> u16 {
        index % self.size()
    }

    fn read<T: DataInit>(&self, offset: usize) -> Result<T, GuestMemoryError> {
        self.mem()
            .read_obj_from_addr(self.addr().unchecked_add(offset))
    }

    fn write<T: DataInit>(&self, offset: usize, val: T) -> Result<(), GuestMemoryError> {
        self.mem()
            .write_obj_at_addr(val, self.addr().unchecked_add(offset))
    }
}

pub trait VirtqRingIndex: VirtqRing {
    type Element: DataInit;

    fn ring_offset() -> usize;

    fn element_offset(&self, index: u16) -> usize {
        Self::ring_offset() + self.normalize_index(index) as usize * size_of::<Self::Element>()
    }

    fn get(&self, index: u16) -> Result<Self::Element, GuestMemoryError> {
        self.read(self.element_offset(index))
    }

    fn set(&self, index: u16, element: Self::Element) -> Result<(), GuestMemoryError> {
        self.write(self.element_offset(index), element)
    }
}
