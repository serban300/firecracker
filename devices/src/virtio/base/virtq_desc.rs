use crate::virtio::base::{VirtqRing, VirtqRingBase, VirtqRingIndex};
use memory_model::{DataInit, GuestAddress, GuestMemory};

/// Shadow structure for the following virtio struct:
/// ```C
/// struct virtq_desc {
///     le64 addr;
///     le32 len;
///     le16 flags;
///     le16 next;
/// };
/// ```
#[derive(Clone, Copy)]
#[repr(C)]
pub struct VirtqDescElem {
    pub(crate) addr: u64,
    pub(crate) len: u32,
    pub(crate) flags: u16,
    pub(crate) next: u16,
}

unsafe impl DataInit for VirtqDescElem {}

pub struct VirtqDesc {
    base: VirtqRingBase,
}

impl VirtqDesc {
    pub fn new(mem: GuestMemory, addr: GuestAddress, size: u16) -> VirtqDesc {
        VirtqDesc {
            base: VirtqRingBase::new(mem, addr, size),
        }
    }
}

impl VirtqRing for VirtqDesc {
    fn base(&self) -> &VirtqRingBase {
        &self.base
    }
}

impl VirtqRingIndex for VirtqDesc {
    type Element = VirtqDescElem;

    fn ring_offset() -> usize {
        0
    }
}
