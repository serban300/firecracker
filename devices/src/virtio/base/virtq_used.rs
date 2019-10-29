use crate::virtio::base::{VirtqRing, VirtqRingBase, VirtqRingIndex};
use memory_model::{DataInit, GuestAddress, GuestMemory, GuestMemoryError};

/// Shadow structure for the following virtio struct:
/// ```C
/// struct virtq_used_elem {
///      le32 id;
///      le32 len;
/// };
/// ```
#[derive(Clone, Copy)]
#[repr(C)]
pub struct VirtqUsedElem {
    pub(crate) id: u32,
    pub(crate) len: u32,
}

unsafe impl DataInit for VirtqUsedElem {}

/// Mirror structure for the following virtio dynamically sized struct:
/// ```C
/// struct virtq_used {
///     le16 flags;
///     le16 idx;
///     struct virtq_used_elem ring[ /* Queue Size */ ];
///     le16 avail_event; /* Only if VIRTIO_F_EVENT_IDX */
/// };
/// ```
pub struct VirtqUsed {
    base: VirtqRingBase,
}

impl VirtqUsed {
    const IDX_OFFSET: usize = 2;
    const RING_OFFSET: usize = 4;

    pub fn new(mem: GuestMemory, addr: GuestAddress, size: u16) -> VirtqUsed {
        VirtqUsed {
            base: VirtqRingBase::new(mem, addr, size),
        }
    }

    pub fn set_idx(&self, idx: u16) -> Result<(), GuestMemoryError> {
        self.write(Self::IDX_OFFSET, idx)
    }
}

impl VirtqRing for VirtqUsed {
    fn base(&self) -> &VirtqRingBase {
        &self.base
    }
}

impl VirtqRingIndex for VirtqUsed {
    type Element = VirtqUsedElem;

    fn ring_offset() -> usize {
        Self::RING_OFFSET
    }
}
