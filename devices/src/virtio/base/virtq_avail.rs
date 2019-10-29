use crate::virtio::base::{VirtqRing, VirtqRingBase, VirtqRingIndex};
use memory_model::{GuestAddress, GuestMemory, GuestMemoryError};

/// Mirror structure for the following virtio dynamically sized struct:
/// ```C
/// struct virtq_avail {
///     le16 flags;
///     le16 idx;
///     le16 ring[ /* Queue Size */ ];
///     le16 used_event; /* Only if VIRTIO_F_EVENT_IDX */
/// };
/// ```
pub struct VirtqAvail {
    base: VirtqRingBase,
}

impl VirtqAvail {
    const IDX_OFFSET: usize = 2;
    const RING_OFFSET: usize = 4;

    pub fn new(mem: GuestMemory, addr: GuestAddress, size: u16) -> VirtqAvail {
        VirtqAvail {
            base: VirtqRingBase::new(mem, addr, size),
        }
    }

    pub fn idx(&self) -> Result<u16, GuestMemoryError> {
        self.read(Self::IDX_OFFSET)
    }
}

impl VirtqRing for VirtqAvail {
    fn base(&self) -> &VirtqRingBase {
        &self.base
    }
}

impl VirtqRingIndex for VirtqAvail {
    type Element = u16;

    fn ring_offset() -> usize {
        Self::RING_OFFSET
    }
}
