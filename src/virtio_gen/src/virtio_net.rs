// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/* automatically generated by rust-bindgen */

#[repr(C)]
#[derive(Default)]
pub struct __IncompleteArrayField<T>(::std::marker::PhantomData<T>, [T; 0]);
impl<T> __IncompleteArrayField<T> {
    #[inline]
    pub fn new() -> Self {
        __IncompleteArrayField(::std::marker::PhantomData, [])
    }
    #[inline]
    pub unsafe fn as_ptr(&self) -> *const T {
        ::std::mem::transmute(self)
    }
    #[inline]
    pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
        ::std::mem::transmute(self)
    }
    #[inline]
    pub unsafe fn as_slice(&self, len: usize) -> &[T] {
        ::std::slice::from_raw_parts(self.as_ptr(), len)
    }
    #[inline]
    pub unsafe fn as_mut_slice(&mut self, len: usize) -> &mut [T] {
        ::std::slice::from_raw_parts_mut(self.as_mut_ptr(), len)
    }
}
impl<T> ::std::fmt::Debug for __IncompleteArrayField<T> {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        fmt.write_str("__IncompleteArrayField")
    }
}
impl<T> ::std::clone::Clone for __IncompleteArrayField<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self::new()
    }
}
pub const __BITS_PER_LONG: u32 = 64;
pub const __FD_SETSIZE: u32 = 1024;
pub const VIRTIO_ID_NET: u32 = 1;
pub const VIRTIO_ID_BLOCK: u32 = 2;
pub const VIRTIO_ID_CONSOLE: u32 = 3;
pub const VIRTIO_ID_RNG: u32 = 4;
pub const VIRTIO_ID_BALLOON: u32 = 5;
pub const VIRTIO_ID_RPMSG: u32 = 7;
pub const VIRTIO_ID_SCSI: u32 = 8;
pub const VIRTIO_ID_9P: u32 = 9;
pub const VIRTIO_ID_RPROC_SERIAL: u32 = 11;
pub const VIRTIO_ID_CAIF: u32 = 12;
pub const VIRTIO_ID_GPU: u32 = 16;
pub const VIRTIO_ID_INPUT: u32 = 18;
pub const VIRTIO_CONFIG_S_ACKNOWLEDGE: u32 = 1;
pub const VIRTIO_CONFIG_S_DRIVER: u32 = 2;
pub const VIRTIO_CONFIG_S_DRIVER_OK: u32 = 4;
pub const VIRTIO_CONFIG_S_FEATURES_OK: u32 = 8;
pub const VIRTIO_CONFIG_S_FAILED: u32 = 128;
pub const VIRTIO_TRANSPORT_F_START: u32 = 28;
pub const VIRTIO_TRANSPORT_F_END: u32 = 33;
pub const VIRTIO_F_NOTIFY_ON_EMPTY: u32 = 24;
pub const VIRTIO_F_ANY_LAYOUT: u32 = 27;
pub const VIRTIO_F_VERSION_1: u32 = 32;
pub const VIRTIO_F_RING_PACKED: u32 = 34;
pub const ETH_ALEN: u32 = 6;
pub const ETH_TLEN: u32 = 2;
pub const ETH_HLEN: u32 = 14;
pub const ETH_ZLEN: u32 = 60;
pub const ETH_DATA_LEN: u32 = 1500;
pub const ETH_FRAME_LEN: u32 = 1514;
pub const ETH_FCS_LEN: u32 = 4;
pub const ETH_P_LOOP: u32 = 96;
pub const ETH_P_PUP: u32 = 512;
pub const ETH_P_PUPAT: u32 = 513;
pub const ETH_P_TSN: u32 = 8944;
pub const ETH_P_IP: u32 = 2048;
pub const ETH_P_X25: u32 = 2053;
pub const ETH_P_ARP: u32 = 2054;
pub const ETH_P_BPQ: u32 = 2303;
pub const ETH_P_IEEEPUP: u32 = 2560;
pub const ETH_P_IEEEPUPAT: u32 = 2561;
pub const ETH_P_BATMAN: u32 = 17157;
pub const ETH_P_DEC: u32 = 24576;
pub const ETH_P_DNA_DL: u32 = 24577;
pub const ETH_P_DNA_RC: u32 = 24578;
pub const ETH_P_DNA_RT: u32 = 24579;
pub const ETH_P_LAT: u32 = 24580;
pub const ETH_P_DIAG: u32 = 24581;
pub const ETH_P_CUST: u32 = 24582;
pub const ETH_P_SCA: u32 = 24583;
pub const ETH_P_TEB: u32 = 25944;
pub const ETH_P_RARP: u32 = 32821;
pub const ETH_P_ATALK: u32 = 32923;
pub const ETH_P_AARP: u32 = 33011;
pub const ETH_P_8021Q: u32 = 33024;
pub const ETH_P_IPX: u32 = 33079;
pub const ETH_P_IPV6: u32 = 34525;
pub const ETH_P_PAUSE: u32 = 34824;
pub const ETH_P_SLOW: u32 = 34825;
pub const ETH_P_WCCP: u32 = 34878;
pub const ETH_P_MPLS_UC: u32 = 34887;
pub const ETH_P_MPLS_MC: u32 = 34888;
pub const ETH_P_ATMMPOA: u32 = 34892;
pub const ETH_P_PPP_DISC: u32 = 34915;
pub const ETH_P_PPP_SES: u32 = 34916;
pub const ETH_P_LINK_CTL: u32 = 34924;
pub const ETH_P_ATMFATE: u32 = 34948;
pub const ETH_P_PAE: u32 = 34958;
pub const ETH_P_AOE: u32 = 34978;
pub const ETH_P_8021AD: u32 = 34984;
pub const ETH_P_802_EX1: u32 = 34997;
pub const ETH_P_TIPC: u32 = 35018;
pub const ETH_P_8021AH: u32 = 35047;
pub const ETH_P_MVRP: u32 = 35061;
pub const ETH_P_1588: u32 = 35063;
pub const ETH_P_PRP: u32 = 35067;
pub const ETH_P_FCOE: u32 = 35078;
pub const ETH_P_TDLS: u32 = 35085;
pub const ETH_P_FIP: u32 = 35092;
pub const ETH_P_80221: u32 = 35095;
pub const ETH_P_LOOPBACK: u32 = 36864;
pub const ETH_P_QINQ1: u32 = 37120;
pub const ETH_P_QINQ2: u32 = 37376;
pub const ETH_P_QINQ3: u32 = 37632;
pub const ETH_P_EDSA: u32 = 56026;
pub const ETH_P_AF_IUCV: u32 = 64507;
pub const ETH_P_802_3_MIN: u32 = 1536;
pub const ETH_P_802_3: u32 = 1;
pub const ETH_P_AX25: u32 = 2;
pub const ETH_P_ALL: u32 = 3;
pub const ETH_P_802_2: u32 = 4;
pub const ETH_P_SNAP: u32 = 5;
pub const ETH_P_DDCMP: u32 = 6;
pub const ETH_P_WAN_PPP: u32 = 7;
pub const ETH_P_PPP_MP: u32 = 8;
pub const ETH_P_LOCALTALK: u32 = 9;
pub const ETH_P_CAN: u32 = 12;
pub const ETH_P_CANFD: u32 = 13;
pub const ETH_P_PPPTALK: u32 = 16;
pub const ETH_P_TR_802_2: u32 = 17;
pub const ETH_P_MOBITEX: u32 = 21;
pub const ETH_P_CONTROL: u32 = 22;
pub const ETH_P_IRDA: u32 = 23;
pub const ETH_P_ECONET: u32 = 24;
pub const ETH_P_HDLC: u32 = 25;
pub const ETH_P_ARCNET: u32 = 26;
pub const ETH_P_DSA: u32 = 27;
pub const ETH_P_TRAILER: u32 = 28;
pub const ETH_P_PHONET: u32 = 245;
pub const ETH_P_IEEE802154: u32 = 246;
pub const ETH_P_CAIF: u32 = 247;
pub const ETH_P_XDSA: u32 = 248;
pub const VIRTIO_NET_F_CSUM: u32 = 0;
pub const VIRTIO_NET_F_GUEST_CSUM: u32 = 1;
pub const VIRTIO_NET_F_CTRL_GUEST_OFFLOADS: u32 = 2;
pub const VIRTIO_NET_F_MTU: u32 = 3;
pub const VIRTIO_NET_F_MAC: u32 = 5;
pub const VIRTIO_NET_F_GUEST_TSO4: u32 = 7;
pub const VIRTIO_NET_F_GUEST_TSO6: u32 = 8;
pub const VIRTIO_NET_F_GUEST_ECN: u32 = 9;
pub const VIRTIO_NET_F_GUEST_UFO: u32 = 10;
pub const VIRTIO_NET_F_HOST_TSO4: u32 = 11;
pub const VIRTIO_NET_F_HOST_TSO6: u32 = 12;
pub const VIRTIO_NET_F_HOST_ECN: u32 = 13;
pub const VIRTIO_NET_F_HOST_UFO: u32 = 14;
pub const VIRTIO_NET_F_MRG_RXBUF: u32 = 15;
pub const VIRTIO_NET_F_STATUS: u32 = 16;
pub const VIRTIO_NET_F_CTRL_VQ: u32 = 17;
pub const VIRTIO_NET_F_CTRL_RX: u32 = 18;
pub const VIRTIO_NET_F_CTRL_VLAN: u32 = 19;
pub const VIRTIO_NET_F_CTRL_RX_EXTRA: u32 = 20;
pub const VIRTIO_NET_F_GUEST_ANNOUNCE: u32 = 21;
pub const VIRTIO_NET_F_MQ: u32 = 22;
pub const VIRTIO_NET_F_CTRL_MAC_ADDR: u32 = 23;
pub const VIRTIO_NET_F_GSO: u32 = 6;
pub const VIRTIO_NET_S_LINK_UP: u32 = 1;
pub const VIRTIO_NET_S_ANNOUNCE: u32 = 2;
pub const VIRTIO_NET_HDR_F_NEEDS_CSUM: u32 = 1;
pub const VIRTIO_NET_HDR_F_DATA_VALID: u32 = 2;
pub const VIRTIO_NET_HDR_GSO_NONE: u32 = 0;
pub const VIRTIO_NET_HDR_GSO_TCPV4: u32 = 1;
pub const VIRTIO_NET_HDR_GSO_UDP: u32 = 3;
pub const VIRTIO_NET_HDR_GSO_TCPV6: u32 = 4;
pub const VIRTIO_NET_HDR_GSO_ECN: u32 = 128;
pub const VIRTIO_NET_OK: u32 = 0;
pub const VIRTIO_NET_ERR: u32 = 1;
pub const VIRTIO_NET_CTRL_RX: u32 = 0;
pub const VIRTIO_NET_CTRL_RX_PROMISC: u32 = 0;
pub const VIRTIO_NET_CTRL_RX_ALLMULTI: u32 = 1;
pub const VIRTIO_NET_CTRL_RX_ALLUNI: u32 = 2;
pub const VIRTIO_NET_CTRL_RX_NOMULTI: u32 = 3;
pub const VIRTIO_NET_CTRL_RX_NOUNI: u32 = 4;
pub const VIRTIO_NET_CTRL_RX_NOBCAST: u32 = 5;
pub const VIRTIO_NET_CTRL_MAC: u32 = 1;
pub const VIRTIO_NET_CTRL_MAC_TABLE_SET: u32 = 0;
pub const VIRTIO_NET_CTRL_MAC_ADDR_SET: u32 = 1;
pub const VIRTIO_NET_CTRL_VLAN: u32 = 2;
pub const VIRTIO_NET_CTRL_VLAN_ADD: u32 = 0;
pub const VIRTIO_NET_CTRL_VLAN_DEL: u32 = 1;
pub const VIRTIO_NET_CTRL_ANNOUNCE: u32 = 3;
pub const VIRTIO_NET_CTRL_ANNOUNCE_ACK: u32 = 0;
pub const VIRTIO_NET_CTRL_MQ: u32 = 4;
pub const VIRTIO_NET_CTRL_MQ_VQ_PAIRS_SET: u32 = 0;
pub const VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MIN: u32 = 1;
pub const VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MAX: u32 = 32768;
pub const VIRTIO_NET_CTRL_GUEST_OFFLOADS: u32 = 5;
pub const VIRTIO_NET_CTRL_GUEST_OFFLOADS_SET: u32 = 0;
pub type __s8 = ::std::os::raw::c_schar;
pub type __u8 = ::std::os::raw::c_uchar;
pub type __s16 = ::std::os::raw::c_short;
pub type __u16 = ::std::os::raw::c_ushort;
pub type __s32 = ::std::os::raw::c_int;
pub type __u32 = ::std::os::raw::c_uint;
pub type __s64 = ::std::os::raw::c_longlong;
pub type __u64 = ::std::os::raw::c_ulonglong;
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct __kernel_fd_set {
    pub fds_bits: [::std::os::raw::c_ulong; 16usize],
}
#[test]
fn bindgen_test_layout___kernel_fd_set() {
    assert_eq!(
        ::std::mem::size_of::<__kernel_fd_set>(),
        128usize,
        concat!("Size of: ", stringify!(__kernel_fd_set))
    );
    assert_eq!(
        ::std::mem::align_of::<__kernel_fd_set>(),
        8usize,
        concat!("Alignment of ", stringify!(__kernel_fd_set))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<__kernel_fd_set>())).fds_bits as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(__kernel_fd_set),
            "::",
            stringify!(fds_bits)
        )
    );
}
pub type __kernel_sighandler_t =
    ::std::option::Option<unsafe extern "C" fn(arg1: ::std::os::raw::c_int)>;
pub type __kernel_key_t = ::std::os::raw::c_int;
pub type __kernel_mqd_t = ::std::os::raw::c_int;
pub type __kernel_old_uid_t = ::std::os::raw::c_ushort;
pub type __kernel_old_gid_t = ::std::os::raw::c_ushort;
pub type __kernel_old_dev_t = ::std::os::raw::c_ulong;
pub type __kernel_long_t = ::std::os::raw::c_long;
pub type __kernel_ulong_t = ::std::os::raw::c_ulong;
pub type __kernel_ino_t = __kernel_ulong_t;
pub type __kernel_mode_t = ::std::os::raw::c_uint;
pub type __kernel_pid_t = ::std::os::raw::c_int;
pub type __kernel_ipc_pid_t = ::std::os::raw::c_int;
pub type __kernel_uid_t = ::std::os::raw::c_uint;
pub type __kernel_gid_t = ::std::os::raw::c_uint;
pub type __kernel_suseconds_t = __kernel_long_t;
pub type __kernel_daddr_t = ::std::os::raw::c_int;
pub type __kernel_uid32_t = ::std::os::raw::c_uint;
pub type __kernel_gid32_t = ::std::os::raw::c_uint;
pub type __kernel_size_t = __kernel_ulong_t;
pub type __kernel_ssize_t = __kernel_long_t;
pub type __kernel_ptrdiff_t = __kernel_long_t;
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct __kernel_fsid_t {
    pub val: [::std::os::raw::c_int; 2usize],
}
#[test]
fn bindgen_test_layout___kernel_fsid_t() {
    assert_eq!(
        ::std::mem::size_of::<__kernel_fsid_t>(),
        8usize,
        concat!("Size of: ", stringify!(__kernel_fsid_t))
    );
    assert_eq!(
        ::std::mem::align_of::<__kernel_fsid_t>(),
        4usize,
        concat!("Alignment of ", stringify!(__kernel_fsid_t))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<__kernel_fsid_t>())).val as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(__kernel_fsid_t),
            "::",
            stringify!(val)
        )
    );
}
pub type __kernel_off_t = __kernel_long_t;
pub type __kernel_loff_t = ::std::os::raw::c_longlong;
pub type __kernel_time_t = __kernel_long_t;
pub type __kernel_clock_t = __kernel_long_t;
pub type __kernel_timer_t = ::std::os::raw::c_int;
pub type __kernel_clockid_t = ::std::os::raw::c_int;
pub type __kernel_caddr_t = *mut ::std::os::raw::c_char;
pub type __kernel_uid16_t = ::std::os::raw::c_ushort;
pub type __kernel_gid16_t = ::std::os::raw::c_ushort;
pub type __le16 = __u16;
pub type __be16 = __u16;
pub type __le32 = __u32;
pub type __be32 = __u32;
pub type __le64 = __u64;
pub type __be64 = __u64;
pub type __sum16 = __u16;
pub type __wsum = __u32;
pub type __virtio16 = __u16;
pub type __virtio32 = __u32;
pub type __virtio64 = __u64;
#[repr(C, packed)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct ethhdr {
    pub h_dest: [::std::os::raw::c_uchar; 6usize],
    pub h_source: [::std::os::raw::c_uchar; 6usize],
    pub h_proto: __be16,
}
#[test]
fn bindgen_test_layout_ethhdr() {
    assert_eq!(
        ::std::mem::size_of::<ethhdr>(),
        14usize,
        concat!("Size of: ", stringify!(ethhdr))
    );
    assert_eq!(
        ::std::mem::align_of::<ethhdr>(),
        1usize,
        concat!("Alignment of ", stringify!(ethhdr))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<ethhdr>())).h_dest as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(ethhdr),
            "::",
            stringify!(h_dest)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<ethhdr>())).h_source as *const _ as usize },
        6usize,
        concat!(
            "Offset of field: ",
            stringify!(ethhdr),
            "::",
            stringify!(h_source)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<ethhdr>())).h_proto as *const _ as usize },
        12usize,
        concat!(
            "Offset of field: ",
            stringify!(ethhdr),
            "::",
            stringify!(h_proto)
        )
    );
}
#[repr(C, packed)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct virtio_net_config {
    pub mac: [__u8; 6usize],
    pub status: __u16,
    pub max_virtqueue_pairs: __u16,
    pub mtu: __u16,
}
#[test]
fn bindgen_test_layout_virtio_net_config() {
    assert_eq!(
        ::std::mem::size_of::<virtio_net_config>(),
        12usize,
        concat!("Size of: ", stringify!(virtio_net_config))
    );
    assert_eq!(
        ::std::mem::align_of::<virtio_net_config>(),
        1usize,
        concat!("Alignment of ", stringify!(virtio_net_config))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_config>())).mac as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_config),
            "::",
            stringify!(mac)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_config>())).status as *const _ as usize },
        6usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_config),
            "::",
            stringify!(status)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<virtio_net_config>())).max_virtqueue_pairs as *const _ as usize
        },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_config),
            "::",
            stringify!(max_virtqueue_pairs)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_config>())).mtu as *const _ as usize },
        10usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_config),
            "::",
            stringify!(mtu)
        )
    );
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct virtio_net_hdr_v1 {
    pub flags: __u8,
    pub gso_type: __u8,
    pub hdr_len: __virtio16,
    pub gso_size: __virtio16,
    pub csum_start: __virtio16,
    pub csum_offset: __virtio16,
    pub num_buffers: __virtio16,
}
#[test]
fn bindgen_test_layout_virtio_net_hdr_v1() {
    assert_eq!(
        ::std::mem::size_of::<virtio_net_hdr_v1>(),
        12usize,
        concat!("Size of: ", stringify!(virtio_net_hdr_v1))
    );
    assert_eq!(
        ::std::mem::align_of::<virtio_net_hdr_v1>(),
        2usize,
        concat!("Alignment of ", stringify!(virtio_net_hdr_v1))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_hdr_v1>())).flags as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr_v1),
            "::",
            stringify!(flags)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_hdr_v1>())).gso_type as *const _ as usize },
        1usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr_v1),
            "::",
            stringify!(gso_type)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_hdr_v1>())).hdr_len as *const _ as usize },
        2usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr_v1),
            "::",
            stringify!(hdr_len)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_hdr_v1>())).gso_size as *const _ as usize },
        4usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr_v1),
            "::",
            stringify!(gso_size)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_hdr_v1>())).csum_start as *const _ as usize },
        6usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr_v1),
            "::",
            stringify!(csum_start)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_hdr_v1>())).csum_offset as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr_v1),
            "::",
            stringify!(csum_offset)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_hdr_v1>())).num_buffers as *const _ as usize },
        10usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr_v1),
            "::",
            stringify!(num_buffers)
        )
    );
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct virtio_net_hdr {
    pub flags: __u8,
    pub gso_type: __u8,
    pub hdr_len: __virtio16,
    pub gso_size: __virtio16,
    pub csum_start: __virtio16,
    pub csum_offset: __virtio16,
}
#[test]
fn bindgen_test_layout_virtio_net_hdr() {
    assert_eq!(
        ::std::mem::size_of::<virtio_net_hdr>(),
        10usize,
        concat!("Size of: ", stringify!(virtio_net_hdr))
    );
    assert_eq!(
        ::std::mem::align_of::<virtio_net_hdr>(),
        2usize,
        concat!("Alignment of ", stringify!(virtio_net_hdr))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_hdr>())).flags as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr),
            "::",
            stringify!(flags)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_hdr>())).gso_type as *const _ as usize },
        1usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr),
            "::",
            stringify!(gso_type)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_hdr>())).hdr_len as *const _ as usize },
        2usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr),
            "::",
            stringify!(hdr_len)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_hdr>())).gso_size as *const _ as usize },
        4usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr),
            "::",
            stringify!(gso_size)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_hdr>())).csum_start as *const _ as usize },
        6usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr),
            "::",
            stringify!(csum_start)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_hdr>())).csum_offset as *const _ as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr),
            "::",
            stringify!(csum_offset)
        )
    );
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct virtio_net_hdr_mrg_rxbuf {
    pub hdr: virtio_net_hdr,
    pub num_buffers: __virtio16,
}
#[test]
fn bindgen_test_layout_virtio_net_hdr_mrg_rxbuf() {
    assert_eq!(
        ::std::mem::size_of::<virtio_net_hdr_mrg_rxbuf>(),
        12usize,
        concat!("Size of: ", stringify!(virtio_net_hdr_mrg_rxbuf))
    );
    assert_eq!(
        ::std::mem::align_of::<virtio_net_hdr_mrg_rxbuf>(),
        2usize,
        concat!("Alignment of ", stringify!(virtio_net_hdr_mrg_rxbuf))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_hdr_mrg_rxbuf>())).hdr as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr_mrg_rxbuf),
            "::",
            stringify!(hdr)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<virtio_net_hdr_mrg_rxbuf>())).num_buffers as *const _ as usize
        },
        10usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_hdr_mrg_rxbuf),
            "::",
            stringify!(num_buffers)
        )
    );
}
#[repr(C, packed)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct virtio_net_ctrl_hdr {
    pub class: __u8,
    pub cmd: __u8,
}
#[test]
fn bindgen_test_layout_virtio_net_ctrl_hdr() {
    assert_eq!(
        ::std::mem::size_of::<virtio_net_ctrl_hdr>(),
        2usize,
        concat!("Size of: ", stringify!(virtio_net_ctrl_hdr))
    );
    assert_eq!(
        ::std::mem::align_of::<virtio_net_ctrl_hdr>(),
        1usize,
        concat!("Alignment of ", stringify!(virtio_net_ctrl_hdr))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_ctrl_hdr>())).class as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_ctrl_hdr),
            "::",
            stringify!(class)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<virtio_net_ctrl_hdr>())).cmd as *const _ as usize },
        1usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_ctrl_hdr),
            "::",
            stringify!(cmd)
        )
    );
}
pub type virtio_net_ctrl_ack = __u8;
#[repr(C, packed)]
#[derive(Default)]
pub struct virtio_net_ctrl_mac {
    pub entries: __virtio32,
    pub macs: __IncompleteArrayField<[__u8; 6usize]>,
}
#[test]
fn bindgen_test_layout_virtio_net_ctrl_mac() {
    assert_eq!(
        ::std::mem::size_of::<virtio_net_ctrl_mac>(),
        4usize,
        concat!("Size of: ", stringify!(virtio_net_ctrl_mac))
    );
    assert_eq!(
        ::std::mem::align_of::<virtio_net_ctrl_mac>(),
        1usize,
        concat!("Alignment of ", stringify!(virtio_net_ctrl_mac))
    );
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct virtio_net_ctrl_mq {
    pub virtqueue_pairs: __virtio16,
}
#[test]
fn bindgen_test_layout_virtio_net_ctrl_mq() {
    assert_eq!(
        ::std::mem::size_of::<virtio_net_ctrl_mq>(),
        2usize,
        concat!("Size of: ", stringify!(virtio_net_ctrl_mq))
    );
    assert_eq!(
        ::std::mem::align_of::<virtio_net_ctrl_mq>(),
        2usize,
        concat!("Alignment of ", stringify!(virtio_net_ctrl_mq))
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<virtio_net_ctrl_mq>())).virtqueue_pairs as *const _ as usize
        },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(virtio_net_ctrl_mq),
            "::",
            stringify!(virtqueue_pairs)
        )
    );
}
