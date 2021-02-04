// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

use crate::virtio::net::tap::Tap;
#[cfg(test)]
use crate::virtio::net::test_utils::Mocks;
use crate::virtio::net::Error;
use crate::virtio::net::Result;
use crate::virtio::net::{MAX_BUFFER_SIZE, QUEUE_SIZE, QUEUE_SIZES, RX_INDEX, TX_INDEX};
use crate::virtio::{
    ActivateResult, DeviceState, Queue, VirtioDevice, TYPE_NET, VIRTIO_MMIO_INT_VRING,
};
use crate::{report_net_event_fail, Error as DeviceError};

use dumbo::pdu::ethernet::EthernetFrame;
use libc::EAGAIN;
use logger::{error, warn, IncMetric, METRICS};
use mmds::ns::MmdsNetworkStack;
use rate_limiter::{BucketUpdate, RateLimiter, TokenType};
#[cfg(not(test))]
use std::io;
use std::io::{Read, Write};
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::{cmp, mem, result};
use utils::eventfd::EventFd;
use utils::net::mac::{MacAddr, MAC_ADDR_LEN};
use virtio_gen::virtio_net::{
    virtio_net_hdr_v1, VIRTIO_F_RING_PACKED, VIRTIO_F_VERSION_1, VIRTIO_NET_F_CSUM,
    VIRTIO_NET_F_GUEST_CSUM, VIRTIO_NET_F_GUEST_TSO4, VIRTIO_NET_F_GUEST_UFO,
    VIRTIO_NET_F_HOST_TSO4, VIRTIO_NET_F_HOST_UFO, VIRTIO_NET_F_MAC,
};
use vm_memory::{ByteValued, Bytes, GuestAddress, GuestMemoryError, GuestMemoryMmap};

enum FrontendError {
    AddUsed,
    DescriptorChainTooSmall,
    EmptyQueue,
    GuestMemory(GuestMemoryError),
    ReadOnlyDescriptor,
}

pub(crate) fn vnet_hdr_len() -> usize {
    mem::size_of::<virtio_net_hdr_v1>()
}

// Frames being sent/received through the network device model have a VNET header. This
// function returns a slice which holds the L2 frame bytes without this header.
fn frame_bytes_from_buf(buf: &[u8]) -> Result<&[u8]> {
    if buf.len() < vnet_hdr_len() {
        Err(Error::VnetHeaderMissing)
    } else {
        Ok(&buf[vnet_hdr_len()..])
    }
}

fn frame_bytes_from_buf_mut(buf: &mut [u8]) -> Result<&mut [u8]> {
    if buf.len() < vnet_hdr_len() {
        Err(Error::VnetHeaderMissing)
    } else {
        Ok(&mut buf[vnet_hdr_len()..])
    }
}

// This initializes to all 0 the VNET hdr part of a buf.
fn init_vnet_hdr(buf: &mut [u8]) {
    // The buffer should be larger than vnet_hdr_len.
    // TODO: any better way to set all these bytes to 0? Or is this optimized by the compiler?
    for i in &mut buf[0..vnet_hdr_len()] {
        *i = 0;
    }
}

#[derive(Clone, Copy)]
pub struct ConfigSpace {
    pub guest_mac: [u8; MAC_ADDR_LEN],
}

impl Default for ConfigSpace {
    fn default() -> ConfigSpace {
        ConfigSpace {
            guest_mac: [0; MAC_ADDR_LEN],
        }
    }
}

unsafe impl ByteValued for ConfigSpace {}

pub struct Net {
    pub(crate) id: String,

    pub(crate) tap: Tap,

    pub(crate) avail_features: u64,
    pub(crate) acked_features: u64,

    pub(crate) queues: Vec<Queue>,
    pub(crate) queue_evts: Vec<EventFd>,

    pub(crate) rx_rate_limiter: RateLimiter,
    pub(crate) tx_rate_limiter: RateLimiter,

    pub(crate) rx_deferred_frame: bool,
    rx_deferred_irqs: bool,

    rx_bytes_read: usize,
    rx_frame_buf: [u8; MAX_BUFFER_SIZE],

    tx_iovec: Vec<(GuestAddress, usize)>,
    tx_frame_buf: [u8; MAX_BUFFER_SIZE],

    pub(crate) interrupt_status: Arc<AtomicUsize>,
    pub(crate) interrupt_evt: EventFd,

    pub(crate) config_space: ConfigSpace,
    pub(crate) guest_mac: Option<MacAddr>,

    pub(crate) device_state: DeviceState,
    pub(crate) activate_evt: EventFd,

    pub(crate) mmds_ns: Option<MmdsNetworkStack>,

    #[cfg(test)]
    pub(crate) mocks: Mocks,
}

impl Net {
    /// Create a new virtio network device with the given TAP interface.
    pub fn new_with_tap(
        id: String,
        tap_if_name: String,
        guest_mac: Option<&MacAddr>,
        rx_rate_limiter: RateLimiter,
        tx_rate_limiter: RateLimiter,
        allow_mmds_requests: bool,
    ) -> Result<Self> {
        let tap = Tap::open_named(&tap_if_name).map_err(Error::TapOpen)?;

        // Set offload flags to match the virtio features below.
        tap.set_offload(
            net_gen::TUN_F_CSUM | net_gen::TUN_F_UFO | net_gen::TUN_F_TSO4 | net_gen::TUN_F_TSO6,
        )
        .map_err(Error::TapSetOffload)?;

        let vnet_hdr_size = vnet_hdr_len() as i32;
        tap.set_vnet_hdr_size(vnet_hdr_size)
            .map_err(Error::TapSetVnetHdrSize)?;

        let mut avail_features = 1 << VIRTIO_NET_F_GUEST_CSUM
            | 1 << VIRTIO_NET_F_CSUM
            | 1 << VIRTIO_NET_F_GUEST_TSO4
            | 1 << VIRTIO_NET_F_GUEST_UFO
            | 1 << VIRTIO_NET_F_HOST_TSO4
            | 1 << VIRTIO_NET_F_HOST_UFO
            | 1 << VIRTIO_F_VERSION_1
            | 1 << VIRTIO_F_RING_PACKED;

        let mut config_space = ConfigSpace::default();
        if let Some(mac) = guest_mac {
            config_space.guest_mac.copy_from_slice(mac.get_bytes());
            // When this feature isn't available, the driver generates a random MAC address.
            // Otherwise, it should attempt to read the device MAC address from the config space.
            avail_features |= 1 << VIRTIO_NET_F_MAC;
        }

        let mut queue_evts = Vec::new();
        for _ in QUEUE_SIZES.iter() {
            queue_evts.push(EventFd::new(libc::EFD_NONBLOCK).map_err(Error::EventFd)?);
        }

        let queues = QUEUE_SIZES.iter().map(|&s| Queue::new(s)).collect();

        let mmds_ns = if allow_mmds_requests {
            Some(MmdsNetworkStack::new_with_defaults(None))
        } else {
            None
        };
        Ok(Net {
            id,
            tap,
            avail_features,
            acked_features: 0u64,
            queues,
            queue_evts,
            rx_rate_limiter,
            tx_rate_limiter,
            rx_deferred_frame: false,
            rx_deferred_irqs: false,
            rx_bytes_read: 0,
            rx_frame_buf: [0u8; MAX_BUFFER_SIZE],
            tx_frame_buf: [0u8; MAX_BUFFER_SIZE],
            tx_iovec: Vec::with_capacity(QUEUE_SIZE as usize),
            interrupt_status: Arc::new(AtomicUsize::new(0)),
            interrupt_evt: EventFd::new(libc::EFD_NONBLOCK).map_err(Error::EventFd)?,
            device_state: DeviceState::Inactive,
            activate_evt: EventFd::new(libc::EFD_NONBLOCK).map_err(Error::EventFd)?,
            config_space,
            mmds_ns,
            guest_mac: guest_mac.copied(),

            #[cfg(test)]
            mocks: Mocks::default(),
        })
    }

    /// Provides the ID of this net device.
    pub fn id(&self) -> &String {
        &self.id
    }

    /// Provides the MAC of this net device.
    pub fn guest_mac(&self) -> Option<&MacAddr> {
        self.guest_mac.as_ref()
    }

    /// Provides a mutable reference to the `MmdsNetworkStack`.
    pub fn mmds_ns_mut(&mut self) -> Option<&mut MmdsNetworkStack> {
        self.mmds_ns.as_mut()
    }

    fn signal_used_queue(&mut self) -> result::Result<(), DeviceError> {
        self.interrupt_status
            .fetch_or(VIRTIO_MMIO_INT_VRING as usize, Ordering::SeqCst);
        self.interrupt_evt.write(1).map_err(|e| {
            error!("Failed to signal used queue: {:?}", e);
            METRICS.net.event_fails.inc();
            DeviceError::FailedSignalingUsedQueue(e)
        })?;

        self.rx_deferred_irqs = false;
        Ok(())
    }

    fn signal_rx_used_queue(&mut self) -> result::Result<(), DeviceError> {
        if self.rx_deferred_irqs {
            return self.signal_used_queue();
        }

        Ok(())
    }

    // Attempts to copy a single frame into the guest if there is enough
    // rate limiting budget.
    // Returns true on successful frame delivery.
    fn rate_limited_rx_single_frame(&mut self) -> bool {
        // If limiter.consume() fails it means there is no more TokenType::Ops
        // budget and rate limiting is in effect.
        if !self.rx_rate_limiter.consume(1, TokenType::Ops) {
            METRICS.net.rx_rate_limiter_throttled.inc();
            return false;
        }
        // If limiter.consume() fails it means there is no more TokenType::Bytes
        // budget and rate limiting is in effect.
        if !self
            .rx_rate_limiter
            .consume(self.rx_bytes_read as u64, TokenType::Bytes)
        {
            // revert the OPS consume()
            self.rx_rate_limiter.manual_replenish(1, TokenType::Ops);
            METRICS.net.rx_rate_limiter_throttled.inc();
            return false;
        }

        // Attempt frame delivery.
        let success = self.write_frame_to_guest();

        // Undo the tokens consumption if guest delivery failed.
        if !success {
            // revert the OPS consume()
            self.rx_rate_limiter.manual_replenish(1, TokenType::Ops);
            // revert the BYTES consume()
            self.rx_rate_limiter
                .manual_replenish(self.rx_bytes_read as u64, TokenType::Bytes);
        }
        success
    }

    // Copies a single frame from `self.rx_frame_buf` into the guest.
    fn do_write_frame_to_guest(&mut self) -> std::result::Result<(), FrontendError> {
        let mut result: std::result::Result<(), FrontendError> = Ok(());
        let mem = match self.device_state {
            DeviceState::Activated(ref mem) => mem,
            // This should never happen, it's been already validated in the event handler.
            DeviceState::Inactive => unreachable!(),
        };

        let queue = &mut self.queues[RX_INDEX];
        let head_descriptor = queue.packed_pop(mem).ok_or_else(|| {
            error!("Empty RX Queue");
            METRICS.net.no_rx_avail_buffer.inc();
            FrontendError::EmptyQueue
        })?;
        let tail_buf_id = head_descriptor.buf_id;
        let head_index = head_descriptor.desc_index;
        let mut chain_len = 0;

        let mut frame_slice = &self.rx_frame_buf[..self.rx_bytes_read];
        let frame_len = frame_slice.len();
        let mut maybe_next_descriptor = Some(head_descriptor);
        while let Some(descriptor) = &maybe_next_descriptor {
            if frame_slice.is_empty() {
                break;
            }

            if !descriptor.is_write_only() {
                result = Err(FrontendError::ReadOnlyDescriptor);
                break;
            }

            let len = std::cmp::min(frame_slice.len(), descriptor.len as usize);
            match mem.write_slice(&frame_slice[..len], descriptor.addr) {
                Ok(()) => {
                    METRICS.net.rx_count.inc();
                    frame_slice = &frame_slice[len..];
                }
                Err(e) => {
                    error!("Failed to write slice: {:?}", e);
                    match e {
                        GuestMemoryError::PartialBuffer { .. } => &METRICS.net.rx_partial_writes,
                        _ => &METRICS.net.rx_fails,
                    }
                    .inc();
                    result = Err(FrontendError::GuestMemory(e));
                    break;
                }
            };

            chain_len += 1;

            maybe_next_descriptor = descriptor.next_descriptor();
        }
        while let Some(descriptor) = &maybe_next_descriptor {
            chain_len += 1;

            maybe_next_descriptor = descriptor.next_descriptor();
        }

        if result.is_ok() && !frame_slice.is_empty() {
            warn!("Receiving buffer is too small to hold frame of current size");
            METRICS.net.rx_fails.inc();
            result = Err(FrontendError::DescriptorChainTooSmall);
        }

        // Mark the descriptor chain as used. If an error occurred, skip the descriptor chain.
        let used_len = if result.is_err() { 0 } else { frame_len as u32 };
        error!(
            "Rx: used chain: buf id {}, chain_len: {}",
            tail_buf_id, chain_len
        );
        queue
            .packed_add_used(mem, head_index, tail_buf_id, chain_len, used_len)
            .map_err(|e| {
                error!("Failed to add available descriptor {}: {}", head_index, e);
                FrontendError::AddUsed
            })?;
        self.rx_deferred_irqs = true;

        if result.is_ok() {
            METRICS.net.rx_bytes_count.add(frame_len);
            METRICS.net.rx_packets_count.inc();
        }
        result
    }

    // Copies a single frame from `self.rx_frame_buf` into the guest. In case of an error retries
    // the operation if possible. Returns true if the operation was successfull.
    fn write_frame_to_guest(&mut self) -> bool {
        let max_iterations = self.queues[RX_INDEX].actual_size();
        for _ in 0..max_iterations {
            match self.do_write_frame_to_guest() {
                Ok(()) => return true,
                Err(FrontendError::EmptyQueue) | Err(FrontendError::AddUsed) => {
                    return false;
                }
                Err(_) => {
                    // retry
                    continue;
                }
            }
        }

        false
    }

    // Tries to detour the frame to MMDS and if MMDS doesn't accept it, sends it on the host TAP.
    //
    // `frame_buf` should contain the frame bytes in a slice of exact length.
    // Returns whether MMDS consumed the frame.
    fn write_to_mmds_or_tap(
        mmds_ns: Option<&mut MmdsNetworkStack>,
        rate_limiter: &mut RateLimiter,
        frame_buf: &[u8],
        tap: &mut Tap,
        guest_mac: Option<MacAddr>,
    ) -> Result<bool> {
        let checked_frame = |frame_buf| {
            frame_bytes_from_buf(frame_buf).map_err(|e| {
                error!("VNET header missing in the TX frame.");
                METRICS.net.tx_malformed_frames.inc();
                e
            })
        };
        if let Some(ns) = mmds_ns {
            if ns.detour_frame(checked_frame(frame_buf)?) {
                METRICS.mmds.rx_accepted.inc();

                // MMDS frames are not accounted by the rate limiter.
                rate_limiter.manual_replenish(frame_buf.len() as u64, TokenType::Bytes);
                rate_limiter.manual_replenish(1, TokenType::Ops);

                // MMDS consumed the frame.
                return Ok(true);
            }
        }

        // This frame goes to the TAP.

        // Check for guest MAC spoofing.
        if let Some(mac) = guest_mac {
            let _ = EthernetFrame::from_bytes(checked_frame(frame_buf)?).map(|eth_frame| {
                if mac != eth_frame.src_mac() {
                    METRICS.net.tx_spoofed_mac_count.inc();
                }
            });
        }

        match tap.write(frame_buf) {
            Ok(_) => {
                METRICS.net.tx_bytes_count.add(frame_buf.len());
                METRICS.net.tx_packets_count.inc();
                METRICS.net.tx_count.inc();
            }
            Err(e) => {
                error!("Failed to write to tap: {:?}", e);
                METRICS.net.tap_write_fails.inc();
            }
        };
        Ok(false)
    }

    // We currently prioritize packets from the MMDS over regular network packets.
    fn read_from_mmds_or_tap(&mut self) -> Result<usize> {
        if let Some(ns) = self.mmds_ns.as_mut() {
            if let Some(len) =
                ns.write_next_frame(frame_bytes_from_buf_mut(&mut self.rx_frame_buf)?)
            {
                let len = len.get();
                METRICS.mmds.tx_frames.inc();
                METRICS.mmds.tx_bytes.add(len);
                init_vnet_hdr(&mut self.rx_frame_buf);
                return Ok(vnet_hdr_len() + len);
            }
        }

        self.read_tap().map_err(Error::IO)
    }

    fn process_rx(&mut self) -> result::Result<(), DeviceError> {
        // Read as many frames as possible.
        loop {
            match self.read_from_mmds_or_tap() {
                Ok(count) => {
                    self.rx_bytes_read = count;
                    METRICS.net.rx_count.inc();
                    if !self.rate_limited_rx_single_frame() {
                        self.rx_deferred_frame = true;
                        break;
                    }
                }
                Err(Error::IO(e)) => {
                    // The tap device is non-blocking, so any error aside from EAGAIN is
                    // unexpected.
                    match e.raw_os_error() {
                        Some(err) if err == EAGAIN => (),
                        _ => {
                            error!("Failed to read tap: {:?}", e);
                            METRICS.net.tap_read_fails.inc();
                            return Err(DeviceError::FailedReadTap);
                        }
                    };
                    break;
                }
                Err(e) => {
                    error!("Spurious error in network RX: {:?}", e);
                }
            }
        }

        // At this point we processed as many Rx frames as possible.
        // We have to wake the guest if at least one descriptor chain has been used.
        self.signal_rx_used_queue()
    }

    // Process the deferred frame first, then continue reading from tap.
    fn handle_deferred_frame(&mut self) -> result::Result<(), DeviceError> {
        if self.rate_limited_rx_single_frame() {
            self.rx_deferred_frame = false;
            // process_rx() was interrupted possibly before consuming all
            // packets in the tap; try continuing now.
            return self.process_rx();
        }

        self.signal_rx_used_queue()
    }

    fn resume_rx(&mut self) -> result::Result<(), DeviceError> {
        if self.rx_deferred_frame {
            self.handle_deferred_frame()
        } else {
            Ok(())
        }
    }

    fn process_tx(&mut self) -> result::Result<(), DeviceError> {
        let mem = match self.device_state {
            DeviceState::Activated(ref mem) => mem,
            // This should never happen, it's been already validated in the event handler.
            DeviceState::Inactive => unreachable!(),
        };

        // The MMDS network stack works like a state machine, based on synchronous calls, and
        // without being added to any event loop. If any frame is accepted by the MMDS, we also
        // trigger a process_rx() which checks if there are any new frames to be sent, starting
        // with the MMDS network stack.
        let mut process_rx_for_mmds = false;
        let mut raise_irq = false;
        let tx_queue = &mut self.queues[TX_INDEX];

        while let Some(head) = tx_queue.packed_pop(mem) {
            // If limiter.consume() fails it means there is no more TokenType::Ops
            // budget and rate limiting is in effect.
            if !self.tx_rate_limiter.consume(1, TokenType::Ops) {
                // Stop processing the queue and return this descriptor chain to the
                // avail ring, for later processing.
                tx_queue.undo_pop();
                METRICS.net.tx_rate_limiter_throttled.inc();
                break;
            }

            let head_desc_index = head.desc_index;
            let mut tail_buf_id = head.buf_id;
            let mut chain_len = 0;
            let mut read_count = 0;
            let mut next_desc = Some(head);

            self.tx_iovec.clear();
            while let Some(desc) = next_desc {
                tail_buf_id = desc.buf_id;
                chain_len += 1;

                if desc.is_write_only() {
                    self.tx_iovec.clear();
                    break;
                }
                self.tx_iovec.push((desc.addr, desc.len as usize));
                read_count += desc.len as usize;
                next_desc = desc.next_descriptor();
            }

            // If limiter.consume() fails it means there is no more TokenType::Bytes
            // budget and rate limiting is in effect.
            if !self
                .tx_rate_limiter
                .consume(read_count as u64, TokenType::Bytes)
            {
                // revert the OPS consume()
                self.tx_rate_limiter.manual_replenish(1, TokenType::Ops);
                // Stop processing the queue and return this descriptor chain to the
                // avail ring, for later processing.
                tx_queue.undo_pop();
                METRICS.net.tx_rate_limiter_throttled.inc();
                break;
            }

            read_count = 0;
            // Copy buffer from across multiple descriptors.
            // TODO(performance - Issue #420): change this to use `writev()` instead of `write()`
            // and get rid of the intermediate buffer.
            for (desc_addr, desc_len) in self.tx_iovec.drain(..) {
                let limit = cmp::min((read_count + desc_len) as usize, self.tx_frame_buf.len());

                let read_result = mem.read_slice(
                    &mut self.tx_frame_buf[read_count..limit as usize],
                    desc_addr,
                );
                match read_result {
                    Ok(()) => {
                        read_count += limit - read_count;
                        METRICS.net.tx_count.inc();
                    }
                    Err(e) => {
                        error!("Failed to read slice: {:?}", e);
                        match e {
                            GuestMemoryError::PartialBuffer { .. } => &METRICS.net.tx_partial_reads,
                            _ => &METRICS.net.tx_fails,
                        }
                        .inc();
                        read_count = 0;
                        break;
                    }
                }
            }

            let frame_consumed_by_mmds = Self::write_to_mmds_or_tap(
                self.mmds_ns.as_mut(),
                &mut self.tx_rate_limiter,
                &self.tx_frame_buf[..read_count],
                &mut self.tap,
                self.guest_mac,
            )
            .unwrap_or_else(|_| false);
            if frame_consumed_by_mmds && !self.rx_deferred_frame {
                // MMDS consumed this frame/request, let's also try to process the response.
                process_rx_for_mmds = true;
            }

            tx_queue
                .packed_add_used(mem, head_desc_index, tail_buf_id, chain_len, 0)
                .map_err(DeviceError::QueueError)?;
            raise_irq = true;
        }

        if raise_irq {
            self.signal_used_queue()?;
        } else {
            METRICS.net.no_tx_avail_buffer.inc();
        }

        // An incoming frame for the MMDS may trigger the transmission of a new message.
        if process_rx_for_mmds {
            self.process_rx()
        } else {
            Ok(())
        }
    }

    /// Updates the parameters for the rate limiters
    pub fn patch_rate_limiters(
        &mut self,
        rx_bytes: BucketUpdate,
        rx_ops: BucketUpdate,
        tx_bytes: BucketUpdate,
        tx_ops: BucketUpdate,
    ) {
        self.rx_rate_limiter.update_buckets(rx_bytes, rx_ops);
        self.tx_rate_limiter.update_buckets(tx_bytes, tx_ops);
    }

    #[cfg(not(test))]
    fn read_tap(&mut self) -> io::Result<usize> {
        self.tap.read(&mut self.rx_frame_buf)
    }

    pub fn process_rx_queue_event(&mut self) {
        error!("process_rx_queue_event");
        METRICS.net.rx_queue_event_count.inc();

        if let Err(e) = self.queue_evts[RX_INDEX].read() {
            // rate limiters present but with _very high_ allowed rate
            error!("Failed to get rx queue event: {:?}", e);
            METRICS.net.event_fails.inc();
        } else {
            // If the limiter is not blocked, resume the receiving of bytes.
            if !self.rx_rate_limiter.is_blocked() {
                self.resume_rx().unwrap_or_else(report_net_event_fail);
            } else {
                METRICS.net.rx_rate_limiter_throttled.inc();
            }
        }
    }

    pub fn process_tap_rx_event(&mut self) {
        let mem = match self.device_state {
            DeviceState::Activated(ref mem) => mem,
            // This should never happen, it's been already validated in the event handler.
            DeviceState::Inactive => unreachable!(),
        };
        METRICS.net.rx_tap_event_count.inc();

        // While there are no available RX queue buffers and there's a deferred_frame
        // don't process any more incoming. Otherwise start processing a frame. In the
        // process the deferred_frame flag will be set in order to avoid freezing the
        // RX queue.
        if self.queues[RX_INDEX].packed_is_empty(mem) && self.rx_deferred_frame {
            error!("process_tap_rx_event: Rx queue is empty");
            METRICS.net.no_rx_avail_buffer.inc();
            return;
        }

        // While limiter is blocked, don't process any more incoming.
        if self.rx_rate_limiter.is_blocked() {
            METRICS.net.rx_rate_limiter_throttled.inc();
            return;
        }

        if self.rx_deferred_frame
        // Process a deferred frame first if available. Don't read from tap again
        // until we manage to receive this deferred frame.
        {
            self.handle_deferred_frame()
                .unwrap_or_else(report_net_event_fail);
        } else {
            self.process_rx().unwrap_or_else(report_net_event_fail);
        }
    }

    pub fn process_tx_queue_event(&mut self) {
        METRICS.net.tx_queue_event_count.inc();
        if let Err(e) = self.queue_evts[TX_INDEX].read() {
            error!("Failed to get tx queue event: {:?}", e);
            METRICS.net.event_fails.inc();
        } else if !self.tx_rate_limiter.is_blocked()
        // If the limiter is not blocked, continue transmitting bytes.
        {
            self.process_tx().unwrap_or_else(report_net_event_fail);
        } else {
            METRICS.net.tx_rate_limiter_throttled.inc();
        }
    }

    pub fn process_rx_rate_limiter_event(&mut self) {
        METRICS.net.rx_event_rate_limiter_count.inc();
        // Upon rate limiter event, call the rate limiter handler
        // and restart processing the queue.

        match self.rx_rate_limiter.event_handler() {
            Ok(_) => {
                // There might be enough budget now to receive the frame.
                self.resume_rx().unwrap_or_else(report_net_event_fail);
            }
            Err(e) => {
                error!("Failed to get rx rate-limiter event: {:?}", e);
                METRICS.net.event_fails.inc();
            }
        }
    }

    pub fn process_tx_rate_limiter_event(&mut self) {
        METRICS.net.tx_rate_limiter_event_count.inc();
        // Upon rate limiter event, call the rate limiter handler
        // and restart processing the queue.
        match self.tx_rate_limiter.event_handler() {
            Ok(_) => {
                // There might be enough budget now to send the frame.
                self.process_tx().unwrap_or_else(report_net_event_fail);
            }
            Err(e) => {
                error!("Failed to get tx rate-limiter event: {:?}", e);
                METRICS.net.event_fails.inc();
            }
        }
    }

    /// Process device virtio queue(s).
    pub fn process_virtio_queues(&mut self) {
        let _ = self.resume_rx();
        let _ = self.process_tx();
    }
}

impl VirtioDevice for Net {
    fn device_type(&self) -> u32 {
        TYPE_NET
    }

    fn queues(&self) -> &[Queue] {
        &self.queues
    }

    fn queues_mut(&mut self) -> &mut [Queue] {
        &mut self.queues
    }

    fn queue_events(&self) -> &[EventFd] {
        &self.queue_evts
    }

    fn interrupt_evt(&self) -> &EventFd {
        &self.interrupt_evt
    }

    fn interrupt_status(&self) -> Arc<AtomicUsize> {
        self.interrupt_status.clone()
    }

    fn avail_features(&self) -> u64 {
        self.avail_features
    }

    fn acked_features(&self) -> u64 {
        self.acked_features
    }

    fn set_acked_features(&mut self, acked_features: u64) {
        self.acked_features = acked_features;
    }

    fn read_config(&self, offset: u64, mut data: &mut [u8]) {
        let config_space_bytes = self.config_space.as_slice();
        let config_len = config_space_bytes.len() as u64;
        if offset >= config_len {
            error!("Failed to read config space");
            METRICS.net.cfg_fails.inc();
            return;
        }
        if let Some(end) = offset.checked_add(data.len() as u64) {
            // This write can't fail, offset and end are checked against config_len.
            data.write_all(
                &config_space_bytes[offset as usize..cmp::min(end, config_len) as usize],
            )
            .unwrap();
        }
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        let data_len = data.len() as u64;
        let config_space_bytes = self.config_space.as_mut_slice();
        let config_len = config_space_bytes.len() as u64;
        if offset + data_len > config_len {
            error!("Failed to write config space");
            METRICS.net.cfg_fails.inc();
            return;
        }

        config_space_bytes[offset as usize..(offset + data_len) as usize].copy_from_slice(data);
        self.guest_mac = Some(MacAddr::from_bytes_unchecked(
            &self.config_space.guest_mac[..MAC_ADDR_LEN],
        ));
        METRICS.net.mac_address_updates.inc();
    }

    fn is_activated(&self) -> bool {
        match self.device_state {
            DeviceState::Inactive => false,
            DeviceState::Activated(_) => true,
        }
    }

    fn activate(&mut self, mem: GuestMemoryMmap) -> ActivateResult {
        if self.activate_evt.write(1).is_err() {
            error!("Net: Cannot write to activate_evt");
            return Err(super::super::ActivateError::BadActivate);
        }
        self.device_state = DeviceState::Activated(mem);
        Ok(())
    }
}
