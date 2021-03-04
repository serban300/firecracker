use std::os::unix::io::{AsRawFd, RawFd};

use super::QUEUE_SIZE;
use io_uring::{IoUring};
use utils::eventfd::EventFd;
use vm_memory::{GuestAddress, GuestMemory, GuestMemoryError, GuestMemoryMmap};

#[derive(Debug)]
pub enum Error {
    GuestMemory(GuestMemoryError),
    IOError(std::io::Error),
    FullSq,
}

pub struct IoUringUserData<T> {
    iovec: [libc::iovec; 1],
    user_data: T,
}

pub struct IoUringTransferEngine {
    ring: io_uring::IoUring,
    completion_evt: EventFd,
    unsubmitted: u32,
    unprocessed: u32,
}

impl IoUringTransferEngine {
    pub fn new(fd: RawFd) -> std::io::Result<IoUringTransferEngine> {
        let ring = IoUring::new(QUEUE_SIZE as u32)?;
        let completion_evt = EventFd::new(libc::EFD_NONBLOCK)?;
        ring.submitter()
            .register_eventfd(completion_evt.as_raw_fd())?;

        ring.submitter().register_files(&[fd])?;

        Ok(IoUringTransferEngine {
            ring,
            completion_evt,
            unsubmitted: 0,
            unprocessed: 0,
        })
    }

    pub fn completion_evt(&self) -> &EventFd {
        &self.completion_evt
    }

    fn push_sqe(&mut self, sqe: io_uring::squeue::Entry) -> Result<(), io_uring::squeue::Entry> {
        {
            let mut sq = self.ring.submission().available();
            unsafe {
                sq.push(sqe)?;
            }
            sq.sync();
        }
        self.unsubmitted += 1;
        Ok(())
    }

    pub fn push_read<T>(
        &mut self,
        fd: RawFd,
        offset: i64,
        mem: &GuestMemoryMmap,
        addr: GuestAddress,
        len: u32,
        user_data: T,
    ) -> Result<(), Error> {
        let buf = mem
            .get_slice(addr, len as usize)
            .map_err(Error::GuestMemory)?
            .as_ptr();

        let boxed_user_data = Box::new(IoUringUserData {
            iovec: [libc::iovec {
                iov_base: buf as *mut _,
                iov_len: len as usize,
            }],
            user_data: user_data,
        });

        let sqe = io_uring::opcode::Readv::new(
            io_uring::opcode::types::Fixed(0),
            boxed_user_data.iovec.as_ptr(),
            1,
        )
        .offset(offset)
        .build()
        .user_data(Box::into_raw(boxed_user_data) as u64);

        self.push_sqe(sqe).map_err(|_| Error::FullSq)
    }

    pub fn push_write<T>(
        &mut self,
        fd: RawFd,
        offset: i64,
        mem: &GuestMemoryMmap,
        addr: GuestAddress,
        len: u32,
        user_data: T,
    ) -> Result<(), Error> {
        let buf = mem
            .get_slice(addr, len as usize)
            .map_err(Error::GuestMemory)?
            .as_ptr();

        let boxed_user_data = Box::new(IoUringUserData {
            iovec: [libc::iovec {
                iov_base: buf as *mut _,
                iov_len: len as usize,
            }],
            user_data: user_data,
        });

        let sqe = io_uring::opcode::Writev::new(
            io_uring::opcode::types::Fixed(0),
            boxed_user_data.iovec.as_ptr(),
            1,
        )
        .offset(offset)
        .build()
        .user_data(Box::into_raw(boxed_user_data) as u64);

        self.push_sqe(sqe).map_err(|_| Error::FullSq)
    }

    pub fn submit(&mut self) -> Result<(), Error> {
        if self.unsubmitted > 0 {
            self.ring.submit().map_err(Error::IOError)?;
            self.unprocessed += self.unsubmitted;
            self.unsubmitted = 0;
        }

        Ok(())
    }

    pub fn pop_cqe<T>(&mut self) -> Option<Result<(u32, T), (std::io::Error, T)>> {
        let cqe = {
            let mut cq = self.ring.completion().available();
            let cqe = cq.next();
            cq.sync();
            cqe
        };

        cqe.map_or(None, |cqe| {
            let user_data = unsafe { Box::from_raw(cqe.user_data() as *mut IoUringUserData<T>) };

            let ret = cqe.result();
            let res = if ret < 0 {
                Err((
                    std::io::Error::from_raw_os_error(cqe.result()),
                    (*user_data).user_data,
                ))
            } else {
                Ok((ret as u32, (*user_data).user_data))
            };

            self.unprocessed -= 1;
            Some(res)
        })
    }

    pub fn unprocessed(&self) -> u32 {
        self.unprocessed
    }
}
