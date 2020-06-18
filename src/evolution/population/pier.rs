use std::sync::atomic::{self, AtomicUsize};

use crossbeam::queue::SegQueue;

pub struct Pier<P> {
    capacity: usize,
    count: AtomicUsize,
    q: SegQueue<P>,
}

impl<P> Pier<P> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            count: AtomicUsize::new(0),
            q: SegQueue::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.count.load(atomic::Ordering::SeqCst)
    }

    fn incr_count(&self) -> usize {
        self.count.fetch_add(1, atomic::Ordering::SeqCst)
    }

    fn decr_count(&self) -> usize {
        self.count.fetch_sub(1, atomic::Ordering::SeqCst)
    }

    pub fn embark(&self, emigrant: P) -> Result<(), P> {
        if self.len() >= self.capacity {
            log::info!("Pier at capacity, returning emigrant");
            return Err(emigrant);
        }
        self.q.push(emigrant);
        let len = self.incr_count();
        log::info!("Emigrant embarked onto pier. Holding {}", len + 1);
        Ok(())
    }

    pub fn disembark(&self) -> Option<P> {
        if let Some(p) = self.q.pop().ok() {
            let len = self.decr_count();
            log::info!("Immigrant disembarked from pier. Holding {}", len - 1);
            Some(p)
        } else {
            None
        }
    }
}
