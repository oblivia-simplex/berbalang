// A Logger needs to asynchronously gather and periodically
// record information on the evolutionary process.

// a hack to make the imports more meaningful
pub use crate::build_observation_mod;

pub trait Observe {
    type Observable;
    type Params;
    type Error;

    type Window: ObservationWindow;

    /// The observe method should take a clone of the observable
    /// and store in something like a sliding observation window.
    fn observe(&self, ob: Self::Observable);

    fn spawn(params: &Self::Params) -> Self;
}

pub trait ObservationWindow {
    type Observable;

    fn insert(&mut self, ob: Self::Observable);

    fn report(&self);
}

// Let's try a horrible macro
// Jackie, I'm sorry.
// TODO: try to replace this with generic programming trick
#[macro_export]
macro_rules! build_observation_mod {
    ($mod_name:ident, $observable:ty, $config: ty) => {
        mod $mod_name {
            use std::sync::mpsc::{channel, SendError, Sender};
            use std::thread::{spawn, JoinHandle};

            use crate::observer::{ObservationWindow, Observe};

            use super::*;

            pub struct Observer {
                pub handle: JoinHandle<()>,
                tx: Sender<$observable>,
            }

            pub struct Window {
                pub frame: Vec<Option<$observable>>,
                i: usize,
                window_size: usize,
            }

            impl ObservationWindow for Window {
                type Observable = $observable;

                fn insert(&mut self, thing: Self::Observable) {
                    self.i = (self.i + 1) % self.window_size;
                    self.frame[self.i] = Some(thing);
                    if self.i == 0 {
                        self.report();
                    }
                }

                fn report(&self) {
                    let fitnesses: Vec<usize> = self
                        .frame
                        .iter()
                        .filter_map(|t| t.as_ref().and_then(<$observable>::fitness))
                        .collect();
                    let avg_fit = fitnesses.iter().sum::<usize>() as f32 / fitnesses.len() as f32;
                    log::info!("Average fitness: {}", avg_fit);
                }
            }

            impl Window {
                pub fn new(window_size: usize) -> Self {
                    assert!(window_size > 0);
                    Self {
                        frame: vec![None; window_size],
                        i: 0,
                        window_size,
                    }
                }
            }

            impl Observe for Observer {
                type Observable = $observable;
                type Params = $config;
                type Error = SendError<Self::Observable>;

                type Window = Window;

                fn observe(&self, ob: Self::Observable) {
                    self.tx.send(ob).expect("tx failure");
                }

                fn spawn(params: &Self::Params) -> Self {
                    let (tx, rx) = channel();

                    let window_size = params.observer_window_size();
                    let handle = spawn(move || {
                        let mut window: Window = Window::new(window_size);

                        for observable in rx {
                            log::debug!("received observable {:?}", observable);
                            window.insert(observable);
                        }
                    });

                    Self { handle, tx }
                }
            }
        }
    };
}
