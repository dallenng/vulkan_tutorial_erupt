use erupt::vk::{Fence, Semaphore};

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SyncObjects<const N: usize> {
    pub current_frame: usize,
    pub images_in_flight: Vec<Option<Fence>>,
    pub in_flight_fences: [Fence; N],
    pub render_finished_semaphores: [Semaphore; N],
    pub image_available_semaphores: [Semaphore; N],
}
