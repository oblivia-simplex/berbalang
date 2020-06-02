#[macro_export]
macro_rules! make_phenome_heap_friendly {
    ($phenome:ty) => {
        impl PartialEq for $phenome {
            fn eq(&self, other: &Self) -> bool {
                self.tag == other.tag
            }
        }

        impl Eq for $phenome {}

        impl PartialOrd for $phenome {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.tag.partial_cmp(&other.tag)
            }
        }

        impl Ord for $phenome {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_dominance_ord_for_phenome {
    ($phenome:ty, $ord:ident) => {
        #[derive(Clone, Debug, Copy)]
        pub struct $ord;

        impl ::non_dominated_sort::DominanceOrd for $ord {
            type T = $phenome;

            fn dominance_ord(&self, a: &Self::T, b: &Self::T) -> std::cmp::Ordering {
                a.fitness()
                    .partial_cmp(&b.fitness())
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }
    };
}
