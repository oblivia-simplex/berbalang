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

        impl ::non_dominated_sort::DominanceOrd<$phenome> for $ord {
            fn dominance_ord(&self, a: &$phenome, b: &$phenome) -> std::cmp::Ordering {
                a.fitness()
                    .partial_cmp(&b.fitness())
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        impl<'a> ::non_dominated_sort::DominanceOrd<&'a $phenome> for $ord {}
    };
}

#[macro_export]
macro_rules! hashmap {
    ($($key:expr => $val:expr$(,)?)*) => {
        {
            let mut map = ::hashbrown::HashMap::new();
            $(
                let _ = map.insert($key.into(), $val);
            )*
            map
        }
    }

}

#[macro_export]
macro_rules! pareto {
    ($($key:expr => $val:expr, $(,)?)*) => {
        Pareto(hashmap!{$( $key => $val, )*})
    }
}

#[macro_export]
macro_rules! lexical {
    ($($v:expr $(,)?)*) => {
        vec![$( $v, )*]
    }
}
