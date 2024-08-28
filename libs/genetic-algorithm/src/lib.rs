use rand::{seq::SliceRandom, Rng, RngCore};

pub struct GeneticAlgorithm<S> {
    selection_method: S,
}

impl<S: SelectionMethod> GeneticAlgorithm<S> {
    pub fn new(selection_method: S) -> Self {
        Self { selection_method }
    }
    pub fn evolve<I: Individual>(&self, rng: &mut dyn RngCore, population: &[I]) -> Vec<I> {
        assert!(!population.is_empty());
        (0..population.len())
            .map(|_| {
                let parent_a = self.selection_method.select(rng, population);
                let parent_b = self.selection_method.select(rng, population);
            })
            .collect()
    }
}

pub trait Individual {
    fn fitness(&self) -> f32;
}

pub trait SelectionMethod {
    fn select<'a, I: Individual>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I;
}

/// 转盘比例选择算法
pub struct RouletteWheelSelection;

impl SelectionMethod for RouletteWheelSelection {
    fn select<'a, I: Individual>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I {
        // let total_fitness = population.iter().map(|x| x.fitness()).sum::<f32>();
        // loop {
        //     let indv = population.choose(rng).expect("got an empty population");
        //     let indv_share = indv.fitness() / total_fitness;
        //     if rng.gen_bool(indv_share as f64) {
        //         return indv;
        //     }
        // }

        population
            .choose_weighted(rng, |x| x.fitness())
            .expect("got an empty population")
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;
    use std::iter::FromIterator;

    use super::*;

    #[derive(Clone, Debug)]
    struct TestIndividual {
        fitness: f32,
    }

    impl TestIndividual {
        fn new(fitness: f32) -> Self {
            Self { fitness }
        }
    }

    impl Individual for TestIndividual {
        fn fitness(&self) -> f32 {
            self.fitness
        }
    }
    #[test]
    fn roulette_wheel_selection() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let population = vec![
            TestIndividual::new(2.0),
            TestIndividual::new(1.0),
            TestIndividual::new(4.0),
            TestIndividual::new(3.0),
        ];
        let mut actual_histogram = BTreeMap::new();
        for _ in 0..1000 {
            let fitness = RouletteWheelSelection
                .select(&mut rng, &population)
                .fitness() as i32;
            *actual_histogram.entry(fitness).or_insert(0) += 1;
        }

        let expected_histogram = BTreeMap::from_iter([
            // (fitness, how many times this fitness has been chosen)
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
        ]);
        assert_eq!(actual_histogram, expected_histogram);
    }
}
